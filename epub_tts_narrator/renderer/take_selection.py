"""Audited user selection among the retained takes of one synthesis request."""

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import tempfile

from .audio_naming import MANUAL_SELECTION_ROLE, speech_take_filename
from .common import digest, file_hash, read_json, regular_file, write_json


_HEX = re.compile(r"[0-9a-f]{64}")
_ATTEMPT = re.compile(r"attempt-(\d+)\.json")
_EVIDENCE = re.compile(r"manual-selection-[0-9a-f]{12}\.json")


def _clip_directory(clips_directory, request_hash):
    if not isinstance(request_hash, str) or _HEX.fullmatch(request_hash) is None:
        raise ValueError("clip directory missing/invalid")
    clips = Path(clips_directory)
    directory = clips / request_hash
    if clips.is_symlink() or directory.is_symlink() or not directory.is_dir():
        raise ValueError("clip directory missing/invalid")
    return directory


def _validate_note(note):
    if not isinstance(note, str) or len(note) > 240:
        raise ValueError("note must be a string of at most 240 characters")
    return note


def _accepted(directory):
    path = directory / "accepted.json"
    if not path.exists() and not path.is_symlink():
        return None
    record = read_json(regular_file(path))
    if not isinstance(record, dict):
        raise ValueError("accepted.json must contain an acceptance record")
    return record


def _attempts(directory, request_hash):
    attempts = []
    seen = set()
    for path in sorted(directory.glob("attempt-*.json")):
        match = _ATTEMPT.fullmatch(path.name)
        if match is None:
            continue
        number = int(match.group(1))
        if number < 1 or number in seen:
            raise ValueError(f"Invalid or duplicate attempt number in {path.name}")
        seen.add(number)
        record = read_json(regular_file(path))
        if not isinstance(record, dict):
            raise ValueError(f"attempt {number} record must be an object")
        request = record.get("request")
        if (not isinstance(request, dict) or record.get("request_hash") != request_hash
                or digest(request) != request_hash):
            raise ValueError(f"attempt {number} request binding does not match request_hash")
        if record.get("attempt") != number:
            raise ValueError(f"attempt {number} record has a different attempt field")
        name = record.get("audio_file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
            raise ValueError(f"attempt {number} audio_file must be a WAV basename")
        report = record.get("report")
        if not isinstance(report, dict):
            raise ValueError(f"attempt {number} report must be an object")
        reasons = report.get("reasons", [])
        if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            raise ValueError(f"attempt {number} report.reasons must be strings")
        method = report.get("text_confirmation_method")
        if method is not None and not isinstance(method, str):
            raise ValueError(f"attempt {number} report.text_confirmation_method must be a string or null")
        transcript = report.get("transcript")
        if transcript is not None and not isinstance(transcript, str):
            raise ValueError(f"attempt {number} report.transcript must be a string or null")
        attempts.append((number, record, directory / name))
    return sorted(attempts, key=lambda item: item[0])


def _take_row(number, record, audio, accepted):
    from .exhaustion_candidates import physical_audio
    from .history_echo import candidate_echo
    from .clip_repair_contract import logical_text

    report = record["report"]
    physical = None
    ineligible_reason = None
    if not audio.exists() and not audio.is_symlink():
        ineligible_reason = "audio file missing"
    elif audio.is_symlink() or not audio.is_file():
        ineligible_reason = "audio file is not a regular nonsymlink file"
    else:
        if file_hash(audio) != record.get("audio_sha256"):
            ineligible_reason = "audio_sha256 does not match audio bytes"
        physical = physical_audio(audio)
        if ineligible_reason is None and physical is None:
            ineligible_reason = "physical_audio rejected corrupt, empty, silent, or invalid PCM16 audio"
    echo = candidate_echo(record, logical_text(record["request"])) is not None
    if ineligible_reason is None and echo:
        ineligible_reason = "history echo"
    return {
        "attempt": number,
        "record": record,
        "audio_path": audio,
        "passed": report.get("passed") is True,
        "reasons": list(report.get("reasons", [])),
        "text_confirmation_method": report.get("text_confirmation_method"),
        "duration_seconds": physical["duration_seconds"] if physical is not None else None,
        "transcript": report.get("transcript") or "",
        "echo": echo,
        "eligible": ineligible_reason is None,
        "ineligible_reason": ineligible_reason,
        "accepted": (accepted is not None
                     and accepted.get("audio_sha256") == record.get("audio_sha256")
                     and accepted.get("audio_file") == record.get("audio_file")),
    }


def list_takes(clips_directory, request_hash) -> list[dict]:
    """List every retained attempt with its current acceptance and eligibility."""
    directory = _clip_directory(clips_directory, request_hash)
    accepted = _accepted(directory)
    return [_take_row(number, record, audio, accepted)
            for number, record, audio in _attempts(directory, request_hash)]


def _policy(note):
    return {"name": "manual_take_selection", "version": 1,
            "eligibility": "bound_full_span_physical_pcm16_v1",
            "selection": "user_decisive_audited", "note": note}


def _validate_manual_marker(request, marker):
    if not isinstance(marker, dict) or set(marker) != {"request", "audit", "selected_record"}:
        raise ValueError("Invalid manual_selection authority")
    selected = marker["selected_record"]
    audit = marker["audit"]
    if not isinstance(request, dict) or marker["request"] != request:
        raise ValueError("manual_selection.request does not match the accepted request")
    if not isinstance(selected, dict) or selected.get("request") != request:
        raise ValueError("manual_selection.selected_record request binding changed")
    if selected.get("request_hash") != digest(request):
        raise ValueError("manual_selection.selected_record.request_hash is invalid")
    number = selected.get("attempt")
    if type(number) is not int or number < 1:
        raise ValueError("manual_selection selected attempt must be a positive integer")
    audio_sha256 = selected.get("audio_sha256")
    if not isinstance(audio_sha256, str) or _HEX.fullmatch(audio_sha256) is None:
        raise ValueError("manual_selection selected audio_sha256 is invalid")
    name = selected.get("audio_file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
        raise ValueError("manual_selection selected audio_file is invalid")
    selected_metadata = selected.get("engine_metadata")
    if (not isinstance(selected_metadata, dict)
            or "manual_selection" in selected_metadata or "exhaustion_selection" in selected_metadata):
        raise ValueError("manual_selection must reference an original attempt")
    selected_report = selected.get("report")
    if not isinstance(selected_report, dict) or selected_report.get("passed") is True:
        raise ValueError("manual_selection.selected_record.report must be a failed report")
    if not isinstance(audit, dict) or set(audit) != {"identity", "choice"}:
        raise ValueError("Invalid manual_selection audit")
    identity, choice = audit["identity"], audit["choice"]
    if not isinstance(identity, dict):
        raise ValueError("manual_selection audit identity must be an object")
    note = identity.get("note")
    _validate_note(note)
    if (identity != _policy(note) or not isinstance(choice, dict)
            or set(choice) != {"attempt", "note", "audio_sha256"}
            or choice != {"attempt": number, "note": note, "audio_sha256": audio_sha256}):
        raise ValueError("manual_selection policy or user choice changed")
    from .exhaustion_selection import validate_record
    validate_record(selected)
    return selected


def validate_manual_selection(record):
    """Fail closed unless a complete manual-basis acceptance matches its proof."""
    if not isinstance(record, dict):
        raise ValueError("Manual acceptance must be an object")
    request = record.get("request")
    if not isinstance(request, dict):
        raise ValueError("Manual acceptance request must be an object")
    metadata = record.get("engine_metadata")
    if not isinstance(metadata, dict) or "manual_selection" not in metadata:
        raise ValueError("Manual acceptance lacks engine_metadata.manual_selection")
    marker = metadata["manual_selection"]
    selected = _validate_manual_marker(request, marker)
    report = record.get("report")
    if not isinstance(report, dict):
        raise ValueError("Manual acceptance report must be an object")
    from .clip_repair_contract import logical_text
    identity = marker["audit"]["identity"]
    if (record.get("request_hash") != digest(request)
            or record.get("attempt") != selected["attempt"]
            or record.get("audio_sha256") != selected["audio_sha256"]
            or record.get("audio_file") != speech_take_filename(
                request, selected["attempt"], role=MANUAL_SELECTION_ROLE)
            or record.get("validation_key") != digest(
                {"selection": marker, "audio": selected["audio_sha256"]})
            or {key: value for key, value in metadata.items() if key != "manual_selection"}
                != selected.get("engine_metadata", {})
            or report.get("passed") is not True or report.get("reasons") != []
            or report.get("expected_text") != logical_text(request)
            or report.get("text_confirmation_method") != "manual_take_selection"
            or report.get("acceptance_policy") != identity
            or report.get("ordinary_verification_passed") is not False
            or report.get("original_failed_report") != selected["report"]):
        raise ValueError("Manual acceptance differs from its audited take selection")
    return selected


def selected_generation_identity(request, voices, metadata):
    """Treat the audited user choice as decisive without replaying lexical vetoes."""
    _validate_manual_marker(request, metadata["manual_selection"])
    selected = metadata["manual_selection"]["selected_record"]
    if ({key: value for key, value in metadata.items() if key != "manual_selection"}
            != selected.get("engine_metadata", {})):
        raise ValueError("Selected synthesis metadata changed")
    from .speech_identity import generation_identity
    return generation_identity(request, voices)


def _manual_record(row, note):
    from .clip_repair_contract import logical_text

    selected = row["record"]
    request = selected["request"]
    identity = _policy(note)
    choice = {"attempt": row["attempt"], "note": note,
              "audio_sha256": selected["audio_sha256"]}
    marker = {"request": deepcopy(request),
              "audit": {"identity": deepcopy(identity), "choice": deepcopy(choice)},
              "selected_record": deepcopy(selected)}
    report = deepcopy(selected["report"])
    record = deepcopy(selected)
    record.update(
        request=deepcopy(request), request_hash=digest(request),
        audio_file=speech_take_filename(request, row["attempt"], role=MANUAL_SELECTION_ROLE),
        validation_key=digest({"selection": marker, "audio": selected["audio_sha256"]}),
        report={**report, "passed": True, "reasons": [], "expected_text": logical_text(request),
                "text_confirmation_method": "manual_take_selection",
                "acceptance_policy": deepcopy(identity), "ordinary_verification_passed": False,
                "original_failed_report": deepcopy(report)},
        engine_metadata={**selected.get("engine_metadata", {}), "manual_selection": marker})
    validate_manual_selection(record)
    return record


def _previous_from_evidence(directory, record, note):
    choice = {"attempt": record["attempt"], "note": note,
              "audio_sha256": record["audio_sha256"]}
    matches = []
    for path in directory.glob("manual-selection-*.json"):
        if _EVIDENCE.fullmatch(path.name) is None:
            continue
        evidence = read_json(regular_file(path))
        if not isinstance(evidence, dict):
            raise ValueError(f"Invalid manual-selection evidence: {path.name}")
        if evidence.get("new_choice") == choice:
            previous = evidence.get("previous_accepted")
            previous_hash = previous.get("audio_sha256", "") if isinstance(previous, dict) else ""
            matches.append((evidence.get("at", ""), path.name, previous_hash))
    return max(matches)[2] if matches else None


def _result(directory, previous, record, basis, note):
    previous_hash = previous.get("audio_sha256", "") if isinstance(previous, dict) else ""
    if previous == record:
        evidenced = _previous_from_evidence(directory, record, note)
        if evidenced is not None:
            previous_hash = evidenced
    return {"previous_audio_sha256": previous_hash,
            "accepted_audio_sha256": record["audio_sha256"],
            "attempt": record["attempt"], "basis": basis, "audio_file": record["audio_file"]}


def _publish_audio(directory, source, target, audio_sha256):
    if target.exists() or target.is_symlink():
        if file_hash(regular_file(target)) != audio_sha256:
            raise ValueError("Existing manual-selection audio differs from audio_sha256")
        return
    with tempfile.TemporaryDirectory(prefix=".manual-selection-", dir=directory) as staging:
        copied = Path(staging) / target.name
        shutil.copyfile(regular_file(source), copied)
        if file_hash(copied) != audio_sha256:
            raise ValueError("Selected attempt audio changed during publication")
        os.replace(copied, target)


def _archive_evidence(directory, previous, choice):
    evidence = {"previous_accepted": deepcopy(previous), "new_choice": deepcopy(choice),
                "at": datetime.now(timezone.utc).isoformat()}
    suffix = digest(evidence)[:12]
    archived = directory / f"accepted.superseded-{suffix}.json"
    evidence_path = directory / f"manual-selection-{suffix}.json"
    for path in (archived, evidence_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Manual-selection archive already exists: {path.name}")
    return evidence, archived, evidence_path


def select_take(clips_directory, request_hash, attempt, *, note="") -> dict:
    """Accept one eligible retained take, archiving rather than deleting prior state."""
    _validate_note(note)
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    directory = _clip_directory(clips_directory, request_hash)
    previous = _accepted(directory)
    rows = list_takes(clips_directory, request_hash)
    row = next((item for item in rows if item["attempt"] == attempt), None)
    if row is None:
        raise ValueError(f"attempt {attempt} does not exist")
    if not row["eligible"]:
        reason = row["ineligible_reason"]
        if reason == "history echo":
            raise ValueError(f"attempt {attempt} is a history echo and cannot be accepted")
        raise ValueError(f"attempt {attempt} is ineligible: {reason}")

    ordinary = row["passed"]
    record = deepcopy(row["record"]) if ordinary else _manual_record(row, note)
    basis = "ordinary" if ordinary else "manual_take_selection"
    target = row["audio_path"] if ordinary else directory / record["audio_file"]
    if previous == record and target.is_file() and not target.is_symlink() \
            and file_hash(target) == record["audio_sha256"]:
        if ordinary:
            from .exhaustion_selection import validate_record
            validate_record(record)
        else:
            validate_manual_selection(record)
        return _result(directory, previous, record, basis, note)

    choice = {"attempt": attempt, "note": note, "audio_sha256": record["audio_sha256"]}
    evidence, archived, evidence_path = _archive_evidence(directory, previous, choice)
    accepted_path = directory / "accepted.json"
    if previous is not None and not ordinary and target == row["audio_path"]:
        raise ValueError("manual-selection audio_file must differ from the attempt audio_file")
    if not ordinary:
        _publish_audio(directory, row["audio_path"], target, record["audio_sha256"])
    if previous is not None:
        accepted_path.rename(archived)
    write_json(evidence_path, evidence)
    write_json(accepted_path, record)
    saved = read_json(regular_file(accepted_path))
    if saved != record:
        raise ValueError("accepted.json changed during manual take selection")
    if ordinary:
        from .exhaustion_selection import validate_record
        validate_record(saved)
    else:
        validate_manual_selection(saved)
    if file_hash(regular_file(target)) != saved["audio_sha256"]:
        raise ValueError("Accepted audio changed during manual take selection")
    return _result(directory, previous, saved, basis, note)
