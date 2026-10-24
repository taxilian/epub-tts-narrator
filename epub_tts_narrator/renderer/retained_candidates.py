"""Fresh, bounded review of retained complete candidates before more synthesis."""

from copy import deepcopy

from .common import digest, file_hash, read_json, regular_file, write_json
from .quality import text_confirmation_unavailable, validate_audio, validation_key
from .recurring_resolution import accepted_report, confirm_rewritten_report, failed_take_context
from .speech_resolver import _accept_pairs, validate_resolution


def recheck_retained_candidates(session, verifier, config, *, notify):
    """Use saved transcripts only to nominate WAVs for fresh recognition and review."""
    from .interrupts import stop_if_requested
    from .workflow import RenderBlocked, VerificationUnavailable

    records, directory, text = session.records, session.directory, session.text
    if len(records) < 2:
        return None
    history = session.state["history"]
    last = history[-1]["audit"] if history else {}
    proposal = last.get("proposal")
    if (last.get("decision") != "resolved" or not isinstance(proposal, dict)
            or proposal.get("action") != "stop"):
        return None
    latest = records[max(records)]["report"]
    if (latest.get("reasons") != ["text_mismatch"]
            or _accept_pairs(text, latest.get("transcript", "")) is not None):
        return None
    checked = 0
    for number in sorted(records)[-30:-1][::-1]:
        saved = records[number]
        report = saved.get("report", {})
        if (report.get("reasons") != ["text_mismatch"]
                or _accept_pairs(text, report.get("transcript", "")) is None):
            continue
        if any(row.get("audio_sha256") == saved.get("audio_sha256")
               and row["audit"].get("error_code") == "sound_review_rejected"
               for row in session.state["history"]):
            continue
        if checked >= 3 or session.rounds >= session.maximum_rounds:
            break
        stop_if_requested()
        name = saved.get("audio_file")
        if (name != f"attempt-{number:03d}.wav" or saved.get("request") != session.request
                or saved.get("request_hash") != digest(session.request)):
            raise RenderBlocked("Retained speech candidate does not match its original request")
        audio = directory / name
        if not audio.exists() and not audio.is_symlink():
            continue
        if file_hash(regular_file(audio)) != saved.get("audio_sha256"):
            raise RenderBlocked("Retained speech candidate changed on disk")
        checked += 1
        notify(f"  Rechecking retained candidate {number} with complete reviewable wording; no new synthesis.")
        record = deepcopy(saved)
        record["report"] = validate_audio(audio, text, verifier,
                                         {**config["quality"], "max_wer": 0.0}, purpose="speech")
        marker = record.get("engine_metadata", {}).get("speech_resolution")
        if marker is not None:
            record["report"] = confirm_rewritten_report(record["report"], marker)
        key = validation_key(text, verifier, config["quality"], purpose="speech")
        if record["report"].get("passed") is True:
            record["validation_key"] = key
            return record
        if text_confirmation_unavailable(record["report"]):
            raise VerificationUnavailable("Retained candidate confirmation is unavailable; its audio is preserved")
        if (record["report"].get("reasons") != ["text_mismatch"]
                or _accept_pairs(text, record["report"].get("transcript", "")) is None):
            continue
        failure = failed_take_context(text, record)
        fingerprint = digest({"request_hash": saved["request_hash"], "audio_sha256": saved["audio_sha256"],
                              "identity": session.resolver.identity, "validation_key": key, "failure": failure})
        path = directory / f"retained-review-{number:03d}.json"
        audit = None
        if path.exists() or path.is_symlink():
            memo = read_json(regular_file(path))
            if not isinstance(memo, dict):
                raise RenderBlocked("Invalid retained candidate review")
            checksum = memo.pop("record_sha256", None)
            if checksum != digest(memo):
                raise RenderBlocked("Retained candidate review integrity check failed")
            if memo.get("fingerprint") == fingerprint:
                audit = memo.get("audit")
        if not isinstance(audit, dict) or audit.get("decision") == "unavailable":
            session.rounds += 1
            audit = session.resolver.resolve(text, [failure], can_rewrite=False, can_split=False,
                                             can_plain_delivery=False)
            memo = {"fingerprint": fingerprint, "audit": audit}
            write_json(path, {**memo, "record_sha256": digest(memo)})
        if audit.get("decision") == "unavailable":
            raise VerificationUnavailable("Retained candidate sound review is unavailable; its audio is preserved")
        proposal = validate_resolution(audit, text, [failure], session.resolver.identity,
                                       can_rewrite=False, can_split=False, can_plain_delivery=False)
        if proposal is not None and proposal["action"] == "accept":
            record["report"] = accepted_report(record["report"], {
                **proposal, "audit": audit, "failures": [failure], "origin": "retained_candidate"})
            record["validation_key"] = key
            return record
    return None
