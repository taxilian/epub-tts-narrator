#!/usr/bin/env python3
"""Evaluate recurring-speech resolution at the configured analysis endpoint.

Sends six small synthetic text cases, optionally plus already saved failure
transcripts. No audio is uploaded, decoded, generated, or changed. Saved-audio
hashes are checked locally; acoustic results and ASR transcripts remain the
original saved evidence. Full model prompts, proposals, and votes are saved.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.config import load_config
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, regular_file, write_json
from epub_tts_narrator.renderer.delivery_recovery import plain_delivery_request
from epub_tts_narrator.renderer.quality import compare_words
from epub_tts_narrator.renderer.recurring_resolution import failed_take_context
from epub_tts_narrator.renderer.speech_resolver import validate_resolution
from epub_tts_narrator.renderer.verification import create_speech_resolver


CASES = (
    ("name_recognition", "Derrins opened the door and waited for the others.",
     "Darren's opened the door and waited for the others.", ["text_mismatch"], {"accept"}),
    ("terminal_s", "We checked the sensors and weapons systems before leaving.",
     "We checked the sensors and weapon systems before leaving.", ["text_mismatch"], {"accept"}),
    ("missing_negation_and_word", "Mara did not open the old door.",
     "Mara did open the door.", ["text_mismatch"], {"retry", "split", "stop"}),
    ("silence", "Mara waited beside the door.", "", ["silent_audio"], {"retry", "split", "stop"}),
    ("unrelated_word", "Mara opened the gate.", "Mara closed the gate.",
     ["text_mismatch"], {"retry", "split", "stop"}),
    ("extra_and_repeated_words", "Mara opened the gate.", "Mara opened opened the gate. Thank you.",
     ["text_mismatch"], {"retry", "split", "stop"}),
)


def synthetic_case(case):
    name, expected, heard, reasons, actions = case
    failures = [failed_take_context(expected, {
                    "attempt": attempt, "audio_sha256": digest(f"synthetic:{name}:{attempt}"),
                    "report": {"transcript": heard, "reasons": list(reasons),
                               "metrics": compare_words(expected, heard) if heard else {"rms_dbfs": None}}})
                for attempt in range(1, 4)]
    return {"case": name, "origin": "synthetic", "expected_text": expected,
            "failures": failures, "expected_actions": sorted(actions), "can_plain_delivery": False}


def saved_case(directory):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--attempt-directory must be a directory")
    records = []
    for path in directory.iterdir():
        match = re.fullmatch(r"attempt-([0-9]{3,})\.json", path.name)
        if not match:
            continue
        number = int(match[1])
        if number < 1 or match[1] != f"{number:03d}":
            continue
        record = read_json(regular_file(path))
        report = record.get("report", {})
        if report.get("passed") is True:
            continue
        reasons = report.get("reasons")
        if (not isinstance(reasons, list) or not reasons
                or any(reason not in {"text_mismatch", "silent_audio", "audio_too_quiet",
                                      "excessive_silence", "internal_silence_stall", "excessive_clipping",
                                      "audio_too_short", "audio_too_long"} for reason in reasons)):
            continue
        expected, heard = report.get("expected_text"), report.get("transcript")
        if not isinstance(expected, str) or (heard is not None and not isinstance(heard, str)):
            raise ValueError(f"Missing saved text evidence in {path.name}")
        if "text_mismatch" in reasons and not isinstance(heard, str):
            raise ValueError(f"Missing saved transcript in {path.name}")
        if record.get("attempt") != number:
            raise ValueError(f"Attempt number mismatch in {path.name}")
        name = record.get("audio_file")
        if not isinstance(name, str) or name != f"attempt-{number:03d}.wav":
            raise ValueError(f"Invalid audio reference in {path.name}")
        audio = regular_file(directory / name)
        if file_hash(audio) != record.get("audio_sha256"):
            raise ValueError(f"Saved audio hash mismatch in {path.name}")
        request = record.get("request")
        bound_hash = None
        if request is not None:
            if not isinstance(request, dict) or record.get("request_hash") != digest(request):
                raise ValueError(f"Saved synthesis request hash mismatch in {path.name}")
            part = request.get("kind") == "speech_recovery_part"
            source = request.get("part" if part else "chunk")
            requested_text = source.get("text" if part else "spoken_text") if isinstance(source, dict) else None
            if requested_text != expected:
                raise ValueError(f"Saved request text differs from its verification text in {path.name}")
            bound_hash = record["request_hash"]
        failure = failed_take_context(expected, record)
        can_plain = request is not None and plain_delivery_request(request) is not None
        records.append((number, expected, failure, bound_hash, can_plain))
    records.sort(key=lambda row: row[0])
    records = records[-30:]
    if not records:
        raise ValueError("No failed speech takes with saved audio and verification evidence were found")
    expected = records[0][1]
    if any(row[1] != expected for row in records):
        raise ValueError("Saved takes do not share one exact expected passage")
    if len({row[3] for row in records}) != 1:
        raise ValueError("Saved takes do not share one bound synthesis request")
    return {"case": "saved_failures", "origin": str(directory), "expected_text": expected,
            "failures": [row[2] for row in records], "expected_actions": None,
            "can_plain_delivery": all(row[4] for row in records)}


def evaluate(case, provider):
    resolver = create_speech_resolver({}, provider)
    if resolver is None:
        raise ValueError("The configured provider could not create a speech resolver")
    can_plain = case.get("can_plain_delivery", False)
    audit = resolver.resolve(case["expected_text"], case["failures"], can_plain_delivery=can_plain)
    proposal = validate_resolution(audit, case["expected_text"], case["failures"], resolver.identity,
                                   can_plain_delivery=can_plain)
    action = proposal["action"] if proposal else None
    votes = [[answer["plausible"] for answer in review.get("response", {}).get("pairs", [])]
             for review in audit["reviews"] if isinstance(review.get("response"), dict)
             and isinstance(review["response"].get("pairs"), list)]
    expected_actions = case["expected_actions"]
    # Saved failures have no human ground-truth action label. Their evaluation
    # establishes that a proposal is valid, not that the rejected audio is good.
    correct = proposal is not None if expected_actions is None else action in expected_actions
    return {**case, "action": action, "correct": correct,
            "evaluation_basis": "valid_proposal" if expected_actions is None else "expected_action",
            "decision": audit["decision"], "votes": votes, "validated_proposal": proposal,
            "audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Global TOML containing [provider]")
    parser.add_argument("--output", type=Path, required=True, help="Directory for complete audit JSON files")
    parser.add_argument("--attempt-directory", type=Path,
                        help="Optional saved failure directory; requires expected text to agree")
    args = parser.parse_args()
    provider = load_config(args.config)["provider"]
    if not provider.get("model"):
        parser.error("Configure [provider].model before running this live evaluation")
    cases = [synthetic_case(case) for case in CASES]
    if args.attempt_directory:
        try:
            cases.append(saved_case(args.attempt_directory))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    # Each resolver makes its planner and sound calls sequentially; this bounds
    # actual concurrent provider requests to two, including optional saved data.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(evaluate, case, provider): case for case in cases}
        for future in as_completed(futures):
            row = future.result()
            write_json(args.output / f"{row['case']}.json", row)
            results.append(row)
            print(f"{row['case']}: {'PASS' if row['correct'] else 'FAIL'} "
                  f"(decision={row['decision']}, action={row['action']}, "
                  f"votes={json.dumps(row['votes'])})", flush=True)
    summary = [{key: row[key] for key in ("case", "origin", "expected_actions", "evaluation_basis",
                                         "action", "correct", "decision", "votes")}
               for row in sorted(results, key=lambda row: row["case"])]
    write_json(args.output / "summary.json", summary)
    print(f"Saved {len(results)} full audits to {args.output}", flush=True)
    return 0 if all(row["correct"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
