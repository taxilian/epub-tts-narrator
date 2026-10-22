#!/usr/bin/env python3
"""Live synthetic regression evaluation; sends text to the configured [provider]."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.config import load_config
from epub_tts_narrator.renderer.phonetic_review import validate_review
from epub_tts_narrator.renderer.verification import create_phonetic_reviewer


# Original synthetic text only: no book passages, recordings, or credentials.
CASES = [
    ("name_spelling", "Derrins spoke quietly.", "Darins spoke quietly.", True),
    ("name_possessive", "Derrins spoke quietly.", "Darren's spoke quietly.", True),
    ("name_possessive_2", "Derrins spoke quietly.", "Darin's spoke quietly.", True),
    ("unseen_name", "Maelin spoke quietly.", "Maylin spoke quietly.", True),
    ("unseen_name_2", "Kaelen spoke quietly.", "Kalen spoke quietly.", True),
    ("unseen_name_3", "Nerys spoke quietly.", "Nerris spoke quietly.", True),
    ("homophone", "He saw a flower.", "He saw a flour.", True),
    ("homophone_2", "He said piece.", "He said peace.", True),
    ("homophone_3", "He said heard.", "He said herd.", True),
    ("minor_vowel", "She held the doll.", "She held the dull.", True),
    ("other_name", "Jasper spoke quietly.", "Jasmine spoke quietly.", False),
    ("number", "There were fifteen candles.", "There were fifty candles.", False),
    ("missing_not", "She did not leave.", "She did leave.", False),
    ("missing_clause", "Derrins opened the door. He went inside.", "Darins opened the door.", False),
    ("extra_speech", "Derrins opened the door.", "Darins opened the door. Thank you.", False),
    ("reordered", "I saw the blue red door.", "I saw the red blue door.", False),
    ("synonym", "He walked along the road.", "He strolled along the road.", False),
    ("missing_syllable", "He saw a planet.", "He saw a plan.", False),
    ("missing_syllable_2", "He saw a candle.", "He saw a can.", False),
    ("extra_syllable", "He saw a cart.", "He saw a carton.", False),
    ("different_action", "He opened the door.", "He closed the door.", False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Global TOML containing [provider]")
    parser.add_argument("--output", type=Path, required=True, help="Save complete prompts and votes as JSON")
    args = parser.parse_args()
    provider = load_config(args.config)["provider"]
    if not provider.get("model"):
        parser.error("Configure [provider].model before running this live evaluation")

    def evaluate(case):
        name, expected, heard, want = case
        reviewer = create_phonetic_reviewer({}, provider)
        record = reviewer.review(expected, heard)
        passed = validate_review(record, expected, heard, reviewer.identity)
        return {"case": name, "expected": want, "accepted": passed,
                "correct": passed == want, "review": record}

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(evaluate, case) for case in CASES]):
            row = future.result()
            results.append(row)
            print(f"{row['case']}: {'PASS' if row['correct'] else 'FAIL'} "
                  f"({row['review']['decision']})", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sorted(results, key=lambda row: row["case"]), indent=2) + "\n")
    return 0 if all(row["correct"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
