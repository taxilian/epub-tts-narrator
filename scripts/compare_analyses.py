#!/usr/bin/env python3
"""Compare two analyses of the same source: cast mapping and per-quotation speakers.

Characters are mapped one-to-one by shared quotation labels; a quotation agrees
when its two speakers are mapped to each other.

Offline and read-only. Inputs are analysis.json files (book directories are
accepted). Output: a JSON summary on stdout and, with --samples, a Markdown
file of random disagreements with their paragraphs for spot-checking. The
sample file contains book text; keep it under a gitignored run directory.

  python3.11 scripts/compare_analyses.py BASELINE NARRATOR [--samples out.md] [--count 50]
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.quote_spans import quote_spans  # noqa: E402


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / "analysis.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(name):
    return " ".join(re.findall(r"\w+", (name or "").casefold()))


def speaker_lookup(document):
    """role_id -> character id ("narrator" for narration roles)."""
    roles = {}
    for role in document["cast"]["roles"]:
        roles[role["id"]] = "narrator" if role["kind"] == "narration" else role["character_id"]
    return roles


def names(document):
    return {row["id"]: row for row in document["cast"]["characters"]}


def span_speakers(document, spans):
    """For each quote span, the speaker of the segment covering its first inner character."""
    roles = speaker_lookup(document)
    by_block = {}
    for row in document["segments"]:
        by_block.setdefault(row["block_id"], []).append(row)
    result = {}
    for span in spans:
        point = min(span["start"] + 1, span["end"] - 1)
        row = next((item for item in by_block.get(span["block_id"], [])
                    if item["start"] <= point < item["end"]), None)
        if row is None:
            result[span["id"]] = ("missing",)
        elif row.get("simultaneous_speakers"):
            result[span["id"]] = ("multi", tuple(sorted(roles.get(p["role_id"]) for p in row["simultaneous_speakers"])))
        elif row["role_id"] is None:
            result[span["id"]] = ("unresolved",)
        elif roles.get(row["role_id"]) == "narrator":
            result[span["id"]] = ("narrator",)
        else:
            result[span["id"]] = ("char", roles[row["role_id"]])
    return result


def map_characters(a_doc, b_doc, a_speakers, b_speakers):
    """Map each A character to the B character it most often co-labels (and the reverse)."""
    co = Counter()
    for qid, left in a_speakers.items():
        right = b_speakers.get(qid)
        if left[0] == "char" and right and right[0] == "char":
            co[(left[1], right[1])] += 1
    # One-to-one: each character on either side maps at most once (largest shared
    # line counts first), so one character absorbing two others shows as disagreement.
    forward, reverse = {}, {}
    for (left, right), count in co.most_common():
        if left not in forward and right not in reverse:
            forward[left] = (right, count)
            reverse[right] = (left, count)
    return forward, reverse, co


def compare(a_doc, b_doc, *, seed=7, count=50):
    if a_doc["source"] != b_doc["source"]:
        if [b["text"] for b in a_doc["source"]["blocks"]] != [b["text"] for b in b_doc["source"]["blocks"]]:
            raise ValueError("Analyses must cover identical source text")
    spans = quote_spans(a_doc["source"])
    a_speakers, b_speakers = span_speakers(a_doc, spans), span_speakers(b_doc, spans)
    a_names, b_names = names(a_doc), names(b_doc)
    forward, reverse, co = map_characters(a_doc, b_doc, a_speakers, b_speakers)

    def same_name(left, right):
        lrow, rrow = a_names.get(left), b_names.get(right)
        if not lrow or not rrow:
            return False
        lset = {_norm(lrow["name"]), *map(_norm, lrow.get("aliases", []))}
        rset = {_norm(rrow["name"]), *map(_norm, rrow.get("aliases", []))}
        return bool(lset & rset)

    outcome, disagreements = Counter(), []
    for span in spans:
        left, right = a_speakers[span["id"]], b_speakers[span["id"]]
        if left[0] == "char" and right[0] == "char":
            if forward.get(left[1], (None,))[0] == right[1]:
                key = "agree_char" if same_name(left[1], right[1]) else "agree_mapped_only"
            else:
                key = "disagree_char"
        elif left[0] == right[0] and left[0] in {"narrator", "unresolved"}:
            key = f"agree_{left[0]}"
        elif "unresolved" in (left[0], right[0]):
            key = "unresolved_one_side"
        elif "narrator" in (left[0], right[0]):
            key = "narrator_vs_speech"
        else:
            key = f"other_{left[0]}_{right[0]}"
        outcome[key] += 1
        if not key.startswith("agree"):
            disagreements.append((span, left, right, key))
    decided = sum(value for key, value in outcome.items() if key in {"agree_char", "agree_mapped_only", "disagree_char"})
    # Cast structure: splits (one A char -> several B chars with >=3 shared spans) and merges.
    splits = {}
    for (left, right), value in co.items():
        if value >= 3:
            splits.setdefault(left, set()).add(right)
    merged = {}
    for (left, right), value in co.items():
        if value >= 3:
            merged.setdefault(right, set()).add(left)
    speaking_a = {item[1] for item in a_speakers.values() if item[0] == "char"}
    speaking_b = {item[1] for item in b_speakers.values() if item[0] == "char"}
    summary = {
        "quote_spans": len(spans),
        "outcomes": dict(outcome),
        "char_agreement_rate": round((outcome["agree_char"] + outcome["agree_mapped_only"]) / decided, 4) if decided else None,
        "strict_name_agreement_rate": round(outcome["agree_char"] / decided, 4) if decided else None,
        "cast": {
            "a_characters": len(a_doc["cast"]["characters"]), "b_characters": len(b_doc["cast"]["characters"]),
            "a_speaking": len(speaking_a), "b_speaking": len(speaking_b),
            "a_unmapped_speaking": sorted(a_names[c]["name"] for c in speaking_a - set(forward) if c in a_names),
            "b_unmapped_speaking": sorted(b_names[c]["name"] for c in speaking_b - set(reverse) if c in b_names),
            "a_split_in_b": {a_names[k]["name"]: sorted(b_names[v]["name"] for v in vs)
                             for k, vs in splits.items() if len(vs) > 1 and k in a_names},
            "a_merged_in_b": {b_names[k]["name"]: sorted(a_names[v]["name"] for v in vs)
                              for k, vs in merged.items() if len(vs) > 1 and k in b_names},
        },
        "unresolved": {"a": sum(1 for v in a_speakers.values() if v[0] == "unresolved"),
                       "b": sum(1 for v in b_speakers.values() if v[0] == "unresolved")},
    }
    random.Random(seed).shuffle(disagreements)
    return summary, disagreements[:count]


def describe(document, speaker):
    if speaker[0] == "char":
        row = names(document).get(speaker[1])
        return row["name"] if row else speaker[1]
    return speaker[0]


def write_samples(path, a_doc, b_doc, samples, labels):
    blocks = {block["id"]: block for block in a_doc["source"]["blocks"]}
    order = [block["id"] for block in a_doc["source"]["blocks"]]
    position = {block_id: index for index, block_id in enumerate(order)}
    lines = [f"# Disagreement spot-check ({labels[0]} vs {labels[1]})", "",
             "Verdict column: fill A, B, both wrong, or ambiguous.", ""]
    for number, (span, left, right, key) in enumerate(samples, 1):
        index = position[span["block_id"]]
        context = [blocks[order[i]]["text"] for i in range(max(0, index - 2), min(len(order), index + 2))]
        lines += [f"## {number}. {span['block_id']} {key}", "",
                  f"- quote: {span['text'][:300]}",
                  f"- {labels[0]}: **{describe(a_doc, left)}**",
                  f"- {labels[1]}: **{describe(b_doc, right)}**", "- verdict: ", "", "> " + "\n>\n> ".join(context), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--samples", type=Path, help="Write random disagreements with context (contains book text)")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--labels", default="A,B")
    args = parser.parse_args(argv)
    a_doc, b_doc = load(args.a), load(args.b)
    summary, samples = compare(a_doc, b_doc, count=args.count)
    if args.samples:
        write_samples(args.samples, a_doc, b_doc, samples, args.labels.split(","))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
