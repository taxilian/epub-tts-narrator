"""Character auditions must retain attribution and exact source provenance."""

from copy import deepcopy
from pathlib import Path
import unittest

from audiobook_renderer.bundle import load_bundle
from audiobook_renderer.samples import select_sample


TEXT = (
    "I remember the road that brought us here. "
    "There is still time to find our friends and bring them safely home before nightfall."
)
FALLBACK = "This is the configured fallback sample."


def row(text=TEXT, *, voice="v1", role="r1", ident="s1", start=0,
        kind="dialogue", sequence=0):
    return {
        "id": ident, "block_id": "b-" + ident, "chapter_id": "ch1",
        "start": start, "end": start + len(text), "text": text,
        "kind": kind, "voice_id": voice, "role_id": role, "sequence": sequence,
        "delivery": {"direction": ""},
        "attribution": {
            "basis": "explicit", "candidates": [role], "review_required": False,
        },
    }


def bundle(rows=None):
    return {
        "source": {"book": {"id": "book1", "title": "An Example Book"}},
        "manifest": {"source_fingerprint": "source-sha256"},
        "cast": {
            "voices": [{"id": value} for value in ("v1", "v2", "vn")],
            "roles": [
                {"id": "r1", "voice_id": "v1", "kind": "dialogue"},
                {"id": "r2", "voice_id": "v2", "kind": "dialogue"},
                {"id": "rn", "voice_id": "vn", "kind": "narration"},
            ],
        },
        "chapters": [{
            "entry": {"title": "The Beginning", "path": "scripts/chapter-0001.jsonl"},
            "rows": [row()] if rows is None else rows,
        }],
    }


class SampleSelectionTests(unittest.TestCase):
    def select(self, data=None, voice="v1", aliases=None):
        return select_sample(
            data or bundle(), voice, canonical_id=lambda value: (aliases or {}).get(value, value),
            fallback_text=FALLBACK,
        )

    def test_sample_is_exact_source_with_block_and_spoken_offsets(self):
        text = "   " + TEXT + "  "
        data = bundle([row(text, start=17)])
        before = deepcopy(data)
        result = self.select(data)
        source = result["source"]
        self.assertEqual(result["reference_text"], TEXT)
        self.assertEqual(source["source_ranges"], [{"start": 20, "end": 20 + len(TEXT)}])
        self.assertEqual((source["spoken_start"], source["spoken_end"]), (3, 3 + len(TEXT)))
        self.assertEqual(source["book_id"], "book1")
        self.assertEqual(source["source_fingerprint"], "source-sha256")
        self.assertEqual(source["segment_ids"], ["s1"])
        self.assertEqual(source["chapter_title"], "The Beginning")
        self.assertEqual(data, before)

    def test_aliases_choose_the_same_dialogue_over_narration(self):
        narration = row(TEXT + " We know the way.", voice="vn", role="rn",
                        ident="n", kind="narration")
        data = bundle([narration, row()])
        aliases = {"vn": "v1"}
        self.assertEqual(self.select(data, "vn", aliases), self.select(data, "v1", aliases))
        self.assertEqual(self.select(data, "vn", aliases)["source"]["voice_id"], "v1")

    def test_unrelated_character_names_or_voice_descriptions_do_not_match(self):
        data = bundle([row(voice="v2", role="r2")])
        data["cast"]["voices"][0]["description"] = "Jasper"
        data["cast"]["voices"][1]["description"] = "Jasper"
        self.assertEqual(self.select(data)["source"]["kind"], "fallback")

    def test_review_or_ambiguous_or_unresolved_rows_are_not_selected(self):
        for update in (
            {"review_required": True},
            {"candidates": ["r1", "r2"]},
            {"candidates": ["r2"]},
            {"basis": "unresolved"},
        ):
            with self.subTest(update=update):
                sample = row()
                sample["attribution"].update(update)
                self.assertEqual(self.select(bundle([sample]))["source"]["kind"], "fallback")

    def test_clear_context_attribution_is_eligible(self):
        sample = row()
        sample["attribution"]["basis"] = "context"
        self.assertEqual(self.select(bundle([sample]))["reference_text"], TEXT)

    def test_neutral_line_preferred_by_excluding_extreme_direction(self):
        bad = row(ident="shout")
        bad["delivery"]["direction"] = "Shouting furiously."
        result = self.select(bundle([bad, row(ident="calm", sequence=1)]))
        self.assertEqual(result["source"]["segment_id"], "calm")

    def test_uppercase_and_control_syntax_are_not_used(self):
        for text in (TEXT.upper(), "[whispering] " + TEXT, "<speaker:0> " + TEXT):
            with self.subTest(text=text[:12]):
                result = self.select(bundle([row(text)]))
                if result["source"]["kind"] == "book_excerpt":
                    self.assertFalse(any(char in result["reference_text"] for char in "[]<>"))
                    self.assertIn(result["reference_text"], text)
                else:
                    self.assertEqual(result["reference_text"], FALLBACK)

    def test_analyzer_bracket_removal_keeps_exact_discontinuous_source_ranges(self):
        original = "I remember [the road] that brought us here. " + TEXT
        sample = row(original, start=9)
        sample["spoken_text"] = original.translate(str.maketrans("", "", "[]"))
        result = self.select(bundle([sample]))
        source = result["source"]
        reconstructed = "".join(
            original[part["start"] - 9:part["end"] - 9] for part in source["source_ranges"]
        )
        self.assertEqual(result["reference_text"], reconstructed)
        self.assertNotIn("[", result["reference_text"])
        self.assertGreater(len(source["source_ranges"]), 1)

    def test_unsupported_controls_use_another_source_passage_without_rewriting(self):
        for control in ("\x00", "\x7f", "\u00ad", "\u200b", "\u200d", "\u202e", "\ud800"):
            with self.subTest(control=repr(control)):
                bad_text = "We can find our friends and bring them " + control + " all safely home before nightfall."
                data = bundle([row(bad_text, ident="bad"), row(ident="clean", sequence=1)])
                before = deepcopy(data)
                result = self.select(data)
                self.assertEqual(result["reference_text"], TEXT)
                self.assertEqual(result["source"]["segment_id"], "clean")
                self.assertEqual(result["source"]["source_ranges"], [{"start": 0, "end": len(TEXT)}])
                self.assertEqual(data, before)

    def test_only_unsupported_control_passage_uses_fallback(self):
        text = "We can find our friends and bring them\u00ad all safely home before nightfall."
        result = self.select(bundle([row(text)]))
        self.assertEqual(result["reference_text"], FALLBACK)
        self.assertEqual(result["source"]["kind"], "fallback")

    def test_clean_sentence_after_format_control_preserves_source_offsets(self):
        bad = "This sen\u00adtence contains a soft hyphen. "
        clean = "We can find our friends and bring them all safely home before nightfall."
        result = self.select(bundle([row(bad + clean, start=17)]))
        self.assertEqual(result["reference_text"], clean)
        source = result["source"]
        self.assertEqual((source["spoken_start"], source["spoken_end"]),
                         (len(bad), len(bad + clean)))
        self.assertEqual(source["source_ranges"],
                         [{"start": 17 + len(bad), "end": 17 + len(bad + clean)}])

    def test_ordinary_whitespace_and_non_ascii_letters_remain_eligible(self):
        text = "We\tcan find our friends\nand bring them all safely home to the café before nightfall."
        result = self.select(bundle([row(text)]))
        self.assertEqual(result["reference_text"], text)
        self.assertEqual(result["source"]["kind"], "book_excerpt")

    def test_unsupported_spoken_edits_are_not_used(self):
        sample = row()
        sample["spoken_text"] = TEXT.replace("road", "river")
        self.assertEqual(self.select(bundle([sample]))["source"]["kind"], "fallback")

    def test_short_but_useful_line_is_retained_and_flagged(self):
        text = "We can find our friends and bring them all safely home before nightfall."
        result = self.select(bundle([row(text)]))
        self.assertEqual(result["reference_text"], text)
        self.assertTrue(result["source"]["short_sample"])

    def test_tiny_lines_fallback_and_are_never_concatenated(self):
        data = bundle([row("Come back to me.", ident=str(i), sequence=i) for i in range(5)])
        result = self.select(data)
        self.assertEqual(result["reference_text"], FALLBACK)
        self.assertEqual(result["source"]["reason"], "insufficient_character_dialogue")

    def test_narration_fallback_for_shared_character_without_dialogue(self):
        data = bundle([row(voice="vn", role="rn", kind="narration")])
        result = self.select(data, aliases={"vn": "v1"})
        self.assertEqual(result["source"]["segment_kind"], "narration")

    def test_substantive_other_text_requires_narration_role(self):
        bad = self.select(bundle([row(kind="other")]))
        good = self.select(bundle([row(voice="vn", role="rn", kind="other")]), "vn")
        self.assertEqual(bad["source"]["kind"], "fallback")
        self.assertEqual(good["source"]["kind"], "book_excerpt")

    def test_heading_is_not_an_audition(self):
        self.assertEqual(self.select(bundle([row(kind="heading")]))["source"]["kind"], "fallback")

    def test_long_row_is_cut_at_sentence_boundaries_without_rewriting(self):
        text = TEXT + " " + ("The long journey continued through the forest and over the mountains. " * 12)
        result = self.select(bundle([row(text)]))
        source = result["source"]
        self.assertLessEqual(len(result["reference_text"]), 400)
        self.assertEqual(result["reference_text"], text[source["spoken_start"]:source["spoken_end"]])
        self.assertTrue(result["reference_text"].endswith("."))
        self.assertTrue(source["spoken_start"] == 0 or text[:source["spoken_start"]].rstrip().endswith("."))

    def test_long_unpunctuated_row_is_not_chopped_into_an_invented_sentence(self):
        data = bundle([row("we walk quietly through the forest " * 20)])
        self.assertEqual(self.select(data)["source"]["kind"], "fallback")

    def test_unknown_voice_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not in"):
            self.select(voice="unknown")

    def test_actual_validated_book_has_traceable_selected_text(self):
        path = Path(__file__).resolve().parents[1] / "runs" / "soulship-series" / "book-01" / "bundle"
        if not path.exists():
            self.skipTest("Local full-book integration fixture is not available")
        data = load_bundle(path)
        seen = False
        for voice in data["cast"]["voices"]:
            result = select_sample(data, voice["id"], canonical_id=lambda value: value,
                                   fallback_text=FALLBACK)
            if result["source"]["kind"] != "book_excerpt":
                continue
            seen = True
            source = result["source"]
            selected = next(
                row for chapter in data["chapters"] for row in chapter["rows"]
                if row["id"] == source["segment_id"]
            )
            self.assertEqual(
                result["reference_text"],
                selected.get("spoken_text", selected["text"])[source["spoken_start"]:source["spoken_end"]],
            )
        self.assertTrue(seen)


if __name__ == "__main__":
    unittest.main()
