"""Character auditions must retain attribution and exact source provenance."""

from copy import deepcopy
from pathlib import Path
import unittest

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.samples import (
    MissingSampleScript, SELECTION_VERSION, select_book_sample, select_sample,
    sample_word_count, strip_speech_wrapper,
)


TEXT = (
    "I remember the road that brought us here. "
    "There is still time to find our friends and bring them safely home before nightfall. "
    "We know which path will lead us there, and we will stay together."
)
FALLBACK = ("I have checked the path ahead, and the bridge is still standing. "
            "Keep close as we cross, then tell me what you saw by the river. "
            "We have time to listen before deciding where to go next.")


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
        "audition_scripts": {value: {"text": FALLBACK, "basis": "casting",
                                     "reason": "An original calm audition matching the character."}
                              for value in ("v1", "v2", "vn")},
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

    def test_speech_wrappers_are_removed_with_exact_source_offsets(self):
        for opening, closing in (("“", "”"), ('"', '"'), ("‘", "’"), ("'", "'")):
            with self.subTest(opening=opening):
                original = " \t" + opening + "  " + TEXT + " " + closing + "\n"
                data = bundle([row(original, start=17)])
                before = deepcopy(data)
                selected = self.select(data)
                source = selected["source"]
                start = original.index(TEXT)
                self.assertEqual(selected["reference_text"], TEXT)
                self.assertEqual(source["selection_version"], SELECTION_VERSION)
                self.assertEqual((source["spoken_start"], source["spoken_end"]),
                                 (start, start + len(TEXT)))
                self.assertEqual(source["source_ranges"],
                                 [{"start": 17 + start, "end": 17 + start + len(TEXT)}])
                self.assertEqual(data, before)

    def test_joined_quoted_statements_have_no_repeated_speech_wrappers(self):
        texts = ["We crossed the river before dawn and left our packs beside the bridge.",
                 "I checked the lantern while the others searched the narrow trail through the woods.",
                 "There is still time to return, but we should make our choice together."]
        quoted = ["“" + texts[0] + "”", '"' + texts[1] + '"', "‘" + texts[2] + "’"]
        data = bundle([row(text, ident=f"quoted-{i}", sequence=i, start=17 * i)
                       for i, text in enumerate(quoted)])
        before = deepcopy(data)
        selected = self.select(data)
        self.assertEqual(selected["reference_text"], " ".join(texts))
        self.assertEqual(selected["source"]["kind"], "book_excerpts")
        for index, excerpt in enumerate(selected["source"]["excerpts"]):
            self.assertEqual(excerpt["text"], texts[index])
            self.assertEqual((excerpt["spoken_start"], excerpt["spoken_end"]),
                             (1, len(quoted[index]) - 1))
            self.assertEqual(excerpt["source_ranges"],
                             [{"start": 17 * index + 1, "end": 17 * index + len(quoted[index]) - 1}])
            self.assertEqual(excerpt["text"], selected["reference_text"][
                excerpt["sample_start"]:excerpt["sample_end"]])
        self.assertEqual(data, before)

    def test_embedded_quotations_and_apostrophes_are_preserved(self):
        for opening, closing, embedded in (
            ("“", "”", "‘friends’"), ('"', '"', "'friends'"),
            ("‘", "’", '“friends”'), ("'", "'", '"friends"'),
        ):
            with self.subTest(opening=opening):
                content = TEXT.replace("our friends", "our " + embedded).replace("We know", "We're sure")
                selected = self.select(bundle([row(opening + content + closing)]))
                self.assertEqual(selected["reference_text"], content)
                self.assertIn(embedded, selected["reference_text"])
                self.assertIn("We're", selected["reference_text"])

    def test_speech_cleanup_does_not_strip_unpaired_or_separate_quotes(self):
        for text in ("'Tis " + TEXT, "’Tis " + TEXT, '"' + TEXT, TEXT + "”",
                     '"The first sentence." "The second sentence."'):
            with self.subTest(text=text[:25]):
                self.assertEqual(strip_speech_wrapper(text), text)

    def test_narrative_quotation_is_not_removed(self):
        text = "“" + TEXT + "”"
        selected = self.select(bundle([row(text, voice="vn", role="rn", kind="narration")]), "vn")
        self.assertEqual(selected["reference_text"], text)
        self.assertEqual(selected["source"]["source_ranges"], [{"start": 0, "end": len(text)}])

    def test_quote_cleanup_preserves_unicode_and_discontinuous_bracket_mapping(self):
        content = TEXT.replace("road", "[café road]")
        original = "\t“ " + content + " ”\n"
        sample = row(original, start=19)
        sample["spoken_text"] = original.translate(str.maketrans("", "", "[]"))
        before = deepcopy(sample)
        selected = self.select(bundle([sample]))
        source = selected["source"]
        self.assertEqual(selected["reference_text"], TEXT.replace("road", "café road"))
        self.assertEqual(selected["reference_text"], sample["spoken_text"][
            source["spoken_start"]:source["spoken_end"]])
        self.assertEqual(selected["reference_text"], "".join(
            original[part["start"] - 19:part["end"] - 19] for part in source["source_ranges"]))
        self.assertGreater(len(source["source_ranges"]), 1)
        self.assertEqual(sample, before)

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
        self.assertEqual(self.select(data)["source"]["kind"], "casting_script")

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
                self.assertEqual(self.select(bundle([sample]))["source"]["kind"], "casting_script")

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
        self.assertEqual(result["source"]["kind"], "casting_script")

    def test_clean_sentence_after_format_control_preserves_source_offsets(self):
        bad = "This sen\u00adtence contains a soft hyphen. "
        clean = "We can find our friends and bring them all safely home before nightfall. " + TEXT
        result = self.select(bundle([row(bad + clean, start=17)]))
        self.assertEqual(result["reference_text"], clean)
        source = result["source"]
        self.assertEqual((source["spoken_start"], source["spoken_end"]),
                         (len(bad), len(bad + clean)))
        self.assertEqual(source["source_ranges"],
                         [{"start": 17 + len(bad), "end": 17 + len(bad + clean)}])

    def test_ordinary_whitespace_and_non_ascii_letters_remain_eligible(self):
        text = ("We\tcan find our friends\nand bring them all safely home to the café before nightfall. "
                "There is still time to talk about the path ahead and decide where we should go together.")
        result = self.select(bundle([row(text)]))
        self.assertEqual(result["reference_text"], text)
        self.assertEqual(result["source"]["kind"], "book_excerpt")

    def test_unsupported_spoken_edits_are_not_used(self):
        sample = row()
        sample["spoken_text"] = TEXT.replace("road", "river")
        self.assertEqual(self.select(bundle([sample]))["source"]["kind"], "casting_script")

    def test_one_short_line_uses_character_specific_creative_script(self):
        text = "We can find our friends and bring them all safely home before nightfall."
        result = self.select(bundle([row(text)]))
        self.assertEqual(result["reference_text"], FALLBACK)
        self.assertEqual(result["source"]["kind"], "casting_script")

    def test_repeated_tiny_line_is_not_duplicated_to_manufacture_length(self):
        data = bundle([row("Come back to me.", ident=str(i), sequence=i) for i in range(5)])
        result = self.select(data)
        self.assertEqual(result["reference_text"], FALLBACK)
        self.assertEqual(result["source"]["basis"], "casting")

    def test_narration_fallback_for_shared_character_without_dialogue(self):
        data = bundle([row(voice="vn", role="rn", kind="narration")])
        result = self.select(data, aliases={"vn": "v1"})
        self.assertEqual(result["source"]["segment_kind"], "narration")

    def test_substantive_other_text_requires_narration_role(self):
        bad = self.select(bundle([row(kind="other")]))
        good = self.select(bundle([row(voice="vn", role="rn", kind="other")]), "vn")
        self.assertEqual(bad["source"]["kind"], "casting_script")
        self.assertEqual(good["source"]["kind"], "book_excerpt")

    def test_heading_is_not_an_audition(self):
        self.assertEqual(self.select(bundle([row(kind="heading")]))["source"]["kind"], "casting_script")

    def test_long_row_is_cut_at_sentence_boundaries_without_rewriting(self):
        text = TEXT + " " + ("The long journey continued through the forest and over the mountains. " * 12)
        result = self.select(bundle([row(text)]))
        source = result["source"]
        self.assertLessEqual(len(result["reference_text"]), 650)
        self.assertEqual(result["reference_text"], text[source["spoken_start"]:source["spoken_end"]])
        self.assertTrue(result["reference_text"].endswith("."))
        self.assertTrue(source["spoken_start"] == 0 or text[:source["spoken_start"]].rstrip().endswith("."))

    def test_long_wrapped_speech_excerpt_does_not_keep_a_stray_outer_quote(self):
        content = TEXT + " " + ("The long journey continued through the forest and over the mountains. " * 12)
        text = "“" + content.rstrip() + "”"
        result = self.select(bundle([row(text, start=31)]))
        source = result["source"]
        self.assertNotIn("“", result["reference_text"])
        self.assertNotIn("”", result["reference_text"])
        self.assertEqual(result["reference_text"], text[source["spoken_start"]:source["spoken_end"]])
        self.assertEqual(source["source_ranges"],
                         [{"start": 31 + source["spoken_start"], "end": 31 + source["spoken_end"]}])

    def test_long_unpunctuated_row_is_not_chopped_into_an_invented_sentence(self):
        data = bundle([row("we walk quietly through the forest " * 20)])
        self.assertEqual(self.select(data)["source"]["kind"], "casting_script")

    def test_unknown_voice_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not in"):
            self.select(voice="unknown")

    def test_multiple_statements_form_substantial_audition_with_separate_provenance(self):
        texts = ["We crossed the river before dawn and left our packs beside the bridge.",
                 "I checked the lantern while the others searched the narrow trail through the woods.",
                 "There is still time to return, but we should make our choice together."]
        data = bundle([row(text, ident=f"quote-{i}", sequence=i, start=17 * i)
                       for i, text in enumerate(texts)])
        before = deepcopy(data)
        result = self.select(data)
        source = result["source"]
        self.assertEqual(source["kind"], "book_excerpts")
        self.assertGreaterEqual(sample_word_count(result["reference_text"]), 35)
        self.assertLessEqual(sample_word_count(result["reference_text"]), 60)
        self.assertNotIn("block_id", source)
        self.assertNotIn("source_ranges", source)
        self.assertEqual(result["reference_text"], " ".join(item["text"] for item in source["excerpts"]))
        rows = {item["id"]: item for item in data["chapters"][0]["rows"]}
        for excerpt in source["excerpts"]:
            original = rows[excerpt["segment_id"]]
            self.assertEqual(excerpt["text"], original["text"][excerpt["spoken_start"]:excerpt["spoken_end"]])
            self.assertEqual(excerpt["text"], result["reference_text"][excerpt["sample_start"]:excerpt["sample_end"]])
            self.assertEqual(excerpt["source_ranges"], [{"start": original["start"], "end": original["end"]}])
        self.assertEqual(data, before)

    def test_join_never_uses_other_characters_or_unresolved_passages(self):
        short = row("The bridge is clear, and our friends are waiting on the other side.")
        another = row(TEXT, voice="v2", role="r2", ident="other-person", sequence=1)
        uncertain = row(TEXT, ident="uncertain", sequence=2)
        uncertain["attribution"]["review_required"] = True
        data = bundle([short, another, uncertain])
        self.assertIsNone(select_book_sample(data, "v1", canonical_id=lambda value: value))
        self.assertEqual(self.select(data)["source"]["kind"], "casting_script")

    def test_resolved_simultaneous_text_is_eligible_for_each_solo_audition(self):
        sample = row(voice=None, role=None)
        sample["attribution"]["candidates"] = []
        sample["simultaneous_speakers"] = [
            {"role_id": "r1", "voice_id": "v1"}, {"role_id": "r2", "voice_id": "v2"}]
        data = bundle([sample])
        for voice, role in (("v1", "r1"), ("v2", "r2")):
            selected = self.select(data, voice)
            self.assertEqual(selected["reference_text"], TEXT)
            self.assertEqual(selected["source"]["role_id"], role)
            self.assertEqual(selected["source"]["simultaneous_speakers"], sample["simultaneous_speakers"])

    def test_creative_script_is_truthfully_marked_and_has_no_source_offsets(self):
        selected = self.select(bundle([]))
        self.assertEqual(selected["reference_text"], FALLBACK)
        self.assertEqual(selected["source"]["kind"], "casting_script")
        self.assertEqual(selected["source"]["basis"], "casting")
        for field in ("source_ranges", "segment_id", "block_id", "chapter_id"):
            self.assertNotIn(field, selected["source"])

    def test_missing_prepared_script_requests_preparation_instead_of_generic_fallback(self):
        data = bundle([])
        del data["audition_scripts"]
        with self.assertRaises(MissingSampleScript) as failure:
            self.select(data)
        self.assertEqual(failure.exception.voice_id, "v1")

    def test_provenance_and_profile_scripts_are_supported_without_a_sidecar(self):
        for location in ("provenance", "profile"):
            data = bundle([])
            script = data.pop("audition_scripts")["v1"]
            if location == "provenance":
                data["manifest"]["provenance"] = {"voice_samples": {"scripts": {"v1": script}}}
            else:
                data["cast"]["voices"][0]["sample_script"] = script
            self.assertEqual(self.select(data)["reference_text"], FALLBACK)

    def test_refresh_overrides_suppress_stale_provenance_and_profile_scripts(self):
        for location in ("provenance", "profile"):
            with self.subTest(location=location):
                data = bundle([])
                old_script = data.pop("audition_scripts")["v1"]
                if location == "provenance":
                    data["manifest"]["provenance"] = {"voice_samples": {"scripts": {"v1": old_script}}}
                else:
                    data["cast"]["voices"][0]["sample_script"] = old_script
                data["audition_script_overrides"] = ["v1"]
                with self.assertRaises(MissingSampleScript):
                    self.select(data)
                # A newly prepared sidecar script still takes precedence.
                refreshed = dict(old_script, text=TEXT, reason="A refreshed audition.")
                data["audition_scripts"] = {"v1": refreshed}
                self.assertEqual(self.select(data)["reference_text"], TEXT)

    def test_refresh_override_does_not_hide_unaffected_voice_scripts(self):
        data = bundle([])
        old_script = data.pop("audition_scripts")["v1"]
        data["manifest"]["provenance"] = {"voice_samples": {"scripts": {"v1": old_script}}}
        data["audition_script_overrides"] = ["v2"]
        self.assertEqual(self.select(data)["reference_text"], FALLBACK)

    def test_too_short_or_control_bearing_creative_script_is_rejected(self):
        for text in ("Hello, my friend.", "[shouting] " + FALLBACK, FALLBACK + "\x00"):
            data = bundle([])
            data["audition_scripts"]["v1"]["text"] = text
            with self.assertRaisesRegex(ValueError, "Invalid prepared casting sample"):
                self.select(data)

    def test_actual_validated_book_has_traceable_selected_text(self):
        path = Path(__file__).resolve().parents[1] / "runs" / "soulship-series" / "book-01" / "bundle"
        if not path.exists():
            self.skipTest("Local full-book integration fixture is not available")
        data = load_bundle(path)
        seen = False
        for voice in data["cast"]["voices"]:
            result = select_book_sample(data, voice["id"], canonical_id=lambda value: value)
            if result is None:
                continue
            seen = True
            source = result["source"]
            excerpts = source.get("excerpts", [dict(source, text=result["reference_text"])])
            for excerpt in excerpts:
                selected = next(row for chapter in data["chapters"] for row in chapter["rows"]
                                if row["id"] == excerpt["segment_id"])
                self.assertEqual(excerpt["text"], selected.get("spoken_text", selected["text"])[
                    excerpt["spoken_start"]:excerpt["spoken_end"]])
        self.assertTrue(seen)


if __name__ == "__main__":
    unittest.main()
