"""Experimental narrator read: offline mechanics with a fake model boundary."""

import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main
from epub_tts_narrator.contracts import readiness
from epub_tts_narrator.narrator_ledger import apply_delta, new_ledger, undo_merge
from epub_tts_narrator.narrator_read import narrated_analyze
from epub_tts_narrator.provider import OpenAICompatibleClient, ProviderError
from epub_tts_narrator.quote_spans import find_tag, quote_spans
from shared_fixtures import make_serial_epub
from test_provider import FakeResponse, delta, delta_end, sse


def block(index, chapter, text, tag="p"):
    return {"id": f"b{index:06d}", "chapter_id": chapter, "text": text,
            "source": {"href": f"{chapter}.xhtml", "element_id": None, "tag": tag}}


def make_source(paragraphs_by_chapter):
    blocks, chapters = [], []
    for number, paragraphs in enumerate(paragraphs_by_chapter, 1):
        chapter = f"ch{number:04d}"
        ids = []
        for text, tag in paragraphs:
            blocks.append(block(len(blocks) + 1, chapter, text, tag))
            ids.append(blocks[-1]["id"])
        chapters.append({"id": chapter, "title": f"Chapter {number}", "source_href": f"{chapter}.xhtml",
                         "block_ids": ids})
    return {"book": {"id": "book", "title": "Fixture", "authors": ["A"], "language": "en"},
            "chapters": chapters, "blocks": blocks, "exclusions": [], "warnings": [],
            "extraction_policy": "fixture"}


BOOK = [
    [("Chapter One", "h1"),
     ("Mara put down the cup. \"You never listen,\" Jon said.", "p"),
     ("\"I thought,\" Mara said, \"that you had gone.\" Her voice was husky and low.", "p"),
     ("\"Before you ask, I kept a copy,\" Jon added.", "p"),
     ("Somebody laughed. \"Nobody asked.\"", "p")],
    [("Chapter Two", "h1"),
     ("From the stairs, Elsie said, \"No, you don't.\"", "p"),
     ("The sign read \"North Station\" in faded paint.", "p")],
]


class NarratorClient:
    """Fake model: reads names from the text like a careful narrator would."""

    NAMES = {"Mara": "f", "Jon": "m", "Elsie": "f"}

    def __init__(self, model="fixture", base_url="http://offline.invalid/v1", *, verify_answer="first",
                 merge_answer="y", series_answer="y", duplicate_answer="y", names=None, review_answer="keep",
                 drop_first_markup_label=False, **kwargs):
        self.model, self.base_url = model, base_url
        self.identity = {"model": model, "base_url": base_url}
        self.calls = []
        self.verify_answer = verify_answer
        self.merge_answer = merge_answer
        self.series_answer = series_answer
        self.duplicate_answer = duplicate_answer
        self.review_answer = review_answer
        if names is not None:
            self.NAMES = names
        self.drop_first_markup_label = drop_first_markup_label
        self.usage_observer = None
        self.include_usage = False

    def _cast(self, text):
        rows = {}
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) >= 3 and re.fullmatch(r"\d+", parts[0]):
                rows[parts[1]] = parts[0]
                for alias in filter(None, parts[2].split(";")):
                    rows[alias] = parts[0]
        return rows

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if self.usage_observer is not None:
            self.usage_observer({"prompt_tokens": len(json.dumps(payload)) // 4, "completion_tokens": 20,
                                 "reasoning_tokens": 5, "cached_tokens": 0, "elapsed_seconds": 0.01,
                                 "request_bytes": len(json.dumps(payload)), "response_bytes": 100})
        mode = payload.get("mode")
        if mode == "narrator_prep":
            known = self._cast(payload["cast"])
            new, voice = [], []
            for line in payload["text"].splitlines():
                number, _, text = line.partition("|")
                for name, sex in self.NAMES.items():
                    if name in text and name not in known and name not in [row[0] for row in new]:
                        new.append([name, "", sex, f"{name} in the fixture", int(number)])
                if "husky and low" in text:
                    voice.append(["Mara", int(number), "Her voice was husky and low."])
                    voice.append(["Mara", int(number), "not in the paragraph"])
            return {"new": new, "voice": voice, "nar": ["third", None]}
        if mode == "inline_performance_review":
            return {"decisions": [{"id": item["id"], "verdict": self.review_answer} for item in payload["proposals"]]}
        if mode == "narrator_duplicate_check":
            return {"r": [[item["id"], self.duplicate_answer] for item in payload["items"]]}
        if mode == "narrator_merge_check":
            return {"r": [[item["id"], self.merge_answer] for item in payload["items"]]}
        if mode == "narrator_series_check":
            rows = []
            for item in payload["items"]:
                number, _, text = item["text"].splitlines()[0].partition("|")
                rows.append([item["id"], "y", int(number), text[:200]] if self.series_answer == "y"
                            else [item["id"], self.series_answer])
            return {"r": rows}
        if mode == "narrator_markup":
            known = self._cast(payload["cast"])
            rows = []
            for line in payload["text"].splitlines():
                for match in re.finditer(r"‹(q\d+)›", line):
                    after = line[match.end():]
                    before = line[:match.start()]
                    if "North Station" in after[:20]:
                        rows.append([match.group(1), "-"])
                        continue
                    speaker = None
                    tail = after.split('"', 2)[-1][:40] if after.count('"') >= 2 else ""
                    for name, identifier in known.items():
                        if name in tail or re.search(rf"{name} said, $", before):
                            speaker = identifier
                    rows.append([match.group(1), speaker or "?" + ",".join(sorted(set(known.values())))])
            if self.drop_first_markup_label and not any(call.get("correction") for call in self.calls[:-1]) \
                    and "correction" not in payload:
                rows = rows[1:]
            return {"q": rows}
        if mode == "narrator_verify":
            answers = []
            for item in payload["items"]:
                answer = item["options"][0] if self.verify_answer == "first" and item["options"] else "?"
                answers.append([item["id"], answer])
            return {"r": answers}
        if mode == "narrator_verify_tiebreak":
            return {"r": [[item["id"], item["choices"][1]["id"]] for item in payload["items"]]}
        if mode == "narrator_accents":
            # Different backgrounds, different accents: the first character is Scottish.
            names = [row[0] for row in payload["characters"]]
            return {"a": [["the north", "Scottish", names[:1]], ["the coast", "Irish", names[1:2]]]}
        if mode == "narrator_voices":
            gender = {"m": "male", "f": "female"}
            # Real-world words only: never a name from the book.
            return {"v": [[row[0], f"A clear {gender.get(row[2], 'androgynous')} voice number {row[0][1:]}.",
                           "General American; even pace",
                           "pitch=mid;pace=even;accent=General American"] for row in payload["voices"]]}
        raise ProviderError("unexpected request in fixture")


class QuoteSpanTests(unittest.TestCase):
    def test_split_quotation_and_trailing_tag_are_found_without_model_calls(self):
        source = make_source([[("\"I thought,\" Mara said, \"that you had gone.\"", "p")]])
        spans = quote_spans(source)
        self.assertEqual([span["text"] for span in spans], ['"I thought,"', '"that you had gone."'])
        self.assertEqual(spans[0]["tag"]["name"], "Mara")
        self.assertEqual(find_tag('"Go," he said.', 0, 5)["pronoun"], "he")

    def test_single_quote_dialogue_is_told_apart_from_apostrophes(self):
        from epub_tts_narrator.quote_spans import dialogue_style
        paragraphs = [("‘I don’t know James’ plan,’ Mara said. ‘Do you?’", "p"),
                      ("'Give 'em hell,' Jon said, 'and come back.'", "p"),
                      ("‘The boys’ toys are here,’ said Elsie.", "p"),
                      ("It's Tom's boat now.", "p"), ("‘Wait—’ Mara began.", "p"), ("‘Fine,’ said Jon.", "p")]
        source = make_source([paragraphs])
        self.assertEqual(dialogue_style(source), "single")
        spans = quote_spans(source)
        self.assertEqual([span["text"] for span in spans],
                         ["‘I don’t know James’ plan,’", "‘Do you?’", "'Give 'em hell,'", "'and come back.'",
                          "‘The boys’ toys are here,’", "‘Wait—’", "‘Fine,’"])
        self.assertEqual(spans[0]["tag"]["name"], "Mara")
        self.assertEqual(spans[2]["tag"]["name"], "Jon")

    def test_dash_dialogue_splits_speech_from_its_tag(self):
        from epub_tts_narrator.quote_spans import dialogue_style
        source = make_source([[("— Hello, Mara said. — How are you?", "p"), ("– Nice weather, said Jon.", "p"),
                               ("—Stop! Elsie shouted. —Now!", "p"), ("— Just a line.", "p"), ("— Fine.", "p"),
                               ("The road was long.", "p")]])
        self.assertEqual(dialogue_style(source), "dash")
        spans = quote_spans(source)
        self.assertEqual([span["text"] for span in spans],
                         ["— Hello,", "— How are you?", "– Nice weather,", "—Stop!", "—Now!", "— Just a line.", "— Fine."])
        self.assertEqual(spans[0]["tag"]["name"], "Mara")

    def test_a_double_quote_book_ignores_dashes_and_nested_single_quotes(self):
        from epub_tts_narrator.quote_spans import dialogue_style
        source = make_source([[('"He said \'go\' and left," Mara said.', "p"), ("— a dash aside.", "p"),
                               ('"Yes," Jon said.', "p")]])
        self.assertEqual(dialogue_style(source), "double")
        self.assertEqual(len(quote_spans(source)), 2)

    def test_multi_paragraph_speech_marks_the_continuing_paragraph(self):
        source = make_source([[("\"First part of a speech", "p"), ("\"and its continuation.\"", "p")]])
        spans = quote_spans(source)
        self.assertFalse(spans[0]["closed"])
        self.assertTrue(spans[1]["continues"])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.blocks = {"b000001": "Mara met Jon. Her voice was husky.", "b000002": "Jon, also called Johnny, waved."}
        self.numbers = {1: "b000001", 2: "b000002"}

    def test_invalid_optional_claims_are_dropped_instead_of_requesting_repair(self):
        ledger = apply_delta(new_ledger(), {"new": [["Mara", "", "f", "keeper", 1]],
                                            "voice": [["Mara", 1, "Her voice was husky."], ["Mara", 1, "invented"],
                                                      ["Nobody", 1, "Mara met Jon."]]},
                             self.blocks, self.numbers)
        self.assertEqual(ledger["characters"][0]["voice"], [{"block_id": "b000001", "quote": "Her voice was husky."}])
        self.assertEqual(ledger["dropped"], {"voice": 2})

    def test_structural_problems_raise_for_repair_and_unknown_keys_are_ignored(self):
        self.assertEqual(apply_delta(new_ledger(), {"characters": []}, self.blocks, self.numbers)["dropped"],
                         {"unknown_key": 1})
        with self.assertRaisesRegex(ValueError, "array"):
            apply_delta(new_ledger(), {"new": "Mara"}, self.blocks, self.numbers)

    def test_malformed_optional_entries_are_dropped_and_counted(self):
        ledger = apply_delta(new_ledger(), {"new": [["Mara", "", "f", "", 1]], "voice": [["Mara"], "x"]},
                             self.blocks, self.numbers)
        self.assertEqual(ledger["dropped"], {"voice": 2})

    def test_rejected_merge_restores_both_entries(self):
        ledger = apply_delta(new_ledger(), {"new": [["Jon", "", "m", "", 1], ["Johnny", "", "m", "", 2]]},
                             self.blocks, self.numbers)
        # Prefixed, bare-string and integer IDs all name the same entries.
        merged = apply_delta(ledger, {"same": [["c2", 1, 2, "Jon, also called Johnny"]]}, self.blocks, self.numbers)
        self.assertEqual([row["id"] for row in merged["characters"]], ["1"])
        self.assertIn("Johnny", merged["characters"][0]["aliases"])
        split = undo_merge(merged, "2", "1")
        self.assertEqual(sorted(row["id"] for row in split["characters"]), ["1", "2"])
        self.assertNotIn("Johnny", split["characters"][0]["aliases"])
        self.assertEqual(split["merges"][0]["status"], "rejected")

    def test_a_name_matching_only_another_entrys_alias_still_gets_its_own_entry(self):
        from epub_tts_narrator.narrator_ledger import name_index
        blocks = {"b000001": "Anderson waved.", "b000002": "Proof laughed."}
        ledger = apply_delta(new_ledger(), {"new": [["Anderson Royce", "Proof", "m", "", 1]]}, blocks, self.numbers)
        ledger = apply_delta(ledger, {"new": [["Proof", "", "m", "", 2]]}, blocks, self.numbers)
        self.assertEqual([row["name"] for row in ledger["characters"]], ["Anderson Royce", "Proof"])
        self.assertEqual(name_index(ledger)["proof"], "2")

    def test_a_merge_into_an_entry_merged_again_later_is_checked_through_the_later_merge(self):
        from epub_tts_narrator.narrator_read import Meter, check_merges
        from epub_tts_narrator.pipeline import CachedRequests
        blocks = {"b000001": "Mara met Mar and Mae.", "b000002": "Mar is Mae, and Mae is Mara."}
        numbers = {1: "b000001", 2: "b000002"}
        ledger = apply_delta(new_ledger(), {"new": [["Mara", "", "f", "", 1], ["Mar", "", "f", "", 1], ["Mae", "", "f", "", 1]]},
                             blocks, numbers)
        ledger = apply_delta(ledger, {"same": [[2, 3, 2, "Mar is Mae"]]}, blocks, numbers)   # Mar into Mae
        ledger = apply_delta(ledger, {"same": [[3, 1, 2, "Mae is Mara"]]}, blocks, numbers)  # Mae into Mara

        class Yes:
            identity = {"model": "fixture", "base_url": "http://offline.invalid/v1"}

            def complete(self, system, payload):
                return {"r": [[item["id"], "y"] for item in payload["items"]]}

        source = make_source([[("Mara met Mar and Mae.", "p"), ("Mar is Mae, and Mae is Mara.", "p")]])
        client = Yes()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        checked, outcomes = check_merges(source, CachedRequests(client, Path(directory.name), repair_attempts=0),
                                         Meter(client), ledger)
        self.assertEqual(sorted(outcome["verdict"] for outcome in outcomes), ["superseded", "y"])
        self.assertEqual([row["name"] for row in checked["characters"]], ["Mara"])

    def test_a_new_row_copied_in_cast_line_shape_is_unpacked_into_its_fields(self):
        ledger = apply_delta(new_ledger(), {"new": [["Trag||m|quiet trainer of new singers", 2],
                                                    ["Pendel||m", "homesick supercargo", 1]]},
                             self.blocks, self.numbers)
        rows = {row["name"]: row for row in ledger["characters"]}
        self.assertEqual(sorted(rows), ["Pendel", "Trag"])
        self.assertEqual((rows["Trag"]["sex"], rows["Trag"]["note"], rows["Trag"]["first"]),
                         ("m", "quiet trainer of new singers", "b000002"))
        self.assertEqual((rows["Pendel"]["aliases"], rows["Pendel"]["note"], rows["Pendel"]["first"]),
                         ([], "homesick supercargo", "b000001"))

    def test_new_rows_with_a_leading_id_text_number_or_flattened_shape_are_read(self):
        def rows(new):
            ledger = apply_delta(new_ledger(), {"new": new}, self.blocks, self.numbers)
            return [(row["name"], row["aliases"], row["sex"], row["note"], row["first"]) for row in ledger["characters"]]

        self.assertEqual(rows([[3, "Captain Andurs||m", "ship captain", 2], [5, "Ford", None, "m", "apprentice", "1"]]),
                         [("Captain Andurs", [], "m", "ship captain", "b000002"), ("Ford", [], "m", "apprentice", "b000001")])
        self.assertEqual(rows([["cargo officer Malaine;Malaine", "", "f", "", 1]]),
                         [("cargo officer Malaine", ["Malaine"], "f", "", "b000001")])
        self.assertEqual(rows(["Senior Lieutenant Tallaf", "Tallaf", "engineer, humorous", "m"]),
                         [("Senior Lieutenant Tallaf", ["Tallaf"], "m", "engineer, humorous", None)])

    def test_a_voice_change_needs_a_quote_about_that_change(self):
        from epub_tts_narrator.narrator_ledger import change_supported
        self.assertTrue(change_supported("condition", "her voice came out as a rasp"))
        self.assertFalse(change_supported("condition", "she had aged a decade, her soft features gone"))
        self.assertTrue(change_supported("age", "Ten years later, Tom was a man"))
        self.assertTrue(change_supported("phone", "he answered the call"))
        self.assertFalse(change_supported("phone", "Protocol, process the following requisition."))
        self.assertTrue(change_supported("helmet", "She sealed her visor."))

    def test_aliases_must_be_short_names_that_do_not_name_another_entry(self):
        ledger = apply_delta(new_ledger(), {"new": [["Guild Master Lanzecki", "", "m", "", 1],
                                                    ["Carrik", "m;the crystal singer, tall and dark", "m", "", 2]]},
                             self.blocks, self.numbers)
        ledger = apply_delta(ledger, {"alias": [[2, "Lanzecki here"], [2, "the Guild Master"], [2, "Carr"],
                                                [1, "Lanzecki"]]}, self.blocks, self.numbers)
        rows = {row["name"]: row for row in ledger["characters"]}
        self.assertEqual(rows["Carrik"]["aliases"], ["Carr"])
        self.assertEqual(rows["Guild Master Lanzecki"]["aliases"], ["Lanzecki"])
        self.assertEqual(ledger["dropped"], {"alias_not_a_name": 2, "alias_names_other": 2})

    def test_an_alias_naming_an_entry_added_later_is_dropped_after_the_prep_read(self):
        from epub_tts_narrator.narrator_ledger import recheck_aliases
        ledger = apply_delta(new_ledger(), {"new": [["Carrik", "Mok|m;Lanzecki's shadow", "m", "", 1]]},
                             self.blocks, self.numbers)
        self.assertEqual(ledger["characters"][0]["aliases"], ["Mok", "Lanzecki's shadow"])
        ledger = apply_delta(ledger, {"new": [["Guild Master Lanzecki", "", "m", "", 2]]}, self.blocks, self.numbers)
        self.assertEqual(recheck_aliases(ledger)["characters"][0]["aliases"], ["Mok"])

    def test_series_characters_start_unseen_and_match_only_through_seen(self):
        ledger = new_ledger([{"id": "series_char_x", "name": "Mara Voss", "aliases": ["Mara"], "description": "keeper"}])
        updated = apply_delta(ledger, {"seen": [[1, 1, "Mara met Jon."]], "new": [["Jon", "", "m", "", 1]]},
                              self.blocks, self.numbers)
        self.assertEqual(updated["characters"][1]["id"], "2")
        self.assertTrue(updated["characters"][0]["seen"])
        self.assertEqual(updated["characters"][0]["series_evidence"][0]["block_id"], "b000001")


class NarratedAnalyzeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "cache"
        self.source = make_source(BOOK)

    def analyze(self, client, **kwargs):
        return narrated_analyze(self.source, client, self.cache, workers=kwargs.pop("workers", 2), **kwargs)

    def by_text(self, document):
        return {row["text"]: row for row in document["segments"]}

    def test_quotations_are_attributed_and_narration_stays_with_the_narrator(self):
        client = NarratorClient()
        document = self.analyze(client)
        self.assertEqual(readiness(document)["errors"], [])
        names = {role["id"]: next((c["name"] for c in document["cast"]["characters"] if c["id"] == role["character_id"]), None)
                 for role in document["cast"]["roles"]}
        rows = self.by_text(document)
        self.assertEqual(names[rows['"You never listen,"']["role_id"]], "Jon")
        self.assertEqual(rows['"You never listen,"']["attribution"]["basis"], "explicit")
        self.assertEqual(names[rows['"I thought,"']["role_id"]], "Mara")
        self.assertEqual(names[rows['"that you had gone."']["role_id"]], "Mara")
        self.assertEqual(names[rows['"No, you don\'t."']["role_id"]], "Elsie")
        self.assertEqual(rows["Chapter One"]["kind"], "heading")
        self.assertEqual(rows['The sign read "North Station" in faded paint.']["role_id"], "narrator_main")
        mara = next(c for c in document["cast"]["characters"] if c["name"] == "Mara")
        self.assertEqual(mara["evidence"], [{"block_id": "b000003", "quote": "Her voice was husky and low."}])
        modes = [call["mode"] for call in client.calls]
        self.assertEqual(modes.count("narrator_prep"), 1)
        self.assertNotIn("narrator_merge_check", modes)
        metrics = document["provenance"]["narrator_read"]["metrics"]
        self.assertEqual(metrics["total"]["provider_calls"], len(client.calls))
        self.assertGreater(metrics["stages"]["markup"]["prompt_tokens"], 0)

    def test_bare_numeric_quotation_and_item_ids_are_accepted_as_the_matching_ids(self):
        class NumericIds(NarratorClient):
            def complete(self, system, payload):
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_markup":
                    result["q"] = [[int(row[0][1:]) if index % 2 else row[0][1:], *row[1:]]
                                   for index, row in enumerate(result["q"])]
                for key in ("r", "v"):
                    if key in result:
                        result[key] = [[int(row[0][1:]), *row[1:]] for row in result[key]]
                return result

        client = NumericIds()
        document = self.analyze(client, repair_attempts=0)
        self.assertTrue(any(call["mode"] == "narrator_verify" for call in client.calls))
        self.assertTrue(any(call["mode"] == "narrator_voices" for call in client.calls))
        self.assertEqual(readiness(document)["errors"], [])
        self.assertFalse(any("correction" in call for call in client.calls))

    def test_action_beats_and_mixed_paragraphs_get_a_targeted_second_look(self):
        from epub_tts_narrator.narrator_read import verification_items
        source = make_source([[('"Fine." Elsie deadpanned. "Keep it."', "p"), ('"Later," Jon said.', "p")]])
        spans = quote_spans(source)
        ledger = new_ledger()
        ledger = apply_delta(ledger, {"new": [["Elsie", "", "f", "", 1], ["Jon", "", "m", "", 2]]},
                             {block["id"]: block["text"] for block in source["blocks"]}, {1: "b000001", 2: "b000002"})
        labels = {spans[0]["id"]: {"who": ("char", "2")}, spans[1]["id"]: {"who": ("char", "1")},
                  spans[2]["id"]: {"who": ("char", "2")}}
        reasons = {item["span"]["id"]: item["reason"] for item in verification_items(source, ledger, spans, labels)}
        self.assertEqual(reasons, {spans[0]["id"]: "beat_disagreement", spans[1]["id"]: "mixed_paragraph"})

    def test_self_checks_flag_an_absent_speaker_and_a_line_calling_its_own_name(self):
        from epub_tts_narrator.narrator_read import scene_checks
        source = make_source([[('Carrik steered the ship. "Good view," he told her.', "p"),
                               ('"Wait, Carrik."', "p"), ('Killa laughed. "Maybe."', "p")],
                              [('Lanzecki spoke. "Welcome."', "p")]])
        spans = quote_spans(source)
        ledger = apply_delta(new_ledger(), {"new": [["Carrik", "", "m", "", 1], ["Guild Master Lanzecki", "Lanzecki", "m", "", 4],
                                                    ["Killashandra", "Killa", "f", "", 3]]},
                             {block["id"]: block["text"] for block in source["blocks"]},
                             {1: "b000001", 3: "b000003", 4: "b000004"})
        labels = {spans[0]["id"]: {"who": ("char", "2")}, spans[1]["id"]: {"who": ("char", "1")},
                  spans[2]["id"]: {"who": ("char", "3")}, spans[3]["id"]: {"who": ("char", "2")}}
        flagged = scene_checks(source, ledger, spans, labels)
        # Lanzecki is never named in chapter one; Carrik calls out his own name.
        self.assertEqual(flagged[spans[0]["id"]], ("absent_speaker", ["1", "3"]))
        self.assertEqual(flagged[spans[1]["id"]][0], "self_address")
        self.assertNotIn(spans[2]["id"], flagged)
        self.assertNotIn(spans[3]["id"], flagged)

    def test_a_generic_speaker_far_from_its_description_is_flagged(self):
        from epub_tts_narrator.narrator_read import scene_checks
        source = make_source([[('The spaceport official bowed. "Welcome."', "p")],
                              [('Francu glared. "Hurry!"', "p"), ('"Sleep?" The indignation in Francu\'s voice was plain.', "p")]])
        spans = quote_spans(source)
        ledger = apply_delta(new_ledger(), {"new": [["spaceport official", "", "m", "", 1], ["Captain Francu", "", "m", "", 2]]},
                             {block["id"]: block["text"] for block in source["blocks"]}, {1: "b000001", 2: "b000002"})
        labels = {spans[0]["id"]: {"who": ("char", "1")}, spans[1]["id"]: {"who": ("char", "2")},
                  spans[2]["id"]: {"who": ("char", "1")}}
        flagged = scene_checks(source, ledger, spans, labels)
        self.assertEqual(flagged, {spans[2]["id"]: ("absent_speaker", ["2"])})

    def test_an_unsure_self_check_keeps_the_markup_label_but_an_unsure_tag_check_does_not(self):
        from epub_tts_narrator.narrator_read import Meter, _verify_items
        from epub_tts_narrator.pipeline import CachedRequests

        class Unsure:
            identity = {"model": "fixture", "base_url": "http://offline.invalid/v1"}

            def complete(self, system, payload):
                return {"r": [[item["id"], "?"] for item in payload["items"]]}

        source = make_source([[('"Good view," he told her.', "p"), ('"Later," Jon said.', "p")]])
        spans = quote_spans(source)
        ledger = apply_delta(new_ledger(), {"new": [["Carrik", "", "m", "", 1], ["Jon", "", "m", "", 2]]},
                             {block["id"]: block["text"] for block in source["blocks"]}, {1: "b000001", 2: "b000002"})
        labels = {spans[0]["id"]: {"who": ("char", "1")}, spans[1]["id"]: {"who": ("char", "1")}}
        items = [{"span": spans[0], "reason": "absent_speaker", "options": ["1", "2"]},
                 {"span": spans[1], "reason": "tag_disagreement", "options": ["2", "1"]}]
        client = Unsure()
        labels, stats = _verify_items(source, CachedRequests(client, self.cache, repair_attempts=0), Meter(client),
                                      ledger, spans, labels, items)
        self.assertEqual(labels[spans[0]["id"]]["who"], ("char", "1"))
        self.assertEqual(labels[spans[1]["id"]]["who"][0], "unresolved")
        self.assertEqual(stats["changed_by_reason"], {"tag_disagreement": 1})

    def test_a_line_left_unsure_by_a_self_check_is_still_rechecked_after_a_nearby_correction(self):
        from epub_tts_narrator.narrator_read import verify, Meter
        from epub_tts_narrator.pipeline import CachedRequests

        class Client:
            identity = {"model": "fixture", "base_url": "http://offline.invalid/v1"}

            def complete(self, system, payload):
                if payload["mode"] == "narrator_verify_tiebreak":
                    return {"r": [[item["id"], item["choices"][1]["id"]] for item in payload["items"]]}
                rows = []
                for item in payload["items"]:
                    # The beat-tagged line is Divinity's; the untagged absent-speaker line is
                    # left unsure among three options, but settled when asked Erika-or-Divinity.
                    if item["target"] == "q1" or item["options"] == ["1", "2"]:
                        rows.append([item["id"], "2"])
                    else:
                        rows.append([item["id"], "?"])
                return {"r": rows}

        source = make_source([[('Sal sat. "His loss." Divinity grinned. "Come in."', "p"),
                               ('"Thanks," Sal said.', "p"), ('"Sit," Divinity said.', "p")]])
        spans = quote_spans(source)
        ledger = apply_delta(new_ledger(), {"new": [["Erika Clifton", "", "f", "", 1], ["Divinity", "", "f", "", 1],
                                                    ["Sal", "", "m", "", 1]]},
                             {block["id"]: block["text"] for block in source["blocks"]}, {1: "b000001"})
        labels = {spans[0]["id"]: {"who": ("char", "1")}, spans[1]["id"]: {"who": ("char", "1")},
                  spans[2]["id"]: {"who": ("char", "3")}, spans[3]["id"]: {"who": ("char", "2")}}
        client = Client()
        labels, stats = verify(source, CachedRequests(client, self.cache, repair_attempts=0), Meter(client),
                               ledger, spans, labels)
        self.assertEqual(stats["reasons"]["absent_speaker"], 1)
        self.assertEqual([labels[span["id"]]["who"] for span in spans[:2]], [("char", "2"), ("char", "2")])

    def test_every_run_reports_its_self_check(self):
        document = self.analyze(NarratorClient())
        report = document["provenance"]["narrator_read"]["self_check"]
        self.assertEqual(set(report), {"lines_rechecked", "lines_changed", "duplicate_checks", "cast_warnings",
                                       "unresolved_lines"})

    def test_cast_audit_warns_about_description_names_and_shared_aliases(self):
        from epub_tts_narrator.narrator_read import cast_audit
        warnings = cast_audit({"characters": [
            {"id": "c1", "name": "Trag//m/unsmiling, reticent", "aliases": []},
            {"id": "c2", "name": "Carrik", "aliases": ["the Singer"]},
            {"id": "c3", "name": "Olin", "aliases": ["the Singer"]}]})
        self.assertEqual(sorted(warning["kind"] for warning in warnings), ["name_shape", "shared_name"])

    def test_a_verified_correction_rechecks_the_same_confusion_nearby_in_the_scene(self):
        from epub_tts_narrator.narrator_read import scene_followups
        source = make_source([[('"One," Jon said.', "p"), ('"Two."', "p"), ('"Three."', "p")],
                              [('"Four."', "p")]])
        spans = quote_spans(source)
        labels = {span["id"]: {"who": ("char", "2")} for span in spans}
        items = scene_followups(source, spans, labels, [(spans[0]["id"], "2", "1")], {spans[0]["id"]})
        self.assertEqual([item["span"]["id"] for item in items], [spans[1]["id"], spans[2]["id"]])
        self.assertEqual(items[0]["options"], ["2", "1"])

    def test_extra_nesting_and_unexpected_value_types_never_crash_validation(self):
        class OddShapes(NarratorClient):
            def complete(self, system, payload):
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_voices":
                    result["v"] = [result["v"]]
                if payload.get("mode") == "narrator_prep":
                    result["nar"] = [["third"], None]
                    result["new"] = [[*row[:2], ["f"], *row[3:]] for row in result["new"]]
                return result

        document = self.analyze(OddShapes(), repair_attempts=0)
        self.assertEqual(readiness(document)["errors"], [])

    def test_a_surname_tag_names_its_character_and_the_placeholder_name_is_not_a_speaker(self):
        from epub_tts_narrator.narrator_ledger import name_index, tag_character
        from epub_tts_narrator.narrator_read import _parse_who
        ledger = apply_delta(new_ledger(), {"new": [["Captain Francu", "", "m", "", 1], ["spaceport official", "", "m", "", 1]]},
                             {"b000001": "x"}, {1: "b000001"})
        index = name_index(ledger)
        self.assertEqual(tag_character(index, "Francu"), "1")
        self.assertEqual(_parse_who("+Name", ledger, index), ("unresolved", []))

    def test_an_untagged_quotation_is_shown_its_paragraphs_single_named_speaker(self):
        from epub_tts_narrator.narrator_read import Meter, _verify_items
        from epub_tts_narrator.pipeline import CachedRequests

        class Recording:
            identity = {"model": "fixture", "base_url": "http://offline.invalid/v1"}
            items = []

            def complete(self, system, payload):
                self.items.extend(payload["items"])
                return {"r": [[item["id"], "?"] for item in payload["items"]]}

        source = make_source([[('"His loss." Divinity grinned. "Come in."', "p")]])
        spans = quote_spans(source)
        ledger = apply_delta(new_ledger(), {"new": [["Erika Clifton", "", "f", "", 1], ["Divinity", "", "f", "", 1]]},
                             {"b000001": source["blocks"][0]["text"]}, {1: "b000001"})
        labels = {span["id"]: {"who": ("char", "1")} for span in spans}
        client = Recording()
        _verify_items(source, CachedRequests(client, self.cache, repair_attempts=0),
                      Meter(client), ledger, spans, labels,
                      [{"span": spans[1], "reason": "absent_speaker", "options": ["1", "2"]}])
        self.assertEqual(client.items[0]["named_beside"], ["Divinity"])

    def test_a_name_contained_in_another_nominates_a_duplicate_check(self):
        from epub_tts_narrator.narrator_read import duplicate_candidates
        ledger = apply_delta(new_ledger(), {"new": [["Captain Andurs", "", "m", "", 1], ["Andurs", "", "m", "", 2],
                                                    ["Mara", "", "f", "", 1]]},
                             {"b000001": "x", "b000002": "y"}, {1: "b000001", 2: "b000002"})
        self.assertEqual(duplicate_candidates(ledger), [("2", "1")])

    def test_name_overlapping_entries_merge_only_after_a_same_person_check(self):
        self.source = make_source([[('"Ready," Barry said.', "p"), ('Much later, Barry Francis waved. "Go," Barry Francis said.', "p")]])
        names = {"Barry Francis": "m", "Barry": "m"}
        for answer, expected_roles in (("y", 1), ("n", 2)):
            with self.subTest(answer=answer):
                self.cache = self.cache.parent / f"cache-{answer}"
                document = self.analyze(NarratorClient(duplicate_answer=answer, names=names))
                roles = {row["role_id"] for row in document["segments"] if row["kind"] == "dialogue"}
                self.assertEqual(len(roles), expected_roles)
                outcomes = document["provenance"]["narrator_read"]["duplicate_checks"]
                self.assertEqual([row["verdict"] for row in outcomes], [answer])

    def markup_with(self, rewrite, **client_options):
        """Run with markup answers rewritten by ``rewrite(qid_by_text, result)``."""
        class Rewriting(NarratorClient):
            def complete(inner, system, payload):
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_markup":
                    qids = {}
                    for line in payload["text"].splitlines():
                        number, _, rest = line.partition("|")
                        if number.isdigit():
                            # "#<paragraph text>" -> its paragraph number, for narration notes.
                            qids["#" + rest] = number
                        for match in re.finditer(r"\u2039(q\d+)\u203a", line):
                            qids[line[match.end():].split('"')[1]] = match.group(1)
                    rewrite(qids, result)
                return result

        return self.analyze(Rewriting(**client_options))

    def test_lines_said_together_become_simultaneous_only_with_source_evidence(self):
        def unison(quote):
            def rewrite(qids, result):
                result["q"] = [[qids["We agree,"], "1&2", "", "", quote]]
            return rewrite

        cases = (('"We agree," Mara and Jon said.', "Mara and Jon said", True),       # quoted by the model
                 ('"We agree," Mara and Jon said together.', "", True),             # narration says so
                 ('"We agree," Mara said, and Jon nodded.', "", False))             # no evidence: human chooses
        for number, (text, quote, simultaneous) in enumerate(cases):
            with self.subTest(text=text):
                self.source = make_source([[(text, "p")]])
                self.cache = self.cache.parent / f"cache-unison-{number}"
                document = self.markup_with(unison(quote), names={"Mara": "f", "Jon": "m"})
                row = self.by_text(document)['"We agree,"']
                self.assertEqual(readiness(document)["errors"], [])
                if simultaneous:
                    self.assertIsNone(row["role_id"])
                    self.assertEqual(len(row["simultaneous_speakers"]), 2)
                    self.assertTrue(row["attribution"]["evidence"])
                else:
                    self.assertNotIn("simultaneous_speakers", row)
                    self.assertEqual(row["attribution"]["basis"], "unresolved")

    def test_a_direction_needs_no_citation_and_keeps_one_only_when_it_is_found(self):
        self.source = make_source([[('"Stay," Mara whispered hoarsely.', "p"), ('"Go," Jon said.', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Stay,"], "1", "", "whispered", "Mara whispered hoarsely"],
                           [qids["Go,"], "2", "", "sharp, impatient", "invented support"]]

        document = self.markup_with(rewrite)
        rows = self.by_text(document)
        self.assertEqual(rows['"Stay,"']["delivery"],
                         {"direction": "whispered", "evidence": [{"block_id": "b000001", "quote": "Mara whispered hoarsely"}]})
        # The line makes the reading clear; a quote that is not in the scene is not kept.
        self.assertEqual(rows['"Go,"']["delivery"], {"direction": "sharp, impatient", "evidence": []})
        stats = document["provenance"]["narrator_read"]["segments"]
        self.assertEqual((stats["delivery"], stats["delivery_cited"]), (2, 1))
        self.assertEqual(readiness(document)["errors"], [])

    def test_a_feeling_from_earlier_in_the_scene_cites_the_text_that_shows_it(self):
        self.source = make_source([[("Mara's hands shook as the footsteps came closer.", "p"),
                                    ('"Is someone there?"', "p"), ('"Ah," Jon said.', "p")],
                                   [('"Fine," Jon said.', "p"), ('"Mara!"', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Is someone there?"], "1", "", "hushed and frightened", "Mara's hands shook"],
                           # A quote wrapped in quotation marks by the model.
                           [qids["Ah,"], "2", "", "soft, pleased", '"Ah"'],
                           # No quote: the direction stands on the line itself.
                           [qids["Fine,"], "2", "", "weary, flat"],
                           # Text from another chapter is not this scene.
                           [qids["Mara!"], "1", "", "frantic", "footsteps came closer"]]

        document = self.markup_with(rewrite)
        rows = self.by_text(document)
        self.assertEqual(rows['"Is someone there?"']["delivery"],
                         {"direction": "hushed and frightened",
                          "evidence": [{"block_id": "b000001", "quote": "Mara's hands shook"}]})
        self.assertEqual(rows['"Ah,"']["delivery"],
                         {"direction": "soft, pleased", "evidence": [{"block_id": "b000003", "quote": "Ah"}]})
        self.assertEqual(rows['"Fine,"']["delivery"], {"direction": "weary, flat", "evidence": []})
        self.assertEqual(rows['"Mara!"']["delivery"], {"direction": "frantic", "evidence": []})
        stats = document["provenance"]["narrator_read"]["segments"]
        self.assertEqual((stats["delivery"], stats["delivery_cited"]), (4, 2))
        self.assertEqual(readiness(document)["errors"], [])

    def test_a_sound_cited_by_its_speech_tag_is_placed_on_the_quotation(self):
        self.source = make_source([[('Mara laughed. "Wait for me, wait," she said.', "p"),
                                    ('"Never," Jon said, and sighed.', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Wait for me, wait,"], "1"], [qids["Never,"], "2"]]
            result["p"] = [[qids["Wait for me, wait,"], "laugh", "laughed", "before"],
                           [qids["Never,"], "sigh", "sighed", "after"],
                           # Text found nowhere in the paragraph is still dropped.
                           [qids["Never,"], "cough", "coughed", "after"]]

        document = self.markup_with(rewrite, review_answer="keep")
        stats = document["provenance"]["narrator_read"]["segments"]
        self.assertEqual((stats["attached"], stats["dropped"]), (2, 1))
        rows = self.by_text(document)
        laugh = rows['"Wait for me, wait,"']["performance_controls"]["controls"]
        sigh = rows['"Never,"']["performance_controls"]["controls"]
        # Before the first "Wait" (not the second "wait"), and after the last word "Never".
        self.assertEqual([(c["kind"], c["sound"], c["offset"]) for c in laugh], [("vocalization", "laugh", 1)])
        self.assertEqual([(c["kind"], c["sound"], c["offset"]) for c in sigh], [("vocalization", "sigh", len('"Never'))])

    def test_delivery_changes_mid_line_narration_reads_and_italics_skip_the_review(self):
        self.source = make_source([[("The cellar door creaked open in the dark.", "p"),
                                    ('"I got the job! But it means moving away," Mara said.', "p"),
                                    ('"You never listen," Jon said.', "p"),
                                    ('"Fine," Jon said.', "p"), ('"Is it far?" Mara asked.', "p")]])
        never = self.source["blocks"][2]["text"].index("never")
        self.source["blocks"][2]["inline_spans"] = [{"start": never, "end": never + 5, "kind": "emphasis", "origin": "tag"}]
        # A whole italic line (a thought, a title) is not stress.
        self.source["blocks"][3]["inline_spans"] = [{"start": 0, "end": 7, "kind": "emphasis", "origin": "tag"}]

        def rewrite(qids, result):
            job, never_q = qids["I got the job! But it means moving away,"], qids["You never listen,"]
            # A change can ride on the line's own row, or come as a separate note.
            result["q"] = [[job, "1", "", "thrilled, breathless", "", [["But it means", "uncertain, deflating"],
                                                                       ["not in the line", "dropped"]]],
                           [never_q, "2", "", "tired, flat"], [qids["Fine,"], "2"],
                           # One pair written as JSON text is still a change.
                           [qids["Is it far?"], "1", "", "curious", "", '["far?", "worried"]']]
            result["p"] = [[never_q, "then", "never listen", "voice rising, exasperated"],
                           # A change at the first word is the line's own delivery.
                           [qids["Fine,"], "then", "Fine", "clipped, final"],
                           [qids["#The cellar door creaked open in the dark."], "read", "tense, hushed"]]

        # The review would remove anything it saw; scene-read directions never reach it.
        document = self.markup_with(rewrite, review_answer="remove")
        rows = self.by_text(document)
        self.assertEqual(rows["The cellar door creaked open in the dark."]["delivery"]["direction"], "tense, hushed")
        job = rows['"I got the job! But it means moving away,"']
        self.assertEqual(job["delivery"]["direction"], "thrilled, breathless")
        self.assertEqual([(c["tag"], c["text"], c["direction"]) for c in job["performance_controls"]["controls"]],
                         [("style", 'But it means moving away,"', "uncertain, deflating")])
        line = rows['"You never listen,"']
        self.assertEqual(sorted((c["tag"], c["text"], c["direction"]) for c in line["performance_controls"]["controls"]),
                         [("emphasis", "never", ""), ("style", 'never listen,"', "voice rising, exasperated")])
        self.assertEqual(rows['"Fine,"']["delivery"]["direction"], "clipped, final")
        self.assertNotIn("performance_controls", rows['"Fine,"'])
        stats = document["provenance"]["narrator_read"]["segments"]
        self.assertEqual([(c["text"], c["direction"]) for c in rows['"Is it far?"']["performance_controls"]["controls"]],
                         [('far?"', "worried")])
        self.assertEqual((stats["shifts"], stats["narration_reads"], stats["italic_emphasis"]), (3, 1, 1))
        self.assertEqual(readiness(document)["errors"], [])

    def test_tie_break_selection_is_published_with_existing_delivery_evidence(self):
        self.source = make_source([[('"Stay," Mara whispered.', "p"), ('"Go," Jon said.', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Stay,"], "2", "", "whispered", "Mara whispered hoarsely"],
                           [qids["Go,"], "2"]]

        document = self.markup_with(rewrite)
        row = self.by_text(document)['"Stay,"']
        self.assertEqual(row["role_id"], "role_1")
        self.assertEqual(row["delivery"],
                         {"direction": "whispered", "evidence": [{"block_id": "b000001", "quote": "Mara whispered"}]})
        self.assertEqual(document["provenance"]["narrator_read"]["verification"]["tie_breaks"], 1)

    def test_spelling_edits_and_cues_go_through_the_remove_only_review(self):
        # "said": no cue from the narration, so every note here is the model's own.
        self.source = make_source([[('"Nooooo, wait," Mara said.', "p"), ("The door went bang.", "p")]])

        def rewrite(qids, result):
            qid = qids["Nooooo, wait,"]
            result["q"] = [[qid, "1"]]
            result["p"] = [[qid, "sp", "Nooooo", "No", "drawn out"], [qid, "whisper", "wait"],
                           [qid, "laugh", "wait", "after"], [qid, "sp", "absent text", "x", "y"], [2, "fast", "went bang"]]

        for answer, kept in (("keep", True), ("remove", False)):
            with self.subTest(answer=answer):
                self.cache = self.cache.parent / f"cache-review-{answer}"
                document = self.markup_with(rewrite, review_answer=answer)
                self.assertEqual(readiness(document)["errors"], [])
                rows = self.by_text(document)
                quote_row = next(row for row in document["segments"] if row["text"].startswith('"Nooooo'))
                self.assertNotIn("inline_proposals", quote_row)
                counts = document["provenance"]["inline_performance"]["counts"]
                stats = document["provenance"]["narrator_read"]["segments"]
                self.assertEqual((stats["attached"], stats["dropped"]), (4, 1))
                # Stretched-letter edits pass the existing mechanical check, not the model review.
                self.assertEqual(quote_row["spoken_text"][:3], '"No')
                self.assertEqual(counts.get("mechanical"), 1)
                controls = quote_row.get("performance_controls", {}).get("controls", [])
                if kept:
                    self.assertEqual(counts.get("kept"), 3)
                    self.assertEqual(len(controls), 2)
                else:
                    self.assertEqual(counts.get("removed"), 3)
                    self.assertEqual(controls, [])
                self.assertIn("The door went bang.", rows)

    def test_the_narration_s_cues_for_a_speaker_are_applied_without_review(self):
        self.source = make_source([[('"Run!" Mara shouted.', "p"), ('"Fine," Mara laughed.', "p"),
                                    ('"Hush," Mara whispered.', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Run!"], "1"], [qids["Fine,"], "1", "", "wry"], [qids["Hush,"], "1"]]
            result.pop("p", None)

        self.cache = self.cache.parent / "cache-cues"
        document = self.markup_with(rewrite, review_answer="remove")
        self.assertEqual(readiness(document)["errors"], [])
        lines = {row["text"]: row for row in document["segments"] if row["kind"] == "dialogue"}
        self.assertEqual(lines['"Run!"']["delivery"]["direction"], "shouting")
        self.assertEqual(lines['"Fine,"']["delivery"]["direction"], "wry")
        self.assertIn({"kind": "vocalization", "offset": len('"Fine'), "sound": "laugh"},
                      lines['"Fine,"']["performance_controls"]["controls"])
        self.assertEqual([(c["tag"], c["text"]) for c in lines['"Hush,"']["performance_controls"]["controls"]],
                         [("whisper", '"Hush,"')])
        stats = document["provenance"]["narrator_read"]["segments"]
        self.assertEqual((stats["cue_sounds"], stats["cue_scoped"], stats["cue_tones"]), (1, 1, 1))

    def test_automatic_cue_sounds_follow_their_clause_around_the_line(self):
        self.source = make_source([[('"Robbers?" Faldor laughed.', "p"),
                                    ('Silk laughed. "Durnik," he said.', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Robbers?"], "1"], [qids["Durnik,"], "2"]]
            result.pop("p", None)

        document = self.markup_with(rewrite, names={"Faldor": "m", "Silk": "m"}, review_answer="remove")
        self.assertEqual(readiness(document)["errors"], [])
        rows = self.by_text(document)
        from epub_tts_narrator.renderer.engines import prepare_synthesis
        for text, word, position in (('"Robbers?"', "Robbers", "after"),
                                     ('"Durnik,"', "Durnik", "before")):
            with self.subTest(text=text, position=position):
                controls = rows[text]["performance_controls"]["controls"]
                self.assertEqual(controls, [{"kind": "vocalization", "sound": "laugh",
                                             "offset": text.index(word) + (len(word) if position == "after" else 0)}])
                _, compiled, _ = prepare_synthesis(text, rows[text]["delivery"]["direction"],
                                                   performance=rows[text]["performance_controls"])
                assert isinstance(compiled, str)
                sound = compiled.index("[laughs]")
                self.assertEqual(compiled.index(word) < sound, position == "after")

    def test_a_cue_tone_keeps_the_tail_of_a_long_delivery_direction(self):
        self.source = make_source([[('"Wait," Faldor said softly.', "p")]])
        original = "calm but firm, with each careful word building toward a resolute final warning"

        def rewrite(qids, result):
            result["q"] = [[qids["Wait,"], "1", "", original]]
            result.pop("p", None)

        document = self.markup_with(rewrite, names={"Faldor": "m"}, review_answer="remove")
        direction = self.by_text(document)['"Wait,"']["delivery"]["direction"]
        self.assertEqual(direction, f"soft and quiet, {original}")
        self.assertLessEqual(len(direction), 120)

    def test_soft_cues_compile_softly_while_explicit_whispering_stays_scoped(self):
        self.source = make_source([[('"Amazing," Mara breathed.', "p"), ('"Stay," Mara said softly.', "p"),
                                    ('"Hush," Mara whispered.', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Amazing,"], "1", "", "awed, admiring"],
                           [qids["Stay,"], "1", "", "whispering, reassuring"], [qids["Hush,"], "1"]]
            result.pop("p", None)

        document = self.markup_with(rewrite, review_answer="remove")
        self.assertEqual(readiness(document)["errors"], [])
        rows = self.by_text(document)
        from epub_tts_narrator.renderer.engines import prepare_synthesis
        for text, direction in (('"Amazing,"', "soft and quiet, awed, admiring"),
                                ('"Stay,"', "soft and quiet, reassuring")):
            with self.subTest(text=text):
                row = rows[text]
                self.assertEqual(row["delivery"]["direction"], direction)
                _, compiled, _ = prepare_synthesis(row["text"], row["delivery"]["direction"],
                                                   performance=row.get("performance_controls"))
                self.assertTrue(compiled.startswith("[soft and quiet, "))
                self.assertNotIn("whisper", compiled.casefold())
        whispered = rows['"Hush,"']
        _, compiled, _ = prepare_synthesis(whispered["text"], whispered["delivery"]["direction"],
                                           performance=whispered["performance_controls"])
        self.assertEqual(compiled, "<whisper>Hush,</whisper>")

    def test_truncated_markup_is_split_at_paragraph_boundaries_and_retried(self):
        from epub_tts_narrator.provider import OutputTruncatedError

        class Truncating(NarratorClient):
            def complete(inner, system, payload):
                if payload.get("mode") == "narrator_markup" and len(payload["text"].splitlines()) > 2:
                    inner.calls.append(deepcopy(payload))
                    raise OutputTruncatedError("The model output was truncated. Increase the output token limit or reduce the input.")
                return super().complete(system, payload)

        client = Truncating()
        document = self.analyze(client, workers=1)
        self.assertEqual(readiness(document)["errors"], [])
        self.assertGreater(document["provenance"]["narrator_read"]["subdivided_markup_groups"], 0)
        names = {role["id"]: next((c["name"] for c in document["cast"]["characters"] if c["id"] == role["character_id"]), None)
                 for role in document["cast"]["roles"]}
        self.assertEqual(names[self.by_text(document)['"You never listen,"']["role_id"]], "Jon")

    def test_a_truncated_prep_reply_is_read_again_in_two_halves(self):
        class RunawayPrep(NarratorClient):
            def complete(self, system, payload):
                if payload.get("mode") == "narrator_prep" and payload["text"].count("\n") >= 4:
                    from epub_tts_narrator.provider import OutputTruncatedError
                    raise OutputTruncatedError("output token limit")
                return super().complete(system, payload)

        client = RunawayPrep()
        document = self.analyze(client)
        self.assertEqual(readiness(document)["errors"], [])
        preps = [call for call in client.calls if call["mode"] == "narrator_prep"]
        self.assertGreaterEqual(len(preps), 2)
        self.assertTrue(all(call["text"].count("\n") < 4 for call in preps))
        self.assertIn("Mara", {person["name"] for person in document["cast"]["characters"]})

    def test_an_alias_naming_a_different_person_is_split_off_after_a_check(self):
        from epub_tts_narrator.narrator_read import alias_conflicts
        self.source = make_source([[('"Coffee," Proof said. While Anderson sleeps, Proof works.', "p"),
                                    ('"Morning," Anderson said.', "p")]])

        class AliasedProof(NarratorClient):
            def complete(inner, system, payload):
                if payload.get("mode") == "narrator_alias_check":
                    inner.calls.append(deepcopy(payload))
                    return {"r": [[item["id"], inner.alias_answer] for item in payload["items"]]}
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_prep":
                    result = {"new": [["Anderson", "Proof;Andy", "m", "", 1]]}
                return result

        for answer, separate in (("n", True), ("y", False)):
            with self.subTest(answer=answer):
                self.cache = self.cache.parent / f"cache-alias-{answer}"
                client = AliasedProof(names={"Anderson": "m", "Proof": "m"})
                client.alias_answer = answer
                document = self.analyze(client)
                outcomes = document["provenance"]["narrator_read"]["alias_checks"]
                self.assertEqual([(row["alias"], row["verdict"]) for row in outcomes], [("Proof", answer)])
                names = {c["name"] for c in document["cast"]["characters"]}
                self.assertEqual("Proof" in names, separate)
        # Parts and shortenings of the main name are never nominated.
        ledger = apply_delta(new_ledger(), {"new": [["Anderson Royce", "Anders;Royce", "m", "", 1]]},
                             {"b000001": "Anderson and Anders and Royce."}, {1: "b000001"})
        self.assertEqual(alias_conflicts(make_source([[("Anderson and Anders and Royce.", "p")]]), ledger), [])

    def test_operator_notes_reach_cast_and_attribution_requests_only_when_configured(self):
        from epub_tts_narrator.narrator_prompts import HINTS

        class Recording(NarratorClient):
            def complete(self, system, payload):
                self.systems = [*getattr(self, "systems", []), (payload.get("mode"), system)]
                return super().complete(system, payload)

        plain = Recording()
        document = self.analyze(plain)
        # Without notes, requests (and so cache identities) are exactly as before.
        self.assertFalse(any("hints" in call for call in plain.calls))
        self.assertFalse(any(HINTS in system for _, system in plain.systems))
        self.assertEqual(document["provenance"]["narrator_read"]["hints"], {})

        noted = Recording()
        self.cache = self.cache.parent / "noted-cache"
        document = self.analyze(noted, cast_notes=" Jon and Mara are siblings. ",
                                attribution_notes="Elsie often speaks from offstage.")
        hints = {}
        for call in noted.calls:
            hints.setdefault(call["mode"], set()).add(call.get("hints"))
        self.assertEqual(hints["narrator_prep"], {"Jon and Mara are siblings."})
        self.assertEqual(hints["narrator_markup"], {"Elsie often speaks from offstage."})
        self.assertEqual(hints["narrator_voices"], {None})
        systems = {mode: HINTS in system for mode, system in noted.systems}
        self.assertEqual((systems["narrator_prep"], systems["narrator_markup"], systems["narrator_voices"]),
                         (True, True, False))
        self.assertEqual(document["provenance"]["narrator_read"]["hints"],
                         {"cast": "Jon and Mara are siblings.", "attribution": "Elsie often speaks from offstage."})

    def test_every_designed_voice_states_its_gender(self):
        from epub_tts_narrator.narrator_read import gendered_description
        self.assertEqual(gendered_description("Deep gravelly old voice.", "m"), "Male voice; deep gravelly old voice.")
        self.assertEqual(gendered_description("Bright quick voice.", "f"), "Female voice; bright quick voice.")
        self.assertEqual(gendered_description("Androgynous, cool and dry.", "m"), "Androgynous, cool and dry.")
        self.assertIsNone(gendered_description("Bright quick voice.", "?"))

        class Genderless(NarratorClient):
            unknown_too = False

            def complete(self, system, payload):
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_voices":
                    # A designer reply that leaves the gender out.
                    sexes = {row[0]: row[2] for row in payload["voices"]}
                    result["v"] = [[row[0], "A clear steady voice.", *row[2:]]
                                   if sexes[row[0]] in "mf" or self.unknown_too else row for row in result["v"]]
                return result

        document = self.analyze(Genderless(), repair_attempts=0)
        for person in document["cast"]["characters"]:
            voice = next(item for item in document["cast"]["voices"] if item["id"] == person["voice_id"])
            self.assertRegex(voice["description"], r"^(Male|Female) voice; ", person["name"])
        # With no known sex to fall back on, a genderless design is sent back to the model.
        client = Genderless()
        client.unknown_too = True
        self.cache = self.cache.parent / "genderless-cache"
        with self.assertRaisesRegex(ValueError, 'must begin with "Male", "Female" or "Androgynous"'):
            self.analyze(client, repair_attempts=0)

    def test_a_voice_described_with_the_books_own_words_is_sent_back(self):
        from epub_tts_narrator.narrator_read import book_terms
        source = make_source([[('Mara rode back to Sendaria. "Wait," Mara said, in her Sendarian burr.', "p"),
                               ("The Sendarian farmers came from Sendaria, past the Guild hall. The Guild met.", "p"),
                               ("He bowed to His Majesty, and his sword hung at his side; his horse waited.", "p"),
                               ("They spoke to His Majesty, and his voice was his own.", "p"),
                               ("The ancient king was sour, an ancient and sour old king.", "p")]])
        ledger = apply_delta(new_ledger(), {"new": [["Mara", "", "f", "", 1],
                                                    ["Belgarath", "Ancient One/Old Sour/King Wolf", "m", "", 1]]},
                             {block["id"]: block["text"] for block in source["blocks"]}, {1: "b000001"})
        terms = book_terms(source, ledger)
        self.assertTrue({"mara", "sendaria", "sendarian", "guild", "belgarath", "wolf"} <= terms)
        # Everyday words in an alias stay usable when the book mostly writes them in lowercase.
        self.assertFalse({"scottish", "american", "his", "ancient", "sour", "king"} & terms)

        class BookWords(NarratorClient):
            def complete(self, system, payload):
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_voices" and "real-world terms" not in str(payload.get("correction", "")):
                    result["v"] = [[row[0], "A warm female voice with a rural Sendarian burr.", *row[2:]]
                                   for row in result["v"]]
                return result

        client = BookWords()
        self.source = source
        document = self.analyze(client, repair_attempts=1)
        mara = next(person for person in document["cast"]["characters"] if person["name"] == "Mara")
        described = next(voice for voice in document["cast"]["voices"] if voice["id"] == mara["voice_id"])["description"]
        self.assertNotIn("Sendarian", described)
        corrections = [call for call in client.calls if call.get("mode") == "narrator_voices" and "correction" in call]
        self.assertTrue(corrections)

    def test_characters_of_different_backgrounds_get_their_assigned_accents(self):
        client = NarratorClient()
        self.analyze(client)
        accents = next(call for call in client.calls if call["mode"] == "narrator_accents")
        first, second = accents["characters"][0][0], accents["characters"][1][0]
        notes = {row[1]: row[3] for call in client.calls if call["mode"] == "narrator_voices" for row in call["voices"]}
        self.assertTrue(notes[first].endswith("; accent: Scottish"))
        self.assertTrue(notes[second].endswith("; accent: Irish"))

    def test_configured_notes_combine_as_in_the_consensus_pipeline(self):
        from epub_tts_narrator.narrator_read import settings_notes
        self.assertEqual(settings_notes({"cast_review_notes": "A", "cast_reconciliation_notes": ["B", ""],
                                         "attribution_review_notes": ""}),
                         {"cast_notes": "A\nB", "attribution_notes": "A"})
        self.assertEqual(settings_notes({"cast_review_notes": "", "cast_reconciliation_notes": [],
                                         "attribution_review_notes": "C"}),
                         {"cast_notes": "", "attribution_notes": "C"})

    def test_voice_changes_become_a_derived_voice_and_equipment_effects_with_line_exceptions(self):
        self.source = make_source([
            [("Chapter One", "h1"), ('"Hi," Jon said.', "p"),
             ("Years later Jon's voice had deepened into a man's.", "p"), ('"Hello again," Jon said.', "p"),
             ("Mara pulled on the helmet.", "p"), ('"Can you hear me?" Mara said.', "p"),
             ('"Visor up," Mara said, lifting it.', "p"), ("Mara took off the helmet.", "p"),
             ('"Better," Mara said.', "p")],
            [("Chapter Two", "h1"), ("Mara pulled on the helmet again.", "p"), ('"Ready," Mara said.', "p")]])

        class Changing(NarratorClient):
            def complete(self, system, payload):
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_prep":
                    numbers = {line.partition("|")[2]: int(line.partition("|")[0]) for line in payload["text"].splitlines()}
                    deep = next(number for text, number in numbers.items() if "deepened" in text)
                    on = [number for text, number in numbers.items() if "pulled on the helmet" in text]
                    off = next(number for text, number in numbers.items() if "took off" in text)
                    result["change"] = [["Jon", deep, "voice had deepened into a man's", "age", "a grown man now, deeper"],
                                        *[["Mara", number, "pulled on the helmet", "helmet", ""] for number in on],
                                        ["Jon", deep, "not in the text", "age", "x"]]
                    result["end"] = [["Mara", off, "took off the helmet", "helmet"]]
                if payload.get("mode") == "narrator_markup":
                    visor = next(match.group(1) for line in payload["text"].splitlines() if "Visor up" in line
                                 for match in [re.search(r"‹(q\d+)›", line)] if match)
                    result["p"] = [[visor, "as", "normal"]]
                return result

        client = Changing()
        document = self.analyze(client)
        self.assertEqual(readiness(document)["errors"], [])
        # Both versions are designed side by side: before the change, and after it.
        names = [row[1] for call in client.calls if call["mode"] == "narrator_voices" for row in call["voices"]]
        self.assertIn("Jon (at the start of the book, before any change)", names)
        self.assertIn("Jon (later: a grown man now, deeper)", names)
        rows = self.by_text(document)
        roles = {role["id"]: role for role in document["cast"]["roles"]}
        voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
        hi, again = roles[rows['"Hi,"']["role_id"]], roles[rows['"Hello again,"']["role_id"]]
        self.assertEqual(hi["character_id"], again["character_id"])
        jon = next(person for person in document["cast"]["characters"] if person["name"] == "Jon")
        # The grown voice at the end of the book is Jon's main voice; the young
        # voice before the change is derived (cloned) from it.
        self.assertEqual(again["voice_id"], jon["voice_id"])
        entry = {row[1]: row[0] for call in client.calls if call["mode"] == "narrator_voices" for row in call["voices"]}
        later, start = entry["Jon (later: a grown man now, deeper)"], entry["Jon (at the start of the book, before any change)"]
        self.assertEqual(voices[jon["voice_id"]]["description"], f"A clear male voice number {later[1:]}.")
        young = voices[hi["voice_id"]]
        self.assertEqual(young["derived_from"]["voice_id"], jon["voice_id"])
        self.assertFalse(young["derived_from"]["through_end"])
        self.assertEqual(young["description"], f"A clear male voice number {start[1:]}.")
        self.assertEqual(young["derived_from"]["instruction"], young["description"])
        self.assertEqual(rows['"Can you hear me?"']["voice_effect"]["preset"], "helmet")
        self.assertNotIn("voice_effect", rows['"Visor up,"'])     # the line's own exception
        self.assertNotIn("voice_effect", rows['"Better,"'])       # after the helmet came off
        self.assertEqual(rows['"Ready,"']["voice_effect"]["preset"], "helmet")
        # Chapter two's helmet never comes off in the text: it ends with its chapter, flagged for review.
        self.assertTrue(any(issue["kind"] == "voice_change" for issue in document["issues"]))
        self.assertEqual(document["provenance"]["narrator_read"]["dropped_ledger_claims"].get("change"), 1)

    def test_bracketed_ability_names_never_reach_spoken_text(self):
        self.source = make_source([[("Chapter One", "h1"),
                                    ("Mara pulled on the helmet and called up [Cracked Phantom Armor].", "p"),
                                    ('"Try [Cracked Phantom Armor] now," Mara said.', "p"),
                                    ('"It holds," Jon said.', "p")]])

        class Helmeted(NarratorClient):
            def complete(self, system, payload):
                result = super().complete(system, payload)
                if payload.get("mode") == "narrator_prep":
                    number = next(int(line.partition("|")[0]) for line in payload["text"].splitlines()
                                  if "pulled on the helmet" in line)
                    result["change"] = [["Mara", number, "pulled on the helmet", "helmet", ""]]
                return result

        document = self.analyze(Helmeted())
        self.assertEqual(readiness(document)["errors"], [])
        for row in document["segments"]:
            spoken = row.get("spoken_text", row["text"])
            self.assertFalse(set(spoken) & set("[]<>"), row)
        pieces = [row for row in document["segments"] if row["block_id"] == "b000003"]
        ability = next(row for row in pieces if "Cracked" in row["text"])
        self.assertEqual(ability["performance"]["kind"], "bracketed")
        self.assertEqual(ability["spoken_text"], "Cracked Phantom Armor")
        # Every piece of Mara's split line keeps her role and the helmet effect.
        spoken_pieces = [row for row in pieces if row["kind"] == "dialogue"]
        self.assertEqual({row["role_id"] for row in spoken_pieces}, {spoken_pieces[0]["role_id"]})
        self.assertTrue(all(row.get("voice_effect", {}).get("preset") == "helmet" for row in spoken_pieces))

    def test_unsettled_speaker_stays_unresolved_for_human_choice_not_the_narrator(self):
        document = self.analyze(NarratorClient(verify_answer="?"))
        row = self.by_text(document)['"Nobody asked."']
        self.assertIsNone(row["role_id"])
        self.assertEqual(row["attribution"]["basis"], "unresolved")
        self.assertTrue(row["attribution"]["review_required"])
        self.assertTrue(row["attribution"]["candidates"])
        self.assertFalse(readiness(document)["ready"])

    def test_rerun_replays_cached_responses_without_provider_calls(self):
        first = self.analyze(NarratorClient())
        again = NarratorClient()
        second = self.analyze(again)
        self.assertEqual(again.calls, [])
        self.assertEqual(first["segments"], second["segments"])
        self.assertEqual(second["provenance"]["narrator_read"]["metrics"]["total"]["provider_calls"], 0)

    def test_missing_markup_label_is_repaired_by_one_validation_retry(self):
        client = NarratorClient(drop_first_markup_label=True)
        document = self.analyze(client, workers=1, repair_attempts=1)
        markup_calls = [call for call in client.calls if call["mode"] == "narrator_markup"]
        self.assertTrue(any("correction" in call for call in markup_calls))
        self.assertIn("missing labels", next(call for call in markup_calls if "correction" in call)
                      ["correction"]["validator_error"])
        self.assertEqual(readiness(document)["errors"], [])


class VerificationTieBreakTests(unittest.TestCase):
    class Client:
        identity = {"model": "fixture", "base_url": "http://offline.invalid/v1"}

        def __init__(self, verdict, choice):
            self.verdict, self.choice = verdict, choice
            self.calls = []

        def complete(self, system, payload):
            self.calls.append(deepcopy(payload))
            if payload["mode"] == "narrator_verify":
                return {"r": [[item["id"], self.verdict] for item in payload["items"]]}
            if payload["mode"] == "narrator_verify_tiebreak":
                return {"r": [[item["id"], self.choice] for item in payload["items"]]}
            raise AssertionError("unexpected request in fixture")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "cache"
        self.source = make_source([[('"Fine," Mara said.', "p")]])
        self.spans = quote_spans(self.source)
        self.ledger = apply_delta(new_ledger(), {"new": [["Mara", "", "f", "", 1], ["Jon", "", "m", "", 1]]},
                                  {"b000001": self.source["blocks"][0]["text"]}, {1: "b000001"})
        self.labels = {self.spans[0]["id"]: {"who": ("char", "1")}}
        self.items = [{"span": self.spans[0], "reason": "tag_disagreement", "options": ["1", "2"]}]

    def run_verification(self, client):
        from epub_tts_narrator.narrator_read import Meter, _verify_items
        from epub_tts_narrator.pipeline import CachedRequests
        return _verify_items(self.source, CachedRequests(client, self.cache, repair_attempts=0), Meter(client),
                             self.ledger, self.spans, self.labels, self.items)

    def test_agreement_keeps_the_original_label_without_a_tie_break_call(self):
        client = self.Client(verdict="1", choice="b")
        labels, stats = self.run_verification(client)
        self.assertEqual(labels[self.spans[0]["id"]]["who"], ("char", "1"))
        self.assertEqual([call["mode"] for call in client.calls], ["narrator_verify"])
        self.assertEqual(stats["tie_breaks"], 0)

    def test_tie_break_can_select_either_supplied_candidate(self):
        for choice, expected, changed in (("a", "1", 0), ("b", "2", 1)):
            with self.subTest(choice=choice):
                self.cache = self.cache.parent / f"cache-choice-{choice}"
                client = self.Client(verdict="2", choice=choice)
                labels, stats = self.run_verification(client)
                self.assertEqual(labels[self.spans[0]["id"]]["who"], ("char", expected))
                self.assertEqual((stats["tie_breaks"], stats["changed"]), (1, changed))
                tie_item = next(call["items"][0] for call in client.calls if call["mode"] == "narrator_verify_tiebreak")
                self.assertEqual(tie_item["target"], self.spans[0]["id"])
                self.assertIn('"Fine," Mara said.', tie_item["text"])
                self.assertEqual(tie_item["choices"], [{"id": "a", "who": "1"}, {"id": "b", "who": "2"}])
                self.assertNotIn("options", tie_item)

    def test_invalid_choice_is_rejected_then_resume_and_replay_use_validated_caches(self):
        original = deepcopy(self.labels)
        with self.assertRaises(ValueError):
            self.run_verification(self.Client(verdict="2", choice="2"))
        self.assertEqual(self.labels, original)

        resumed = self.Client(verdict="2", choice="a")
        labels, stats = self.run_verification(resumed)
        self.assertEqual(labels[self.spans[0]["id"]]["who"], ("char", "1"))
        self.assertEqual([call["mode"] for call in resumed.calls], ["narrator_verify_tiebreak"])

        replayed = self.Client(verdict="2", choice="b")
        cached_labels, cached_stats = self.run_verification(replayed)
        self.assertEqual(replayed.calls, [])
        self.assertEqual(cached_labels, labels)
        self.assertEqual(cached_stats, stats)


class SeriesCheckedMatchTests(unittest.TestCase):
    def test_quote_backed_series_check_settles_a_name_shared_by_two_prior_characters(self):
        from epub_tts_narrator.series import reconcile_series_cast
        from test_series_identity_scan import fixture
        source, cast, registry = fixture()
        prior = registry["books"][0]["cast"]
        prior["characters"].append(dict(deepcopy(prior["characters"][0]), id="prior_other", voice_id="prior_voice_other",
                                        description="A different Mara, a baker."))
        prior["voices"].append({"id": "prior_voice_other", "description": "Separate baker voice.", "traits": []})
        mara = cast["characters"][0]
        mara["series_identity"] = {"character_id": "prior_mara", "basis": "context", "review_required": False,
                                   "reason": "Recognized during the narrator read.", "evidence": deepcopy(mara["evidence"])}
        for checked, voice in ((None, None), ({"c1": "prior_mara"}, "prior_voice_mara"), ({"c1": "prior_other"}, None)):
            with self.subTest(checked=checked):
                linked, issues, _ = reconcile_series_cast(source, deepcopy(cast), registry, checked_matches=checked)
                if voice:
                    self.assertEqual(linked["characters"][0]["voice_id"], voice)
                else:
                    # A shared name alone, or a check that disagrees with the proposal, never reuses a voice.
                    self.assertNotIn(linked["characters"][0]["voice_id"], {"prior_voice_mara", "prior_voice_other"})
                    self.assertTrue(any(issue["kind"] == "series_identity" for issue in issues))


class SeriesCheckEvidenceTests(unittest.TestCase):
    """The series check compares this book's best-matching paragraphs with the earlier description."""

    PRIOR = {"id": "series_kenneth", "name": "Kenneth", "aliases": [],
             "description": "Leader of the Foxes Guild; black rimmed glasses"}

    class Client:
        def __init__(self, answer):
            self.answer, self.items = answer, []
            self.identity = {"model": "fixture", "base_url": "http://offline.invalid/v1"}

        def complete(self, system, payload):
            self.items.append(deepcopy(payload["items"]))
            return {"r": [self.answer(item) for item in payload["items"]]}

    def check(self, answer):
        from epub_tts_narrator.narrator_read import Meter, check_series
        from epub_tts_narrator.pipeline import CachedRequests
        source = make_source([[("Kenneth waved.", "p"), ("Kenneth sat down.", "p"), ("Kenneth sipped coffee.", "p"),
                               ("Kenneth took off his black rimmed glasses, tired after the Foxes Guild meeting.", "p"),
                               ("Kenneth yawned.", "p")]])
        ledger = new_ledger([self.PRIOR])
        ledger["characters"][0].update(seen=True, note="Waves and yawns a lot")  # rewritten by the prep read
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        client = self.Client(answer)
        requests = CachedRequests(client, Path(directory.name), repair_attempts=0)
        ledger, outcomes = check_series(source, requests, Meter(client), ledger)
        return ledger, outcomes, client

    def test_the_earlier_description_and_its_best_matching_paragraph_are_shown(self):
        def answer(item):
            if "Foxes Guild" in item["prior"] and "4|" in item["text"]:
                return [item["id"], "y", 4, "tired after the Foxes Guild meeting"]
            return [item["id"], "?"]

        ledger, outcomes, client = self.check(answer)
        self.assertEqual(outcomes[0]["verdict"], "y")
        self.assertTrue(ledger["characters"][0]["series_confirmed"])
        self.assertEqual(len(client.items), 1)
        self.assertNotIn("Waves", client.items[0][0]["prior"])

    def test_an_elided_or_annotated_quote_keeps_only_its_longest_exact_piece(self):
        def answer(item):
            # Wrong paragraph number, an elision, and the model's own commentary.
            return [item["id"], "y", 1, "Kenneth took off his black rimmed glasses... the Foxes Guild meeting"
                                        " \u2014 he leads the guild, matching the description"]

        ledger, outcomes, _ = self.check(answer)
        self.assertEqual(outcomes[0]["verdict"], "y")
        self.assertEqual(ledger["characters"][0]["series_evidence"],
                         [{"block_id": "b000004", "quote": "Kenneth took off his black rimmed glasses"}])
        _, outcomes, _ = self.check(lambda item: [item["id"], "y", 4, "Kenneth leads... the Guild"])
        self.assertEqual(outcomes[0]["verdict"], "?")

    def test_exact_quote_never_returns_text_absent_from_the_paragraph(self):
        from epub_tts_narrator.narrator_ledger import exact_quote
        text = 'Liz nodded sagely. "The Delvers Guild has broken a lot of trust." She smiled.'
        self.assertEqual(exact_quote("Liz nodded sagely.", text), "Liz nodded sagely.")
        self.assertEqual(exact_quote('"The Delvers Guild has broken a lot of trust." Liz nodded', text),
                         "The Delvers Guild has broken a lot of trust.")
        self.assertIsNone(exact_quote("Liz agreed with the Guild", text))
        self.assertIsNone(exact_quote("Liz... smiled", text))

    def test_an_unsure_verdict_gets_one_wider_look(self):
        def answer(item):
            if item["text"].count("\n") >= 2:
                return [item["id"], "y", 4, "tired after the Foxes Guild meeting"]
            return [item["id"], "?"]

        from epub_tts_narrator import narrator_read
        with patch.object(narrator_read, "_COMMON", narrator_read._COMMON | {"foxes", "guild", "black", "rimmed", "glasses"}):
            ledger, outcomes, client = self.check(answer)
        self.assertEqual([len(batch) for batch in client.items], [1, 1])
        self.assertEqual(outcomes[0]["verdict"], "y")

        _, outcomes, client = self.check(lambda item: [item["id"], "?"])
        self.assertEqual(len(client.items), 2)
        self.assertEqual(outcomes[0]["verdict"], "?")


class NarratorProviderUsageTests(unittest.TestCase):
    def setUp(self):
        self.opener = unittest.mock.Mock()
        patcher = patch("epub_tts_narrator.provider.build_opener", return_value=self.opener)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_usage_is_reported_as_numbers_and_requested_only_when_enabled(self):
        usage = {"prompt_tokens": 120, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 64},
                 "completion_tokens_details": {"reasoning_tokens": 12}}
        self.opener.open.return_value = FakeResponse(sse(delta('{"r": []}'), delta_end("stop"),
                                                         {"choices": [], "usage": usage}, "[DONE]"))
        records = []
        client = OpenAICompatibleClient("model", include_usage=True, usage_observer=records.append)
        client.complete("System", {"text": "Story"})
        body = json.loads(self.opener.open.call_args.args[0].data)
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual((records[0]["prompt_tokens"], records[0]["completion_tokens"],
                          records[0]["cached_tokens"], records[0]["reasoning_tokens"]), (120, 30, 64, 12))
        self.assertNotIn("Story", json.dumps(records))
        self.assertEqual(client.identity, OpenAICompatibleClient("model").identity)

        self.opener.open.return_value = FakeResponse(sse(delta('{"r": []}'), delta_end("stop"), "[DONE]"))
        OpenAICompatibleClient("model").complete("System", {"text": "Story"})
        self.assertNotIn("stream_options", json.loads(self.opener.open.call_args.args[0].data))


class NarratorCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "epub-tts-narrator.toml").write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nrepair_attempts = 0\nattribution_workers = 2\n', encoding="utf-8")
        self.clients = []

    def client(self, *args, **kwargs):
        client = NarratorClient(*args, **kwargs)
        self.clients.append(client)
        return client

    def run_cli(self, *args):
        previous = Path.cwd()
        stdout, stderr = io.StringIO(), io.StringIO()
        os.chdir(self.root)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                    patch("sys.stdin", io.StringIO()):
                status = main([str(arg) for arg in args])
        finally:
            os.chdir(previous)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_narrator_is_the_default_strategy_for_a_new_book(self):
        epub = make_serial_epub(self.root / "book.epub", "Fixture", "fixture-book", [
            ("One", ['"You never listen," Jon said.', '"I heard you," Mara said.'])])
        output = self.root / "book"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, stdout, stderr = self.run_cli("analyze", epub, "-o", output)
        self.assertEqual(status, 0, stderr)
        self.assertTrue((output / "manifest.json").is_file())
        analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["provenance"]["strategy"], "narrator_read")
        # Series reconciliation renames IDs; each voice is still designed from its own character's notes.
        voices = {voice["id"]: voice for voice in analysis["cast"]["voices"]}
        for person in analysis["cast"]["characters"]:
            self.assertRegex(voices[person["voice_id"]]["description"], r"^A clear (male|female) voice number \d+\.$")
        modes = {call.get("mode") for client in self.clients for call in client.calls}
        self.assertTrue(modes <= {"narrator_prep", "narrator_markup", "narrator_verify", "narrator_verify_tiebreak", "narrator_voices",
                                  "narrator_merge_check", "narrator_series_check", "narrator_accents", "audition_script"}, modes)

    def test_a_narrator_read_book_updates_its_new_chapters_with_the_narrator_read(self):
        with (self.root / "epub-tts-narrator.toml").open("a", encoding="utf-8") as handle:
            handle.write('cast_review_notes = "Jon keeps the lighthouse."\n')
        chapter_one = ("One", ['"You never listen," Jon said.', '"I heard you," Mara said.'])
        first = make_serial_epub(self.root / "v1.epub", "Serial", "serial-1", [chapter_one])
        second = make_serial_epub(self.root / "v2.epub", "Serial", "serial-2", [chapter_one, (
            "Two", ['Jon returned to the lighthouse he once kept. "Back again," Jon said.',
                    'From the stairs, Elsie said, "No, you don\'t."'])])
        book = self.root / "book"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, _, stderr = self.run_cli("analyze", first, "-o", book, "--strategy", "narrator")
            self.assertEqual(status, 0, stderr)
            original = json.loads((book / "analysis.json").read_text(encoding="utf-8"))
            calls_before = sum(len(client.calls) for client in self.clients)
            # No --strategy: the update follows the strategy the book was analyzed with.
            status, stdout, stderr = self.run_cli("analyze", second, "-o", book)
        self.assertEqual(status, 0, stderr + stdout)
        sent = [call for client in self.clients for call in client.calls][calls_before:]
        modes = {call.get("mode") for call in sent}
        self.assertIn("narrator_markup", modes)
        self.assertTrue(modes <= {"narrator_prep", "narrator_markup", "narrator_verify", "narrator_verify_tiebreak", "narrator_voices",
                                  "narrator_merge_check", "narrator_series_check", "narrator_accents", "audition_script"}, modes)
        # Only the new chapter is analyzed, with the configured notes.
        self.assertFalse(any("I heard you" in json.dumps(call) for call in sent
                             if call.get("mode", "").startswith("narrator_")))
        self.assertEqual({call.get("hints") for call in sent if call.get("mode") == "narrator_prep"},
                         {"Jon keeps the lighthouse."})
        document = json.loads((book / "analysis.json").read_text(encoding="utf-8"))
        update = document["provenance"]["book_updates"][-1]
        self.assertEqual(update["strategy"], "narrator")
        self.assertEqual(document["provenance"]["strategy"], "narrator_read")
        self.assertEqual(document["cast"]["characters"][:len(original["cast"]["characters"])],
                         original["cast"]["characters"])
        names = {character["id"]: character["name"] for character in document["cast"]["characters"]}
        self.assertEqual(sorted(names.values()), ["Elsie", "Jon", "Mara"])
        jon = next(c for c in original["cast"]["characters"] if c["name"] == "Jon")
        roles = {role["id"]: role.get("character_id") for role in document["cast"]["roles"]}
        speakers = {row["text"]: names.get(roles.get(row["role_id"])) for row in document["segments"]}
        self.assertEqual(speakers['"Back again,"'], "Jon")
        self.assertIn(jon["id"], update["existing_characters"])
        self.assertEqual(speakers['"No, you don\'t."'], "Elsie")

    def continue_series(self, series_answer):
        first = make_serial_epub(self.root / "one.epub", "One", "book-one", [
            ("One", ['"You never listen," Jon said.', '"I heard you," Mara said.'])])
        second = make_serial_epub(self.root / "two.epub", "Two", "book-two", [
            ("One", ['Jon returned to the lighthouse he once kept. "Back again," Jon said.'])])
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, _, stderr = self.run_cli("analyze", first, "-o", self.root / "one", "--strategy", "narrator")
            self.assertEqual(status, 0, stderr)
            factory = lambda *args, **kwargs: self.client(*args, series_answer=series_answer, **kwargs)
            with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
                status, _, stderr = self.run_cli("analyze", second, "-o", self.root / "two", "--strategy", "narrator",
                                                 "--continue-from", self.root / "one")
        self.assertEqual(status, 0, stderr)
        one = json.loads((self.root / "one" / "analysis.json").read_text(encoding="utf-8"))
        two = json.loads((self.root / "two" / "analysis.json").read_text(encoding="utf-8"))
        jon_one = next(c for c in one["cast"]["characters"] if c["name"] == "Jon")
        jon_two = next(c for c in two["cast"]["characters"] if c["name"] == "Jon")
        return jon_one, jon_two, two

    def test_the_voice_a_character_grows_into_is_the_one_later_books_use(self):
        first = make_serial_epub(self.root / "one.epub", "One", "book-one", [
            ("One", ['"Hi," Jon said.', "Years later Jon's voice had deepened into a man's.", '"Hello again," Jon said.'])])
        second = make_serial_epub(self.root / "two.epub", "Two", "book-two", [
            ("One", ['Jon returned to the lighthouse he once kept. "Back again," Jon said.',
                     "Old age came; Jon's voice had deepened into a man's rumble.", '"Still here," Jon said.'])])
        third = make_serial_epub(self.root / "three.epub", "Three", "book-three", [
            ("One", ['Jon returned to the lighthouse he once kept. "Once more," Jon said.'])])

        def factory(*args, **kwargs):
            client = self.client(*args, **kwargs)
            complete = client.complete

            def changing(system, payload):
                result = complete(system, payload)
                if payload.get("mode") == "narrator_prep":
                    for line in payload["text"].splitlines():
                        number, _, text = line.partition("|")
                        if "deepened" in text:
                            result["change"] = [["Jon", int(number), "voice had deepened into a man's", "age",
                                                 "a grown man now, deeper"]]
                return result

            client.complete = changing
            return client

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli("analyze", first, "-o", self.root / "one", "--strategy", "narrator")
            self.assertEqual(status, 0, stderr)
            status, _, stderr = self.run_cli("analyze", second, "-o", self.root / "two", "--strategy", "narrator",
                                             "--continue-from", self.root / "one")
            self.assertEqual(status, 0, stderr)
            status, _, stderr = self.run_cli("analyze", third, "-o", self.root / "three", "--strategy", "narrator",
                                             "--continue-from", self.root / "two")
        self.assertEqual(status, 0, stderr)
        books = [json.loads((self.root / name / "analysis.json").read_text(encoding="utf-8"))
                 for name in ("one", "two", "three")]

        def voice_of(document, text):
            return next(row for row in document["segments"] if row["text"] == text)["voice_id"]

        grown = next(person for person in books[0]["cast"]["characters"] if person["name"] == "Jon")["voice_id"]
        self.assertEqual(voice_of(books[0], '"Hello again,"'), grown)
        # Book two inherits the grown voice as Jon's main voice; its own later
        # change is derived from it and is still in effect when the book ends.
        self.assertEqual(voice_of(books[1], '"Back again,"'), grown)
        older = voice_of(books[1], '"Still here,"')
        later = next(voice for voice in books[1]["cast"]["voices"] if voice["id"] == older)
        self.assertEqual(later["derived_from"]["voice_id"], grown)
        self.assertTrue(later["derived_from"]["through_end"])
        # Book three starts where book two ended.
        self.assertEqual(voice_of(books[2], '"Once more,"'), older)
        carried = next(voice for voice in books[2]["cast"]["voices"] if voice["id"] == older)
        self.assertEqual(carried["derived_from"]["evidence"], [])
        self.assertIn("series_origin", carried)

    def test_confirmed_returning_character_keeps_the_series_identity_and_voice(self):
        jon_one, jon_two, _ = self.continue_series("y")
        self.assertEqual(jon_two["id"], jon_one["id"])
        self.assertEqual(jon_two["voice_id"], jon_one["voice_id"])

    def test_name_match_alone_never_reuses_a_series_identity(self):
        jon_one, jon_two, two = self.continue_series("?")
        self.assertNotEqual(jon_two["id"], jon_one["id"])
        self.assertTrue(any(issue["kind"] == "series_identity" and issue["target"] == jon_two["id"]
                            for issue in two["issues"]))


if __name__ == "__main__":
    unittest.main()
