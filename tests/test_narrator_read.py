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
        if mode == "narrator_voices":
            return {"v": [[row[0], f"A clear voice for {row[1]}.", "General American; even pace",
                           "pitch=mid;pace=even;accent=General American"] for row in payload["voices"]]}
        raise ProviderError("unexpected request in fixture")


class QuoteSpanTests(unittest.TestCase):
    def test_split_quotation_and_trailing_tag_are_found_without_model_calls(self):
        source = make_source([[("\"I thought,\" Mara said, \"that you had gone.\"", "p")]])
        spans = quote_spans(source)
        self.assertEqual([span["text"] for span in spans], ['"I thought,"', '"that you had gone."'])
        self.assertEqual(spans[0]["tag"]["name"], "Mara")
        self.assertEqual(find_tag('"Go," he said.', 0, 5)["pronoun"], "he")

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

    def test_delivery_needs_a_supporting_quote_from_the_paragraph(self):
        self.source = make_source([[('"Stay," Mara whispered hoarsely.', "p"), ('"Go," Jon said.', "p")]])

        def rewrite(qids, result):
            result["q"] = [[qids["Stay,"], "1", "", "whispered", "Mara whispered hoarsely"],
                           [qids["Go,"], "2", "", "shouted", "invented support"]]

        document = self.markup_with(rewrite)
        rows = self.by_text(document)
        self.assertEqual(rows['"Stay,"']["delivery"],
                         {"direction": "whispered", "evidence": [{"block_id": "b000001", "quote": "Mara whispered hoarsely"}]})
        self.assertEqual(rows['"Go,"']["delivery"]["direction"], "")
        stats = document["provenance"]["narrator_read"]["segments"]
        self.assertEqual((stats["delivery"], stats["delivery_dropped"]), (1, 1))

    def test_spelling_edits_and_cues_go_through_the_remove_only_review(self):
        self.source = make_source([[('"Nooooo, wait," Mara whispered.', "p"), ("The door went bang.", "p")]])

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

    def test_configured_notes_combine_as_in_the_consensus_pipeline(self):
        from epub_tts_narrator.narrator_read import settings_notes
        self.assertEqual(settings_notes({"cast_review_notes": "A", "cast_reconciliation_notes": ["B", ""],
                                         "attribution_review_notes": ""}),
                         {"cast_notes": "A\nB", "attribution_notes": "A"})
        self.assertEqual(settings_notes({"cast_review_notes": "", "cast_reconciliation_notes": [],
                                         "attribution_review_notes": "C"}),
                         {"cast_notes": "", "attribution_notes": "C"})

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

    def test_narrator_strategy_publishes_a_book_directory_without_consensus_requests(self):
        epub = make_serial_epub(self.root / "book.epub", "Fixture", "fixture-book", [
            ("One", ['"You never listen," Jon said.', '"I heard you," Mara said.'])])
        output = self.root / "book"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, stdout, stderr = self.run_cli("analyze", epub, "-o", output, "--strategy", "narrator")
        self.assertEqual(status, 0, stderr)
        self.assertTrue((output / "manifest.json").is_file())
        analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["provenance"]["strategy"], "narrator_read")
        # Series reconciliation renames IDs; each voice is still designed from its own character's notes.
        voices = {voice["id"]: voice for voice in analysis["cast"]["voices"]}
        for person in analysis["cast"]["characters"]:
            self.assertEqual(voices[person["voice_id"]]["description"], f"A clear voice for {person['name']}.")
        modes = {call.get("mode") for client in self.clients for call in client.calls}
        self.assertTrue(modes <= {"narrator_prep", "narrator_markup", "narrator_verify", "narrator_voices",
                                  "narrator_merge_check", "narrator_series_check", "audition_script"}, modes)

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
        self.assertTrue(modes <= {"narrator_prep", "narrator_markup", "narrator_verify", "narrator_voices",
                                  "narrator_merge_check", "narrator_series_check", "audition_script"}, modes)
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
