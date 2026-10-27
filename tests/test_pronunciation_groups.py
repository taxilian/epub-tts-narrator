"""Capitalization families preserve source evidence and explicit decisions."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.pronunciation_groups import case_key, entry_words, family_words
from epub_tts_narrator.pronunciations import (find_pronunciation, load_pronunciations, reset_pronunciation,
    save_candidates, save_proposal, set_pronunciation, validate_entry, word_id)
from epub_tts_narrator.series import link_analysis
from shared_fixtures import snapshot
from test_pronunciations import candidate
from test_series import fixture, propose


QUIET = {"alphabet": "cmu_arpabet", "value": "K W AY1 AH0 T"}
OTHER = {"alphabet": "cmu_arpabet", "value": "K W IY1 T"}


class PronunciationGroupsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def minimal(self, text):
        directory = self.root / "minimal"
        directory.mkdir()
        document = {"source": {"book": {"id": "case-test"}, "blocks": [{"id": "b1", "text": text}]}, "cast": {}}
        write_json(directory / "source.json", document["source"])
        return directory, document

    def book(self, name, text, previous=None):
        directory = self.root / name
        document = fixture(sequel=previous is not None)
        document["source"]["book"].update(id=name, title=name)
        document["source"]["blocks"][1]["text"] = text
        segment = next(row for row in document["segments"] if row["block_id"] == "b2")
        segment.update(text=text, end=len(text))
        if previous is None:
            registry = fresh_registry(directory)
        else:
            registry = continuation_registry(previous, document["source"])
            prior = json.loads((previous / "analysis.json").read_text())
            propose(document, prior["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, directory, export_bundle, continue_from=previous)
        return directory, document

    def test_pending_case_guesses_group_without_rewriting_registry_or_source(self):
        directory, document = self.minimal("Quiet quiet QUIET Pit pit Marah")
        entries = [candidate(word) for word in ("Quiet", "quiet", "QUIET", "Pit", "pit", "Marah")]
        entries[0]["proposal"], entries[1]["proposal"] = QUIET, OTHER
        save_candidates(directory, document, entries)
        before = snapshot(directory)
        rows = load_pronunciations(directory)
        self.assertEqual(len(rows), 3)
        quiet = find_pronunciation(rows, "quiet")
        self.assertEqual(quiet["word"], "Quiet")
        self.assertEqual(set(entry_words(quiet)), {"Quiet", "quiet", "QUIET"})
        self.assertEqual(quiet["proposal"], QUIET)
        self.assertEqual(find_pronunciation(rows, "QUIET"), quiet)
        self.assertEqual(len(load_pronunciations(directory, group_cases=False)), 6)
        self.assertEqual(snapshot(directory), before)
        self.assertNotEqual(word_id("Quiet"), word_id("quiet"))

    def test_capitonyms_initialisms_and_different_spellings_stay_separate(self):
        words = "Polish polish Job job Reading reading August august Nice nice Mobile mobile US us ID id Mara Marah O'Neill O’Neill Straße Strasse".split()
        directory, document = self.minimal(" ".join(words))
        save_candidates(directory, document, [dict(candidate(word), kind="other") for word in words])
        rows = load_pronunciations(directory)
        self.assertEqual(len(rows), len(words))
        self.assertNotEqual(case_key("Straße"), case_key("Strasse"))
        set_pronunciation(directory, document, "US", QUIET)
        self.assertEqual(find_pronunciation(load_pronunciations(directory), "us")["status"], "pending")

    def test_natural_case_name_metadata_groups_shouted_names_including_short_names(self):
        directory, document = self.minimal("Stell STELL Avalon AVALON Jas JAS US us")
        save_candidates(directory, document, [candidate(word) for word in ("Stell", "Avalon", "Jas", "US", "us")])
        rows = load_pronunciations(directory)
        self.assertEqual(set(entry_words(find_pronunciation(rows, "JAS"))), {"Jas", "JAS"})
        self.assertEqual(set(entry_words(find_pronunciation(rows, "STELL"))), {"Stell", "STELL"})
        self.assertEqual(set(entry_words(find_pronunciation(rows, "AVALON"))), {"Avalon", "AVALON"})
        self.assertNotEqual(find_pronunciation(rows, "US"), find_pronunciation(rows, "us"))

    def test_uppercase_only_candidate_uses_observed_natural_spelling_for_review(self):
        directory, document = self.minimal("AVALON Avalon")
        save_candidates(directory, document, [candidate("AVALON")])
        row = load_pronunciations(directory)[0]
        self.assertEqual(row["word"], "Avalon")
        self.assertEqual(row["origin_word"], "AVALON")
        save_proposal(directory, document, "AVALON", QUIET)
        self.assertEqual(load_pronunciations(directory)[0]["proposal"], QUIET)

    def test_decision_and_reset_cover_all_observed_variants_in_one_local_write(self):
        directory, document = self.minimal("Quiet quiet QUIET quietly")
        save_candidates(directory, document, [candidate("Quiet"), candidate("quietly")])
        with patch("epub_tts_narrator.pronunciations.write_json", wraps=write_json) as writes:
            set_pronunciation(directory, document, "quiet", QUIET)
            self.assertEqual(writes.call_count, 1)
        record = json.loads((directory / "pronunciations.json").read_text())
        self.assertEqual({entry["word"] for entry in record["entries"]}, {"Quiet", "quiet", "QUIET", "quietly"})
        for entry in record["entries"]:
            self.assertEqual(validate_entry(entry, document["source"]), entry)
        self.assertEqual(find_pronunciation(load_pronunciations(directory), "QUIET")["pronunciation"], QUIET)
        set_pronunciation(directory, document, "QUIET", status="ignored")
        self.assertEqual(find_pronunciation(load_pronunciations(directory), "quiet")["status"], "ignored")
        reset_pronunciation(directory, "quiet")
        self.assertEqual([entry["word"] for entry in load_pronunciations(directory)], ["quietly"])

    def test_failed_family_write_does_not_partially_update_decisions(self):
        directory, document = self.minimal("Quiet quiet")
        save_candidates(directory, document, [candidate("Quiet"), candidate("quiet")])
        before = (directory / "pronunciations.json").read_bytes()
        with patch("epub_tts_narrator.pronunciations.write_json", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                set_pronunciation(directory, document, "quiet", QUIET)
        self.assertEqual((directory / "pronunciations.json").read_bytes(), before)

    def test_legacy_conflicting_explicit_choices_remain_visible_until_user_resolves_family(self):
        directory, document = self.minimal("Quiet quiet QUIET")
        save_candidates(directory, document, [candidate(word) for word in ("Quiet", "quiet", "QUIET")])
        path = directory / "pronunciations.json"
        record = json.loads(path.read_text())
        for entry in record["entries"]:
            if entry["word"] != "QUIET":
                entry.update(status="confirmed", pronunciation=QUIET if entry["word"] == "Quiet" else OTHER)
        write_json(path, record, overwrite=True)
        rows = load_pronunciations(directory)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["case_conflict"] for row in rows))
        self.assertEqual(find_pronunciation(rows, "Quiet")["pronunciation"], QUIET)
        self.assertEqual(find_pronunciation(rows, "quiet")["pronunciation"], OTHER)
        self.assertEqual(find_pronunciation(rows, "QUIET")["status"], "pending")
        self.assertEqual(set(family_words("Quiet", rows, {word for row in rows for word in entry_words(row)})), {"Quiet", "quiet", "QUIET"})
        save_proposal(directory, document, "QUIET", QUIET)
        self.assertEqual(find_pronunciation(load_pronunciations(directory), "QUIET")["proposal"], QUIET)
        before = path.read_bytes()
        save_proposal(directory, document, "QUIET", OTHER)
        self.assertEqual(path.read_bytes(), before)
        set_pronunciation(directory, document, "quiet", QUIET)
        rows = load_pronunciations(directory)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("case_conflict", rows[0])

    def test_legacy_confirmed_and_ignored_choices_are_not_silently_merged(self):
        directory, document = self.minimal("Quiet quiet")
        save_candidates(directory, document, [candidate("Quiet"), candidate("quiet")])
        path = directory / "pronunciations.json"
        record = json.loads(path.read_text())
        record["entries"][0].update(status="confirmed", pronunciation=QUIET)
        record["entries"][1].update(status="ignored")
        write_json(path, record, overwrite=True)
        self.assertEqual({row["status"] for row in load_pronunciations(directory)}, {"confirmed", "ignored"})

    def test_only_case_change_in_descendant_inherits_live_choice_without_source_or_ancestor_mutation(self):
        first, first_doc = self.book("first", "Quiet returned.")
        second, second_doc = self.book("second", "quiet returned.", first)
        third, third_doc = self.book("third", "QUIET returned.", second)
        save_candidates(second, second_doc, [candidate("quiet", "b2")])
        save_candidates(third, third_doc, [candidate("QUIET", "b2")])
        set_pronunciation(first, first_doc, "Quiet", QUIET)
        before = snapshot(first)
        inherited = load_pronunciations(third)[0]
        self.assertEqual(inherited["word"], "QUIET")
        self.assertEqual(inherited["origin_word"], "Quiet")
        self.assertEqual(inherited["origin_directory"], str(first))
        self.assertEqual(inherited["pronunciation"], QUIET)
        self.assertEqual(inherited["evidence"][0]["quote"], "Quiet")
        self.assertTrue(inherited["inherited"])
        self.assertIsNone(save_proposal(third, third_doc, "not-present", OTHER))
        self.assertEqual(save_proposal(third, third_doc, "QUIET", OTHER)["pronunciation"], QUIET)
        set_pronunciation(second, second_doc, "quiet", status="ignored")
        self.assertEqual(load_pronunciations(third)[0]["status"], "ignored")
        reset_pronunciation(second, "quiet")
        self.assertEqual(load_pronunciations(third)[0]["pronunciation"], QUIET)
        self.assertEqual(snapshot(first), before)

    def test_inherited_pending_case_proposal_is_saved_with_current_exact_evidence(self):
        first, first_doc = self.book("first", "Quiet returned.")
        second, second_doc = self.book("second", "quiet returned.", first)
        save_candidates(first, first_doc, [candidate("Quiet", "b2")])
        before = snapshot(first)
        saved = save_proposal(second, second_doc, "quiet", QUIET)
        self.assertFalse(saved["inherited"])
        self.assertEqual(saved["proposal"], QUIET)
        local = json.loads((second / "pronunciations.json").read_text())["entries"][0]
        self.assertEqual(local["word"], "quiet")
        self.assertEqual(validate_entry(local, second_doc["source"]), local)
        self.assertEqual(snapshot(first), before)

    def test_case_guesses_need_only_one_preparation_request_and_checkpoint(self):
        from epub_tts_narrator.pronunciation_proposals import propose_name_pronunciations
        directory, document = self.minimal("Quiet quiet QUIET Pit pit")
        save_candidates(directory, document, [candidate(word) for word in ("Quiet", "quiet", "QUIET", "Pit", "pit")])
        calls = []
        with patch("epub_tts_narrator.pronunciation_proposals.propose_pronunciation", side_effect=lambda doc, entry, requests: calls.append(entry["word"]) or QUIET):
            for _ in range(2):
                propose_name_pronunciations(document, load_pronunciations(directory), object(),
                    on_proposal=lambda entry, value: save_proposal(directory, document, entry["word"], value))
        self.assertEqual(calls, ["Pit", "Quiet"])
        self.assertEqual(len(load_pronunciations(directory)), 2)


if __name__ == "__main__":
    unittest.main()
