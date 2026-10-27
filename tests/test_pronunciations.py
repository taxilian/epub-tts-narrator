from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import fingerprint, write_json
from epub_tts_narrator.pronunciations import DISCOVERY_VERSION, discovery_pending, load_pronunciations, preparation_pending, pronunciation_counts, reset_pronunciation, save_candidates, save_proposal, set_pronunciation, validate_entry, validate_pronunciation, validate_word, word_id
from epub_tts_narrator.series import link_analysis
from shared_fixtures import snapshot
from test_series import fixture, propose


def candidate(word="Mara", block_id="b1"):
    return {"id": word_id(word), "word": word, "status": "pending", "kind": "name",
            "reason": "The name may have more than one reading.", "evidence": [{"block_id": block_id, "quote": word}]}


CMU_MARA = {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"}


class PronunciationsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def book(self, name="first", previous=None, include_softly=False):
        directory = self.root / name
        document = fixture(sequel=previous is not None)
        if include_softly:
            document["source"]["blocks"][0]["text"] = "Mara Vale returned softly."
            document["segments"][0].update(text="Mara Vale returned softly.", end=len("Mara Vale returned softly."))
        if previous is None:
            registry = fresh_registry(directory)
        else:
            document["source"]["book"].update(id=name, title=name)
            registry = continuation_registry(previous, document["source"])
            prior = json.loads((previous / "analysis.json").read_text())
            propose(document, prior["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, directory, export_bundle, continue_from=previous)
        return directory, document

    def test_absent_registry_is_read_only_even_for_minimal_existing_bundles(self):
        self.assertEqual(load_pronunciations(self.root / "missing"), [])
        directory, _ = self.book()
        before = snapshot(directory)
        self.assertEqual(load_pronunciations(directory), [])
        self.assertEqual(snapshot(directory), before)

    def test_single_word_contract_and_identity_preserve_exact_case_and_typography(self):
        for word in ("Mara", "Mara-Vale", "O'Neill", "O’Neill", "Æsir"):
            self.assertEqual(validate_word(word), word)
        for word in ("", "Mara Vale", "<Mara>", "[Mara]", "123", "Mara\n", "Mara_One", "-Mara", "Mara-"):
            with self.subTest(word=word), self.assertRaises(ValueError):
                validate_word(word)
        self.assertEqual(len({word_id(word) for word in ("Mara", "mara", "Marah", "O'Neill", "O’Neill")}), 5)

    def test_confirmed_pronunciation_validates_respelling_or_cmu_without_markup(self):
        for pronunciation in ({"alphabet": "respelling", "value": "MAH-ruh"}, {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"}):
            self.assertEqual(validate_pronunciation(pronunciation), pronunciation)
        for pronunciation in ({"alphabet": "ipa", "value": "mara"}, {"alphabet": "respelling", "value": "say mara"}, {"alphabet": "cmu_arpabet", "value": "NOPE"}, {"alphabet": "respelling", "value": "<pause/>"}):
            with self.subTest(pronunciation=pronunciation), self.assertRaises(ValueError):
                validate_pronunciation(pronunciation)

    def test_discovery_marker_handles_zero_candidates_and_manual_choices_separately(self):
        directory, document = self.book()
        self.assertTrue(discovery_pending(directory, document))
        set_pronunciation(directory, document, "Mara", {"alphabet": "respelling", "value": "MAH-ruh"})
        self.assertTrue(discovery_pending(directory, document))
        save_candidates(directory, document, [])
        self.assertFalse(discovery_pending(directory, document))
        changed = deepcopy(document)
        changed["cast"]["characters"][0]["description"] = "New context."
        self.assertTrue(discovery_pending(directory, changed))

    def test_preparation_resumes_missing_name_proposals_after_discovery_completes(self):
        directory, document = self.book()
        self.assertTrue(preparation_pending(directory, document))
        save_candidates(directory, document, [candidate(), dict(candidate("softly"), kind="ambiguous")])
        self.assertFalse(discovery_pending(directory, document))
        self.assertTrue(preparation_pending(directory, document))
        save_proposal(directory, document, "Mara", CMU_MARA)
        self.assertFalse(preparation_pending(directory, document))
        saved = json.loads((directory / "pronunciations.json").read_text())
        saved["discovery"]["version"] = DISCOVERY_VERSION - 1
        write_json(directory / "pronunciations.json", saved, overwrite=True)
        self.assertTrue(preparation_pending(directory, document))

    def test_proposal_is_pending_only_and_requires_valid_cmu(self):
        directory, document = self.book()
        save_candidates(directory, document, [candidate()])
        bad = [None, {"alphabet": "respelling", "value": "MAH-ruh"},
               {"alphabet": "cmu_arpabet", "value": "NOPE"},
               {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0", "status": "confirmed"}]
        before = (directory / "pronunciations.json").read_bytes()
        for proposal in bad:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                save_proposal(directory, document, "Mara", proposal)
        for status in ("confirmed", "ignored"):
            entry = dict(candidate(), status=status, proposal=CMU_MARA)
            if status == "confirmed":
                entry["pronunciation"] = CMU_MARA
            with self.subTest(status=status), self.assertRaises(ValueError):
                validate_entry(entry, document["source"])
        self.assertEqual((directory / "pronunciations.json").read_bytes(), before)
        saved = save_proposal(directory, document, "Mara", CMU_MARA)
        self.assertEqual(saved["proposal"], CMU_MARA)
        self.assertEqual(saved["status"], "pending")
        self.assertEqual(saved["proposal_origin_directory"], str(directory))
        self.assertNotIn("pronunciation", saved)

    def test_candidate_rediscovery_preserves_pending_proposal_and_explicit_choices_remove_it(self):
        directory, document = self.book()
        save_candidates(directory, document, [candidate()])
        original = deepcopy(document)
        protected = {name: (directory / name).read_bytes() for name in ("analysis.json", "manifest.json", "source.json", "cast.json")}
        save_proposal(directory, document, "Mara", CMU_MARA)
        refreshed = dict(candidate(), reason="A newly discovered reason for review.")
        saved = save_candidates(directory, document, [refreshed])[0]
        self.assertEqual(saved["proposal"], CMU_MARA)
        self.assertEqual(saved["reason"], refreshed["reason"])
        before = (directory / "pronunciations.json").read_bytes()
        saved = save_proposal(directory, document, "Mara", {"alphabet": "cmu_arpabet", "value": "M EH1 R AH0"})
        self.assertEqual(saved["proposal"], CMU_MARA)
        self.assertEqual((directory / "pronunciations.json").read_bytes(), before)
        saved = set_pronunciation(directory, document, "Mara", CMU_MARA)
        self.assertNotIn("proposal", saved)
        self.assertEqual(saved["pronunciation"], CMU_MARA)
        set_pronunciation(directory, document, "Mara", status="pending")
        save_proposal(directory, document, "Mara", CMU_MARA)
        self.assertNotIn("proposal", set_pronunciation(directory, document, "Mara", status="ignored"))
        self.assertEqual(document, original)
        for name, data in protected.items():
            self.assertEqual((directory / name).read_bytes(), data)

    def test_proposal_save_does_not_create_a_candidate_or_replace_a_resolved_decision(self):
        directory, document = self.book()
        self.assertIsNone(save_proposal(directory, document, "Mara", CMU_MARA))
        self.assertFalse((directory / "pronunciations.json").exists())
        save_candidates(directory, document, [candidate()])
        for status in ("confirmed", "ignored"):
            set_pronunciation(directory, document, "Mara", CMU_MARA if status == "confirmed" else None, status)
            before = (directory / "pronunciations.json").read_bytes()
            stat_before = (directory / "pronunciations.json").stat()
            saved = save_proposal(directory, document, "Mara", CMU_MARA)
            self.assertEqual(saved["status"], status)
            self.assertNotIn("proposal", saved)
            self.assertEqual((directory / "pronunciations.json").read_bytes(), before)
            self.assertEqual((directory / "pronunciations.json").stat().st_mtime_ns, stat_before.st_mtime_ns)

    def test_pending_proposals_inherit_even_after_local_discovery_without_changing_ancestors(self):
        first, first_doc = self.book()
        second, second_doc = self.book("second", first)
        third, third_doc = self.book("third", second)
        save_candidates(first, first_doc, [candidate()])
        save_candidates(second, second_doc, [candidate()])
        save_candidates(third, third_doc, [candidate()])
        self.assertTrue(preparation_pending(third, third_doc))
        save_proposal(first, first_doc, "Mara", CMU_MARA)
        before = snapshot(first)
        self.assertEqual(load_pronunciations(second)[0]["proposal"], CMU_MARA)
        self.assertEqual(load_pronunciations(third)[0]["proposal"], CMU_MARA)
        self.assertEqual(load_pronunciations(third)[0]["proposal_origin_directory"], str(first))
        self.assertFalse(load_pronunciations(third)[0]["inherited"])
        self.assertFalse(preparation_pending(third, third_doc))
        save_candidates(third, third_doc, [candidate()])
        self.assertEqual(load_pronunciations(third)[0]["proposal"], CMU_MARA)
        for directory in (second, third):
            local = json.loads((directory / "pronunciations.json").read_text())["entries"][0]
            self.assertNotIn("proposal", local)
            self.assertNotIn("proposal_origin_directory", local)
        save_proposal(third, third_doc, "Mara", {"alphabet": "cmu_arpabet", "value": "M EH1 R AH0"})
        self.assertEqual(load_pronunciations(third)[0]["proposal"], CMU_MARA)
        self.assertEqual(snapshot(first), before)

    def test_proposal_for_inherited_pending_word_has_evidence_from_current_book(self):
        first, first_doc = self.book()
        second, second_doc = self.book("second", first)
        save_candidates(first, first_doc, [candidate()])
        before = snapshot(first)
        saved = save_proposal(second, second_doc, "Mara", CMU_MARA)
        self.assertEqual(saved["proposal"], CMU_MARA)
        self.assertFalse(saved["inherited"])
        local = json.loads((second / "pronunciations.json").read_text())["entries"][0]
        self.assertEqual(validate_entry(local, second_doc["source"]), local)
        self.assertEqual(snapshot(first), before)

    def test_late_parent_resolution_prevents_saving_a_completed_model_guess(self):
        first, first_doc = self.book()
        second, second_doc = self.book("second", first)
        save_candidates(second, second_doc, [candidate()])
        before = (second / "pronunciations.json").read_bytes()
        for status in ("confirmed", "ignored"):
            set_pronunciation(first, first_doc, "Mara", CMU_MARA if status == "confirmed" else None, status)
            saved = save_proposal(second, second_doc, "Mara", CMU_MARA)
            self.assertEqual(saved["status"], status)
            self.assertTrue(saved["inherited"])
            self.assertNotIn("proposal", saved)
            self.assertFalse(preparation_pending(second, second_doc))
            self.assertEqual((second / "pronunciations.json").read_bytes(), before)

    def test_proposal_is_shared_by_case_variants_but_not_another_spelling(self):
        directory = self.root / "minimal"
        directory.mkdir()
        document = {"source": {"book": {"id": "case-test"}, "blocks": [{"id": "b1", "text": "Mara mara Marah"}]}, "cast": {}}
        write_json(directory / "source.json", document["source"])
        save_candidates(directory, document, [candidate(word) for word in ("Mara", "mara", "Marah")])
        save_proposal(directory, document, "Mara", CMU_MARA)
        entries = {entry["word"]: entry for entry in load_pronunciations(directory)}
        self.assertEqual(entries["Mara"]["proposal"], CMU_MARA)
        self.assertEqual(entries["Mara"]["variants"], ["Mara", "mara"])
        exact = {entry["word"]: entry for entry in load_pronunciations(directory, group_cases=False)}
        self.assertNotIn("proposal", exact["mara"])
        self.assertNotIn("proposal", entries["Marah"])
        self.assertTrue(preparation_pending(directory, document))

    def test_candidate_merge_preserves_confirmed_and_ignored_local_choices(self):
        directory, document = self.book()
        original = deepcopy(document)
        preserved = {name: (directory / name).read_bytes() for name in ("analysis.json", "manifest.json", "source.json", "cast.json")}
        save_candidates(directory, document, [candidate()])
        set_pronunciation(directory, document, "Mara", {"alphabet": "respelling", "value": "MAH-ruh"})
        saved = save_candidates(directory, document, [candidate()])
        self.assertEqual(saved[0]["status"], "confirmed")
        self.assertFalse(saved[0]["inherited"])
        self.assertEqual(saved[0]["origin_directory"], str(directory))
        set_pronunciation(directory, document, "Mara", status="ignored")
        saved = save_candidates(directory, document, [candidate()])
        self.assertEqual(saved[0]["status"], "ignored")
        self.assertNotIn("pronunciation", saved[0])
        self.assertEqual(document, original)
        for name, contents in preserved.items():
            self.assertEqual((directory / name).read_bytes(), contents)

    def test_late_ancestor_confirmation_defeats_child_pending_and_nearest_override_wins(self):
        first, first_doc = self.book()
        second, second_doc = self.book("second", first)
        third, third_doc = self.book("third", second)
        save_candidates(third, third_doc, [candidate()])
        self.assertEqual(load_pronunciations(third)[0]["status"], "pending")
        set_pronunciation(first, first_doc, "Mara", {"alphabet": "respelling", "value": "MAH-ruh"})
        ancestor_before = snapshot(first)
        inherited = load_pronunciations(third)[0]
        self.assertEqual(inherited["status"], "confirmed")
        self.assertEqual(inherited["origin_directory"], str(first))
        self.assertTrue(inherited["inherited"])
        save_candidates(second, second_doc, [candidate()])
        self.assertEqual(load_pronunciations(third)[0]["origin_directory"], str(first))
        set_pronunciation(second, second_doc, "Mara", {"alphabet": "respelling", "value": "MAIR-uh"})
        self.assertEqual(load_pronunciations(third)[0]["pronunciation"]["value"], "MAIR-uh")
        set_pronunciation(third, third_doc, "Mara", status="ignored")
        self.assertEqual(load_pronunciations(third)[0]["status"], "ignored")
        reset_pronunciation(third, "Mara")
        self.assertEqual(load_pronunciations(third)[0]["origin_directory"], str(second))
        reset_pronunciation(second, "Mara")
        self.assertEqual(load_pronunciations(third)[0]["origin_directory"], str(first))
        self.assertEqual(snapshot(first), ancestor_before)

    def test_inherited_ignored_decision_is_not_shadowed_by_local_pending(self):
        first, first_doc = self.book()
        second, second_doc = self.book("second", first)
        set_pronunciation(first, first_doc, "Mara", status="ignored")
        save_candidates(second, second_doc, [candidate()])
        self.assertEqual(load_pronunciations(second)[0]["status"], "ignored")
        self.assertEqual(pronunciation_counts(load_pronunciations(second)), {"pending": 0, "confirmed": 0, "ignored": 1})

    def test_ancestor_words_absent_here_are_hidden_but_remain_available_to_later_books(self):
        first, first_doc = self.book()
        set_pronunciation(first, first_doc, "softly", status="ignored")
        second, _ = self.book("second", first)
        self.assertEqual(load_pronunciations(second), [])
        third, _ = self.book("third", second, include_softly=True)
        entry = load_pronunciations(third)[0]
        self.assertEqual(entry["word"], "softly")
        self.assertEqual(entry["origin_directory"], str(first))

    def test_wrong_source_registry_and_symlink_are_rejected(self):
        directory, document = self.book()
        save_candidates(directory, document, [candidate()])
        path = directory / "pronunciations.json"
        record = json.loads(path.read_text())
        record["source_fingerprint"] = "0" * 64
        write_json(path, record, overwrite=True)
        with self.assertRaises(ValueError):
            load_pronunciations(directory)
        path.unlink()
        path.symlink_to(self.root / "outside")
        with self.assertRaises(ValueError):
            set_pronunciation(directory, document, "Mara", status="ignored")
        self.assertFalse((self.root / "outside").exists())

    def test_discovery_cannot_confirm_or_forge_source_evidence(self):
        directory, document = self.book()
        bad = [dict(candidate(), status="confirmed", pronunciation={"alphabet": "respelling", "value": "MAH-ruh"}),
               dict(candidate(), evidence=[{"block_id": "b2", "quote": "Mara"}]),
               dict(candidate(), word="Mar", id=word_id("Mar")),
               dict(candidate(), evidence=[{"block_id": "b1", "quote": "Mara said loudly"}]),
               dict(candidate(), status=[]), dict(candidate(), kind=[])]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                save_candidates(directory, document, [item])
        self.assertFalse((directory / "pronunciations.json").exists())

    def test_quote_of_prefix_is_not_whole_source_word_evidence(self):
        directory = self.root / "minimal"
        directory.mkdir()
        document = {"source": {"book": {"id": "prefix-test"}, "blocks": [{"id": "b1", "text": "Qarmian walked."}]}, "cast": {}}
        write_json(directory / "source.json", document["source"])
        with self.assertRaises(ValueError):
            save_candidates(directory, document, [candidate("Qarm")])

    def test_manual_decision_requires_exact_word_in_current_source(self):
        directory, document = self.book()
        for word in ("mara", "Mar", "Absent"):
            with self.subTest(word=word), self.assertRaises(ValueError):
                set_pronunciation(directory, document, word, status="ignored")

    def test_reset_removes_only_one_local_decision(self):
        directory, document = self.book()
        save_candidates(directory, document, [candidate(), candidate("They", "b2")])
        reset_pronunciation(directory, "Mara")
        self.assertEqual([entry["word"] for entry in load_pronunciations(directory)], ["They"])

    def test_case_variants_share_decisions_while_exact_storage_and_similar_names_remain_distinct(self):
        directory = self.root / "minimal"
        directory.mkdir()
        document = {"source": {"book": {"id": "case-test"}, "blocks": [{"id": "b1", "text": "Mara mara Marah"}]}, "cast": {}}
        write_json(directory / "source.json", document["source"])
        write_json(directory / "manifest.json", {"source_fingerprint": fingerprint(document["source"])})
        save_candidates(directory, document, [candidate(word) for word in ("Mara", "mara", "Marah")])
        set_pronunciation(directory, document, "Mara", status="ignored")
        self.assertEqual({entry["word"]: entry["status"] for entry in load_pronunciations(directory)}, {"Mara": "ignored", "Marah": "pending"})
        self.assertEqual({entry["word"]: entry["status"] for entry in load_pronunciations(directory, group_cases=False)},
                         {"Mara": "ignored", "mara": "ignored", "Marah": "pending"})


if __name__ == "__main__":
    unittest.main()
