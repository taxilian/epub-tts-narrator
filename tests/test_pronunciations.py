from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import fingerprint, write_json
from epub_tts_narrator.pronunciations import discovery_pending, load_pronunciations, pronunciation_counts, reset_pronunciation, save_candidates, set_pronunciation, validate_pronunciation, validate_word, word_id
from epub_tts_narrator.series import link_analysis
from shared_fixtures import snapshot
from test_series import fixture, propose


def candidate(word="Mara", block_id="b1"):
    return {"id": word_id(word), "word": word, "status": "pending", "kind": "name",
            "reason": "The name may have more than one reading.", "evidence": [{"block_id": block_id, "quote": word}]}


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

    def test_exact_case_variants_remain_independent_entries(self):
        directory = self.root / "minimal"
        directory.mkdir()
        document = {"source": {"book": {"id": "case-test"}, "blocks": [{"id": "b1", "text": "Mara mara Marah"}]}, "cast": {}}
        write_json(directory / "source.json", document["source"])
        write_json(directory / "manifest.json", {"source_fingerprint": fingerprint(document["source"])})
        save_candidates(directory, document, [candidate(word) for word in ("Mara", "mara", "Marah")])
        set_pronunciation(directory, document, "Mara", status="ignored")
        self.assertEqual({entry["word"]: entry["status"] for entry in load_pronunciations(directory)}, {"Mara": "ignored", "mara": "pending", "Marah": "pending"})


if __name__ == "__main__":
    unittest.main()
