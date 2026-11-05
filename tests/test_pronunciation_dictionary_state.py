"""Dictionary queue policy over real registries, inheritance and review sidecars."""

import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pronunciation_dictionary import review_candidates
from epub_tts_narrator.pronunciations import (
    load_pronunciations, preparation_pending, save_candidates, save_proposal, set_pronunciation,
)
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import digest, write_json
from epub_tts_narrator.renderer.pronunciation_review import PronunciationReviewSession
from shared_fixtures import renderer_config
import test_pronunciations as registry_tests


class PronunciationDictionaryStateTests(unittest.TestCase):
    book = registry_tests.PronunciationsTests.book

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = renderer_config(self.root)
        for section, key in (("voices", "directory"), ("voices", "series_id"),
                             ("render", "work_dir"), ("render", "output_dir"), ("render", "epub")):
            self.config[section][key] = None

    def test_legacy_pending_is_hidden_without_deleting_registry_or_proposal(self):
        directory, document = self.book()
        save_candidates(directory, document, [registry_tests.candidate("softly")])
        save_proposal(directory, document, "softly", registry_tests.CMU_MARA)
        before = (directory / "pronunciations.json").read_bytes()
        session = PronunciationReviewSession(load_bundle(directory), self.config)

        rows = session.rows()

        self.assertEqual(rows, [])
        self.assertEqual((directory / "pronunciations.json").read_bytes(), before)
        registry_tests.PronunciationsTests.prepare_names(self, directory)
        self.assertFalse(preparation_pending(directory, document))
        self.assertEqual(load_pronunciations(directory)[0]["proposal"], registry_tests.CMU_MARA)

    def test_draft_on_filtered_pending_word_survives_and_stays_reviewable(self):
        directory, document = self.book()
        save_candidates(directory, document, [registry_tests.candidate("softly")])
        session = PronunciationReviewSession(load_bundle(directory), self.config)
        session.edit("softly", registry_tests.CMU_MARA)

        rows = session.rows()

        self.assertEqual([(row["word"], row["status"]) for row in rows], [("softly", "pending")])
        self.assertEqual(rows[0]["draft"], registry_tests.CMU_MARA)
        self.assertNotIn("pronunciation", load_pronunciations(directory)[0])

    def test_manual_pending_selection_survives_rediscovery(self):
        directory, document = self.book()
        save_candidates(directory, document, [registry_tests.candidate("softly")])
        set_pronunciation(directory, document, "softly", status="pending")

        entries = save_candidates(directory, document, [registry_tests.candidate("softly")])

        self.assertEqual([row["word"] for row in review_candidates(document, entries)], ["softly"])
        self.assertTrue(entries[0]["review_requested"])

    def test_resolved_ancestor_wins_over_filtered_pending_descendant(self):
        parent, original = self.book()
        set_pronunciation(parent, original, "left", registry_tests.CMU_MARA)
        child, document = self.book("second", previous=parent)
        save_candidates(child, document, [registry_tests.candidate("left", block_id="b2")])
        before = (parent / "pronunciations.json").read_bytes()

        entries = review_candidates(document, load_pronunciations(child))

        self.assertEqual([(row["word"], row["status"], row["inherited"]) for row in entries],
                         [("left", "confirmed", True)])
        self.assertEqual(entries[0]["pronunciation"], registry_tests.CMU_MARA)
        self.assertEqual((parent / "pronunciations.json").read_bytes(), before)

    def test_ignored_dictionary_word_stays_visible_after_rediscovery(self):
        directory, document = self.book()
        set_pronunciation(directory, document, "softly", status="ignored")

        entries = save_candidates(directory, document, [registry_tests.candidate("softly")])

        self.assertEqual([(row["word"], row["status"]) for row in review_candidates(document, entries)],
                         [("softly", "ignored")])

    def test_inherited_manual_review_survives_local_proposal_checkpoint(self):
        parent, original = self.book()
        save_candidates(parent, original, [registry_tests.candidate("left", block_id="b2")])
        set_pronunciation(parent, original, "left", status="pending")
        child, document = self.book("second", previous=parent)

        save_proposal(child, document, "left", registry_tests.CMU_MARA)

        entries = review_candidates(document, load_pronunciations(child))
        self.assertEqual([row["word"] for row in entries], ["left"])
        self.assertEqual(entries[0]["proposal"], registry_tests.CMU_MARA)

    def test_legacy_display_estimate_is_ignored_without_changing_choice(self):
        directory, document = self.book()
        set_pronunciation(directory, document, "Mara", status="ignored")
        session = PronunciationReviewSession(load_bundle(directory), self.config)
        session.directory.mkdir(parents=True)
        request = {"word": "Mara", "respelling": None}
        entries = {digest(request): {"request": request, "pronunciation": registry_tests.CMU_MARA}}
        write_json(session.directory / "cmu-estimates.json", {
            "format": "epub-tts-pronunciation-cmu-estimates", "version": 1,
            "book_id": document["source"]["book"]["id"],
            "source_fingerprint": session.bundle["manifest"]["source_fingerprint"],
            "entries": entries, "sha256": digest(entries)})
        registry = (directory / "pronunciations.json").read_bytes()
        estimates = (session.directory / "cmu-estimates.json").read_bytes()

        self.assertEqual(session.description(session.rows()[0]), "default pronunciation")
        self.assertEqual(session.description(session.rows()[0]), "default pronunciation")

        self.assertEqual((session.directory / "cmu-estimates.json").read_bytes(), estimates)
        self.assertEqual((directory / "pronunciations.json").read_bytes(), registry)
