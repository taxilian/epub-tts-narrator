import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.pronunciation_dictionary import review_candidates
from epub_tts_narrator.pronunciations import load_pronunciations, save_candidates, save_proposal, set_pronunciation
from epub_tts_narrator.renderer.pronunciation_review import PronunciationReviewSession
from epub_tts_narrator.renderer.common import digest
from shared_fixtures import renderer_config, snapshot
import test_pronunciation_groups as families
from test_pronunciations import candidate


class PronunciationDictionaryGroupsTests(unittest.TestCase):
    minimal = families.PronunciationGroupsTests.minimal
    book = families.PronunciationGroupsTests.book

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def session(self, directory, document):
        bundle = dict(document, directory=str(directory), manifest={})
        return PronunciationReviewSession(bundle, renderer_config(self.root))

    def test_cast_name_retains_family_when_representative_is_another_case(self):
        directory, document = self.minimal("ROSE Rose rose softly")
        document["cast"] = {"characters": [{"name": "Rose", "aliases": []}]}
        save_candidates(directory, document, [candidate("rose"), candidate("softly")])
        before = snapshot(directory)

        rows = self.session(directory, document).rows()

        self.assertEqual([row["word"] for row in rows], ["rose"])
        self.assertEqual(set(rows[0]["variants"]), {"ROSE", "Rose", "rose"})
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(snapshot(directory), before)

    def test_nonrepresentative_manual_intent_survives_grouping_and_rediscovery(self):
        directory, document = self.minimal("Softly softly SOFTLY")
        save_candidates(directory, document, [candidate("Softly"), dict(candidate("softly"), review_requested=True)])
        save_candidates(directory, document, [candidate("Softly"), candidate("softly")])

        rows = self.session(directory, document).rows()

        self.assertEqual([row["word"] for row in rows], ["Softly"])
        self.assertTrue(rows[0]["review_requested"])
        self.assertEqual(set(rows[0]["variants"]), {"Softly", "softly", "SOFTLY"})

    def test_manual_selection_marks_every_exact_family_member(self):
        directory, document = self.minimal("Softly softly SOFTLY")
        save_candidates(directory, document, [candidate("Softly")])

        set_pronunciation(directory, document, "softly", status="pending")
        save_candidates(directory, document, [candidate("Softly"), candidate("softly")])

        rows = self.session(directory, document).rows()
        self.assertEqual(len(rows), 1)
        raw = load_pronunciations(directory, group_cases=False)
        self.assertEqual({row["word"] for row in raw}, {"Softly", "softly", "SOFTLY"})
        self.assertTrue(all(row["review_requested"] for row in raw))

    def test_existing_local_proposal_target_keeps_inherited_manual_intent(self):
        parent, original = self.book("first", "Quiet returned.")
        save_candidates(parent, original, [candidate("Quiet", "b2")])
        set_pronunciation(parent, original, "Quiet", status="pending")
        child, document = self.book("second", "quiet returned.", parent)
        save_candidates(child, document, [candidate("quiet", "b2")])
        before = snapshot(parent)

        save_proposal(child, document, "quiet", families.QUIET)

        rows = review_candidates(document, load_pronunciations(child))
        self.assertEqual([row["word"] for row in rows], ["quiet"])
        raw = json.loads((child / "pronunciations.json").read_text())["entries"][0]
        self.assertTrue(raw["review_requested"])
        self.assertEqual(raw["evidence"], [{"block_id": "b2", "quote": "quiet"}])
        self.assertEqual(raw["proposal"], families.QUIET)
        self.assertEqual(snapshot(parent), before)

    def test_case_conflicts_remain_visible_for_dictionary_words(self):
        directory, document = self.minimal("Quiet quiet QUIET")
        save_candidates(directory, document, [candidate(word) for word in ("Quiet", "quiet", "QUIET")])
        path = directory / "pronunciations.json"
        record = json.loads(path.read_text())
        for row in record["entries"]:
            if row["word"] != "QUIET":
                row.update(status="confirmed", pronunciation=families.QUIET if row["word"] == "Quiet" else families.OTHER)
        write_json(path, record, overwrite=True)
        before = snapshot(directory)

        rows = self.session(directory, document).rows()

        self.assertEqual({row["word"] for row in rows}, {"Quiet", "quiet", "QUIET"})
        self.assertTrue(all(row["case_conflict"] for row in rows))
        self.assertEqual(snapshot(directory), before)

    def test_conflicting_drafts_keep_exact_partitions_and_explicit_word_lookup(self):
        directory, document = self.minimal("Quiet quiet QUIET")
        save_candidates(directory, document, [candidate("Quiet")])
        session = self.session(directory, document)
        session._write_draft("Quiet", families.QUIET, variants=["Quiet"])
        path = session.directory / "drafts.json"
        legacy = json.loads(path.read_text())
        legacy["entries"]["quiet"] = families.OTHER
        legacy["sha256"] = digest(legacy["entries"])
        write_json(path, legacy, overwrite=True)
        before = snapshot(directory)

        rows = session.rows()

        self.assertEqual({row["word"] for row in rows}, {"Quiet", "quiet", "QUIET"})
        self.assertEqual(session.rows("quiet")[0]["draft"], families.OTHER)
        self.assertEqual(session.rows("Quiet")[0]["draft"], families.QUIET)
        self.assertEqual(snapshot(directory), before)

    def test_explicit_word_can_select_filtered_dictionary_family(self):
        directory, document = self.minimal("Quiet quiet QUIET")
        save_candidates(directory, document, [candidate("Quiet")])
        session = self.session(directory, document)

        self.assertEqual(session.rows(), [])
        self.assertEqual(session.rows("QUIET")[0]["word"], "Quiet")

    def test_repeated_menu_rows_do_not_rescan_frozen_cast_source(self):
        class CountedBlocks(list):
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        directory, document = self.minimal("Rose softly")
        document["cast"] = {"characters": [{"name": "Rose", "aliases": []}]}
        save_candidates(directory, document, [candidate("Rose"), candidate("softly")])
        blocks = CountedBlocks(document["source"]["blocks"])
        document["source"]["blocks"] = blocks
        session = self.session(directory, document)
        first = session.rows()
        first_scan = blocks.iterations

        repeated = session.rows()

        self.assertEqual([row["word"] for row in first], ["Rose"])
        self.assertEqual(repeated, first)
        self.assertGreater(first_scan, 0)
        self.assertEqual(blocks.iterations, first_scan)
