"""Only bounded copied letter runs may be restored, visibly and reviewably."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.contracts import readiness, validate_document
from epub_tts_narrator.pipeline import (
    CachedRequests, align_attribution_response, align_units, analyze, base_document,
    make_units, restore_repeated_runs, seed_cast,
)
from test_contracts import valid_document
from test_pipeline import FakeClient


def fixture(text, pieces):
    source = deepcopy(valid_document()["source"])
    source["blocks"] = [dict(source["blocks"][0], text=text)]
    source["chapters"][0]["block_ids"] = ["b1"]
    cast = seed_cast()
    cast["characters"] = [{"id": "actor", "name": "Mara", "aliases": [], "description": "The speaker.", "voice_id": "actor_voice", "evidence": []}]
    cast["roles"].append({"id": "actor_role", "kind": "dialogue", "character_id": "actor", "voice_id": "actor_voice", "description": "The speaker's dialogue."})
    cast["voices"].append({"id": "actor_voice", "description": "A distinct voice.", "traits": []})
    units = [unit for batch in make_units(source) for unit in batch]
    response = {"units": [{"unit_id": units[0]["unit_id"], "segments": [
        {"text": piece, "kind": "narration" if index == 0 else "dialogue",
         "role_id": "narrator_main" if index == 0 else "actor_role", "basis": "explicit",
         "reason": "Original attribution reason.", "evidence": [], "candidates": [], "review_required": False,
         "delivery": {"direction": "", "evidence": []}} for index, piece in enumerate(pieces)]}]}
    return source, cast, units, response


class RepeatedRunTests(unittest.TestCase):
    def test_inserted_and_omitted_repeated_letters_restore_exact_source_and_require_review(self):
        for letter, source_count, model_count in (("s", 4, 3), ("s", 3, 4), ("é", 3, 2), ("s", 2, 3)):
            with self.subTest(letter=letter, source=source_count, model=model_count):
                prefix = "Mara said, "
                source_text = prefix + "“I " + letter * source_count + "aid stop.”"
                model_text = "“I " + letter * model_count + "aid stop.”"
                source, cast, units, response = fixture(source_text, [prefix, model_text])
                original = deepcopy((source, cast, units, response))
                with self.assertRaises(ValueError):
                    align_units(source, cast, units, response)
                accepted = align_attribution_response(source, cast, units, response)
                self.assertEqual("".join(row["text"] for row in accepted), source_text)
                self.assertEqual([row["role_id"] for row in accepted], ["narrator_main", "actor_role"])
                self.assertFalse(accepted[0]["attribution"]["review_required"])
                self.assertTrue(accepted[1]["attribution"]["review_required"])
                self.assertIn("Source repeated-letter run count restored", accepted[1]["attribution"]["reason"])
                record = accepted[1]["source_restoration"]
                change = record["changes"][0]
                self.assertEqual(change["source_text"], letter * source_count)
                self.assertEqual(change["model_text"], letter * model_count)
                self.assertEqual(source_text[change["source_start"]:change["source_end"]], change["source_text"])
                self.assertEqual(change["source_count"], source_count)
                self.assertEqual(change["model_count"], model_count)
                document = base_document(source, cast, accepted)
                self.assertEqual(validate_document(document), [])
                self.assertTrue(readiness(document)["ready"])
                self.assertEqual((source, cast, units, response), original)

    def test_multiple_changed_runs_restore_together_and_keep_explicit_records(self):
        source, cast, units, response = fixture("Mara ssssaid buzzzz.", ["Mara sssaid buzzzzz."])
        restored, records = restore_repeated_runs(units, response)
        self.assertEqual(restored["units"][0]["segments"][0]["text"], source["blocks"][0]["text"])
        self.assertEqual(len(records), 1)
        self.assertEqual([(row["source_count"], row["model_count"]) for row in records[0]["changes"]], [(4, 3), (4, 5)])
        self.assertLess(len(json.dumps(records)), 4000)
        accepted = align_attribution_response(source, cast, units, response)
        self.assertEqual(accepted[0]["source_restoration"], records[0])

    def test_normal_spelling_words_quantities_punctuation_whitespace_and_large_changes_are_rejected(self):
        pairs = [("Mara goes to town.", "Mara goes too town."),
                 ("Mara said hello.", "Mara sald hello."),
                 ("It had even been lit.", "It had been lit."),
                 ("Mara has 1000 coins.", "Mara has 10000 coins."),
                 ("Mara paused...", "Mara paused.."),
                 ("Mara  spoke.", "Mara   spoke."),
                 ("Mara Café.", "Mara Cafe\u0301."),
                 ("Mara " + "s" * 33 + "aid.", "Mara " + "s" * 25 + "aid."),
                 ("Mara " + "s" * 12 + "aid.", "Mara " + "s" * 3 + "aid."),
                 ("Mara " + "sss " * 17 + "spoke.", "Mara " + "ss " * 17 + "spoke.")]
        for expected, model in pairs:
            with self.subTest(source=expected, model=model):
                source, cast, units, response = fixture(expected, [model])
                original = deepcopy(response)
                with self.assertRaises(ValueError):
                    restore_repeated_runs(units, response)
                with self.assertRaises(ValueError):
                    align_attribution_response(source, cast, units, response)
                self.assertEqual(response, original)

    def test_any_segment_boundary_inside_an_altered_run_is_rejected(self):
        for pieces in (["Mara ss", "said stop."], ["Mara ", "s", "ssaid stop."]):
            source, cast, units, response = fixture("Mara ssssaid stop.", pieces)
            with self.subTest(pieces=pieces), self.assertRaisesRegex(ValueError, "boundary inside"):
                restore_repeated_runs(units, response)
            with self.assertRaises(ValueError):
                align_attribution_response(source, cast, units, response)

    def test_boundaries_at_run_edges_remain_and_only_changed_piece_is_flagged(self):
        source, cast, units, response = fixture("Mara ssssaid stop.", ["Mara ", "sss", "aid stop."])
        accepted = align_attribution_response(source, cast, units, response)
        self.assertEqual([row["text"] for row in accepted], ["Mara ", "ssss", "aid stop."])
        self.assertEqual([row["attribution"]["review_required"] for row in accepted], [False, True, False])
        self.assertEqual([row["role_id"] for row in accepted], ["narrator_main", "actor_role", "actor_role"])
        self.assertEqual([(row["start"], row["end"]) for row in accepted], [(0, 5), (5, 9), (9, 18)])

    def test_restoration_cannot_fill_malformed_attribution_or_trust_model_metadata(self):
        for bad in ({"reason": None}, {"review_required": "false"}):
            source, cast, units, response = fixture("Mara ssssaid stop.", ["Mara sssaid stop."])
            response["units"][0]["segments"][0].update(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                align_attribution_response(source, cast, units, response)
        source, cast, units, response = fixture("Mara ssssaid stop.", ["Mara sssaid stop."])
        response["units"][0]["segments"][0]["source_restoration"] = {"method": "model-approved", "review_required": False}
        accepted = align_attribution_response(source, cast, units, response)
        self.assertEqual(accepted[0]["source_restoration"]["method"], "bounded_repeated_letter_source_restoration")
        self.assertTrue(accepted[0]["attribution"]["review_required"])
        response["units"][0]["segments"][0]["text"] = source["blocks"][0]["text"]
        self.assertNotIn("source_restoration", align_attribution_response(source, cast, units, response)[0])

    def test_cache_preserves_raw_model_response_and_replays_identical_reviewable_restoration(self):
        source, cast, units, response = fixture("Mara ssssaid stop.", ["Mara sssaid stop."])
        class Client:
            identity = {"model": "test", "base_url": "https://example.invalid"}
            calls = 0
            def complete(self, system, payload):
                self.calls += 1
                return deepcopy(response)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, directory, repair_attempts=0)
            validate = lambda value: align_attribution_response(source, cast, units, value)
            first = requests.run("unchanged system", {"units": units}, validate)
            second = requests.run("unchanged system", {"units": units}, validate)
            cached = json.loads(next(Path(directory).glob("*.json")).read_text())
        self.assertEqual(client.calls, 1)
        self.assertEqual(first, second)
        self.assertEqual(cached, response)
        self.assertEqual(cached["units"][0]["segments"][0]["text"], "Mara sssaid stop.")
        self.assertFalse(cached["units"][0]["segments"][0]["review_required"])
        with self.assertRaises(ValueError):
            align_units(source, cast, units, cached)

    def test_full_analysis_restores_stutter_without_extra_calls_and_records_review(self):
        source = deepcopy(valid_document()["source"])
        source["blocks"][0]["text"] = "Mara said softly, “I ssssaid stop.”"
        class StutterClient(FakeClient):
            def attribute(self, payload):
                response = super().attribute(payload)
                for unit in response["units"]:
                    for piece in unit["segments"]:
                        piece["text"] = piece["text"].replace("ssss", "sss")
                return response
        client = StutterClient()
        with tempfile.TemporaryDirectory() as directory:
            document = analyze(source, client, directory, repair_attempts=0, identity_consensus=False)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(validate_document(document), [])
        restored = [item for item in document["segments"] if "source_restoration" in item]
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["text"], "“I ssssaid stop.”")
        self.assertTrue(restored[0]["attribution"]["review_required"])
        self.assertTrue(document["readiness"]["ready"])


if __name__ == "__main__":
    unittest.main()
