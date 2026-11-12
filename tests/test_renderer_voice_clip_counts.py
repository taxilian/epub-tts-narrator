"""Per-voice reference clip counts chosen in casting reach the next voice creation only."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

from epub_tts_narrator.audition_scripts import prepare_samples
from epub_tts_narrator.renderer import reference_clips
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS, extra_reference_clips
from epub_tts_narrator.renderer.voice_clip_counts import clip_choice, clip_overrides, extra_clip_plan
from audition_fixtures import prepare_bundle
from test_audition_scripts import ExtraRequests, bundle as script_bundle


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"
GEMINI = {"engine": "gemini_voice_design", "model": "fixture"}


class ClipCountTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "book"
        shutil.copytree(EXAMPLE, self.directory)
        self.bundle = prepare_bundle(load_bundle(self.directory))
        self.config = resolve_book_config(self.bundle, deepcopy(DEFAULTS))
        self.session = CastingSession(self.bundle, self.config, notify=lambda _: None)
        self.addCleanup(self.session.close)
        self.voice_id = self.bundle["cast"]["voices"][0]["id"]
        self.other_id = self.bundle["cast"]["voices"][1]["id"]

    def row(self, voice_id):
        return next(row for row in self.session.rows() if row["voice_id"] == voice_id)

    def gemini(self):
        return dict(self.config, voice_design={"type": "gemini"})

    def test_a_saved_count_shows_in_casting_and_none_returns_to_the_default(self):
        self.assertEqual(self.row(self.voice_id)["reference_clips"]["count"], 3)
        self.assertEqual(self.row(self.voice_id)["reference_clips"]["source"], "default")
        self.session.set_reference_clips(self.voice_id, 1)
        self.assertEqual((self.row(self.voice_id)["reference_clips"]["count"],
                          self.row(self.voice_id)["reference_clips"]["source"]), (1, "voice"))
        self.assertEqual(self.row(self.other_id)["reference_clips"]["source"], "default")
        record = read_json(self.directory / "voice-clips.json")
        self.assertEqual(record["reference_clips"], {self.voice_id: 1})
        self.session.set_reference_clips(self.voice_id, None)
        self.assertEqual(self.row(self.voice_id)["reference_clips"]["source"], "default")

    def test_invalid_counts_and_foreign_files_fail_without_changing_the_sidecar(self):
        self.session.set_reference_clips(self.voice_id, 2)
        path = self.directory / "voice-clips.json"
        before = path.read_bytes()
        for value in (0, 4, "2", True, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.session.set_reference_clips(self.voice_id, value)
            self.assertEqual(path.read_bytes(), before)
        good = read_json(path)
        for key, value in (("book_id", "wrong"), ("series_id", "wrong"), ("reference_clips", {"unknown": 2})):
            write_json(path, {**good, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                clip_overrides(self.bundle, self.config)

    def test_the_shared_voice_count_wins_over_a_role_count(self):
        overrides = {self.voice_id: 2, self.other_id: 1}
        canonical = {self.other_id: self.voice_id}.get
        self.assertEqual(clip_choice(self.bundle, self.config, self.other_id,
                                     canonical_id=lambda value: canonical(value, value), overrides=overrides),
                         (2, "voice"))

    def test_books_without_choices_prepare_exactly_the_global_extra_count(self):
        for config in (self.config, self.gemini()):
            with self.subTest(designer=config.get("voice_design")):
                self.assertEqual(extra_clip_plan(self.bundle, config), extra_reference_clips(config))

    def test_saved_counts_become_a_per_voice_plan_only_for_a_designer_that_keeps_one_voice(self):
        self.session.set_reference_clips(self.voice_id, 1)
        plan = extra_clip_plan(self.bundle, self.gemini())
        self.assertEqual(plan[self.voice_id], 0)
        self.assertEqual(plan[self.other_id], 2)
        self.assertEqual(extra_clip_plan(self.bundle, self.config), 0)

    def test_creation_plans_the_voice_s_own_number_of_extra_scripts(self):
        def planned():
            return reference_clips.plan_scripts(self.config, GEMINI, self.bundle, self.voice_id,
                                                lambda value: value, {})
        self.assertTrue(planned())
        self.session.set_reference_clips(self.voice_id, 1)
        self.assertEqual(planned(), [])
        self.session.set_reference_clips(self.voice_id, None)
        self.assertTrue(planned())

    def test_stop_sharing_removes_only_this_book_s_link(self):
        with self.assertRaisesRegex(ValueError, "no sharing link chosen in this book"):
            self.session.stop_sharing(self.other_id)
        self.session.share_voice(self.other_id, self.voice_id)
        self.assertEqual(self.row(self.other_id)["canonical_voice_id"], self.voice_id)
        result = self.session.stop_sharing(self.other_id)
        self.assertEqual(result["affected_voice_ids"], [self.other_id])
        self.assertEqual(self.row(self.other_id)["canonical_voice_id"], self.other_id)


class PerVoiceExtraScriptTests(unittest.TestCase):
    def test_a_voice_with_no_extra_clips_asks_for_no_invented_scripts(self):
        requests = ExtraRequests()
        result = prepare_samples(script_bundle(), requests, workers=1, extra_clips={"v1": 2, "vn": 0})
        extra_calls = [call for call in requests.calls if call["mode"] == "extra_reference_scripts"]
        self.assertEqual([call["voice"]["id"] for call in extra_calls], ["v1"])
        self.assertEqual(set(result["extra_scripts"]), {"v1"})

    def test_one_extra_clip_asks_for_one_script(self):
        requests = ExtraRequests()
        prepare_samples(script_bundle(), requests, workers=1, extra_clips={"v1": 1})
        call = next(call for call in requests.calls if call["mode"] == "extra_reference_scripts")
        self.assertEqual(call["count"], 1)

    def test_invalid_per_voice_counts_are_refused(self):
        for value in ({"v1": 3}, {"v1": -1}, {"v1": "1"}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "zero through two"):
                prepare_samples(script_bundle(), None, extra_clips=value)


if __name__ == "__main__":
    unittest.main()
