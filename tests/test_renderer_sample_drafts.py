"""Excerpt defaults must preserve human edits and immutable saved voice references."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.voice_prompts import default_voice_prompt


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"
EXCERPT = (
    "I remember the road that brought us here. "
    "There is still time to bring our friends safely home before nightfall. "
    "We know the way across the river, and we can travel together."
)


class SampleDraftTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = load_bundle(EXAMPLE)
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "tts-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="sample-series")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"))
        self.engine_factory = Mock(side_effect=AssertionError("Drafts must not load TTS"))
        self.designer_factory = Mock(side_effect=AssertionError("Drafts must not load voice design"))
        self.verifier_factory = Mock(side_effect=AssertionError("Drafts must not load ASR"))
        self.sample = {
            "reference_text": EXCERPT,
            "source": {
                "kind": "book_excerpt", "book_id": "earlier-book",
                "chapter_id": "chapter-1", "segment_ids": ["mara-dialogue-1"],
                "source_ranges": [{"start": 0, "end": len(EXCERPT)}],
            },
        }
        self.selector = self.enterContext(patch(
            "epub_tts_narrator.renderer.samples.select_sample", return_value=self.sample))
        self.session = self.new_session()
        self.voice_id = self.bundle["cast"]["voices"][0]["id"]

    def new_session(self, bundle=None):
        session = CastingSession(
            bundle or self.bundle, self.config, engine_factory=self.engine_factory,
            verifier_factory=self.verifier_factory, designer_factory=self.designer_factory,
            notify=lambda _: None)
        self.addCleanup(session.close)
        return session

    def assert_no_models(self):
        self.engine_factory.assert_not_called()
        self.designer_factory.assert_not_called()
        self.verifier_factory.assert_not_called()

    def legacy_draft(self, *, text=None, prompt="The voice prompt I carefully edited."):
        payload = {
            "format": "audiobook-casting-draft", "version": 1,
            "series_id": self.config["voices"]["series_id"], "voice_id": self.voice_id,
            "prompt": prompt,
            "reference_text": self.config["voices"]["reference_text"] if text is None else text,
        }
        path = self.session._draft_path(self.voice_id)
        write_json(path, {**payload, "sha256": digest(payload)})
        return path

    def saved_voice(self, *, text=None, source=None, candidate=True):
        text = self.config["voices"]["reference_text"] if text is None else text
        audio_path = self.root / "existing-reference.wav"
        with wave.open(str(audio_path), "wb") as audio:
            audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x01\x00" * 4410)
        metadata = {
            "requested_description": "A selected voice with an edited prompt.",
            "requested_reference_text": text, "description_applied": True,
        }
        if source is not None:
            metadata["sample_source"] = deepcopy(source)
        arguments = (
            self.voice_id, self.bundle["cast"]["voices"][0], audio_path, text,
            {"engine": "fixture", "model": "fixture-v1"}, {"passed": True}, metadata,
        )
        with self.session._locked():
            if candidate:
                saved = self.session.store.publish_candidate(
                    *arguments, candidate_id="a" * 32)
                self.session.store.select_candidate(self.voice_id, saved["candidate_id"])
            else:
                saved = self.session.store.publish(*arguments)
        return saved

    def test_fresh_rows_show_book_excerpt_without_loading_models_or_writing_files(self):
        original_sample = deepcopy(self.sample)
        row = self.session.rows()[0]
        self.assertEqual(row["reference_text"], EXCERPT)
        self.assertEqual(row["sample_source"]["kind"], "book_excerpt")
        self.assertEqual(row["sample_source"]["segment_ids"], ["mara-dialogue-1"])
        self.assertEqual(row["prompt"], default_voice_prompt(self.bundle["cast"]["voices"][0]))
        self.assertIsNone(row["selected"])
        self.assertFalse((self.root / "voices").exists())
        self.assertEqual(self.sample, original_sample)
        call = self.selector.call_args_list[0]
        self.assertEqual(call.args, (self.bundle, self.voice_id))
        self.assertEqual(call.kwargs["fallback_text"], self.config["voices"]["reference_text"])
        self.assertEqual(call.kwargs["canonical_id"](self.voice_id), self.voice_id)
        self.assert_no_models()

    def test_prepare_saves_version_two_once_without_changing_selected_audio_or_binding(self):
        saved = self.saved_voice()
        before = {
            path: path.read_bytes() for path in self.session.store.directory.rglob("*")
            if path.is_file()
        }
        prepared = self.session.prepare_samples()
        row = next(item for item in prepared if item["voice_id"] == self.voice_id)
        self.assertTrue(row["updated"])
        self.assertEqual(row["reference_text"], EXCERPT)
        self.assertEqual(row["prompt"], saved["creation"]["requested_description"])
        path = self.session._draft_path(self.voice_id)
        record = read_json(path)
        self.assertEqual(record["version"], 2)
        self.assertEqual(record["sample_source"]["kind"], "book_excerpt")
        self.assertEqual(record["sha256"], digest({k: v for k, v in record.items() if k != "sha256"}))
        draft_bytes, draft_time = path.read_bytes(), path.stat().st_mtime_ns
        self.assertTrue(all(not item["updated"] for item in self.session.prepare_samples()))
        self.assertEqual(path.read_bytes(), draft_bytes)
        self.assertEqual(path.stat().st_mtime_ns, draft_time)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(self.session.store.load(self.voice_id), saved)
        self.assert_no_models()

    def test_legacy_binding_remains_unchanged_when_its_generic_draft_is_updated(self):
        saved = self.saved_voice(candidate=False)
        folder = self.session.store.folder(self.voice_id)
        before = {path: path.read_bytes() for path in folder.iterdir()}
        self.session.prepare_samples()
        self.assertEqual(self.session.rows()[0]["reference_text"], EXCERPT)
        self.assertEqual(self.session.store.load(self.voice_id), saved)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assert_no_models()

    def test_legacy_generic_draft_migrates_script_but_preserves_edited_prompt(self):
        path = self.legacy_draft()
        old_bytes = path.read_bytes()
        row = self.session.rows()[0]
        self.assertEqual(path.read_bytes(), old_bytes, "Browsing alone must not migrate files")
        self.assertEqual(row["prompt"], "The voice prompt I carefully edited.")
        self.assertEqual(row["reference_text"], EXCERPT)
        self.assertEqual(
            row["sample_source"]["replaces_reference_text"], self.config["voices"]["reference_text"])
        self.session.prepare_samples()
        saved = read_json(path)
        self.assertEqual(saved["version"], 2)
        self.assertEqual(saved["prompt"], row["prompt"])
        self.assertEqual(saved["reference_text"], EXCERPT)
        self.assertEqual(saved["sample_source"], row["sample_source"])
        self.assert_no_models()

    def test_legacy_custom_script_is_preserved_and_marked_as_user_text(self):
        path = self.legacy_draft(text="This is a sample I wrote specifically for Mara.")
        row = self.session.rows()[0]
        self.assertEqual(row["reference_text"], "This is a sample I wrote specifically for Mara.")
        self.assertEqual(row["sample_source"], {"kind": "user"})
        self.session.prepare_samples()
        record = read_json(path)
        self.assertEqual(record["version"], 2)
        self.assertEqual(record["reference_text"], row["reference_text"])
        self.assertEqual(record["sample_source"], {"kind": "user"})
        calls = [call for call in self.selector.call_args_list if call.args[1] == self.voice_id]
        self.assertEqual(calls, [])
        self.assert_no_models()

    def test_explicitly_entering_generic_script_preserves_that_user_choice(self):
        generic = self.config["voices"]["reference_text"]
        self.session.update_prompt(self.voice_id, reference_text=generic)
        later = self.new_session()
        row = later.rows()[0]
        self.assertEqual(row["reference_text"], generic)
        self.assertEqual(row["sample_source"], {"kind": "user"})
        later.prepare_samples()
        record = read_json(later._draft_path(self.voice_id))
        self.assertEqual(record["reference_text"], generic)
        self.assertEqual(record["sample_source"], {"kind": "user"})
        self.assert_no_models()

    def test_prompt_only_edit_retains_automatic_excerpt_and_its_provenance(self):
        before = self.session.rows()[0]
        updated = self.session.update_prompt(self.voice_id, prompt="Warm, expressive, and quiet.")
        self.assertEqual(updated["prompt"], "Warm, expressive, and quiet.")
        self.assertEqual(updated["reference_text"], before["reference_text"])
        self.assertEqual(updated["sample_source"], before["sample_source"])
        self.assertEqual(self.new_session().rows()[0]["sample_source"], before["sample_source"])
        self.assert_no_models()

    def test_shared_character_roles_use_one_persisted_draft_and_identical_excerpt(self):
        data = deepcopy(self.bundle)
        alias = "mara-narration-voice"
        data["cast"]["voices"].append(
            {"id": alias, "description": "Mara's narration voice.", "traits": []})
        data["cast"]["roles"][0]["voice_id"] = alias
        session = self.new_session(bundle=data)
        rows = {item["voice_id"]: item for item in session.rows()}
        self.assertEqual(rows[alias]["canonical_voice_id"], self.voice_id)
        self.assertEqual(rows[alias]["reference_text"], rows[self.voice_id]["reference_text"])
        self.assertEqual(rows[alias]["sample_source"], rows[self.voice_id]["sample_source"])
        self.assertEqual(self.selector.call_count, 2)
        session.prepare_samples()
        self.assertEqual(session._draft_path(alias), session._draft_path(self.voice_id))
        self.assertEqual(len(list((session.store.directory / "casting-drafts").glob("*.json"))), 2)
        session.update_prompt(alias, reference_text="A shared script edited through the narrator.")
        rows = {item["voice_id"]: item for item in session.rows()}
        self.assertEqual(rows[self.voice_id]["reference_text"], rows[alias]["reference_text"])
        self.assertEqual(rows[self.voice_id]["sample_source"], {"kind": "user"})
        self.assert_no_models()

    def test_selected_earlier_excerpt_is_preserved_in_a_later_book(self):
        saved = self.saved_voice(text=EXCERPT, source=self.sample["source"])
        next_book = deepcopy(self.bundle)
        next_book["source"]["book"]["id"] = "later-book"
        next_book["cast"]["voices"][0]["description"] = "A later description."
        session = self.new_session(bundle=next_book)
        row = session.rows()[0]
        self.assertEqual(row["reference_text"], EXCERPT)
        self.assertEqual(row["sample_source"], self.sample["source"])
        self.assertEqual(row["prompt"], saved["creation"]["requested_description"])
        session.prepare_samples()
        self.assertEqual(session.store.load(self.voice_id), saved)
        voice_calls = [call for call in self.selector.call_args_list if call.args[1] == self.voice_id]
        self.assertEqual(voice_calls, [])
        self.assert_no_models()

    def test_persisted_earlier_excerpt_draft_is_not_replaced_by_later_book_selection(self):
        self.session.prepare_samples()
        original = read_json(self.session._draft_path(self.voice_id))
        next_book = deepcopy(self.bundle)
        next_book["source"]["book"]["id"] = "later-book"
        self.selector.reset_mock()
        self.selector.side_effect = AssertionError("Established excerpts must survive later books")
        session = self.new_session(bundle=next_book)
        session.prepare_samples()
        self.assertEqual(read_json(session._draft_path(self.voice_id)), original)
        self.assert_no_models()

    def test_selected_user_script_is_preserved_in_a_later_book(self):
        generic = self.config["voices"]["reference_text"]
        saved = self.saved_voice(text=generic, source={"kind": "user"})
        next_book = deepcopy(self.bundle)
        next_book["source"]["book"]["id"] = "later-book"
        session = self.new_session(bundle=next_book)
        row = session.rows()[0]
        self.assertEqual(row["reference_text"], generic)
        self.assertEqual(row["sample_source"], {"kind": "user"})
        session.prepare_samples()
        self.assertEqual(session.store.load(self.voice_id), saved)
        voice_calls = [call for call in self.selector.call_args_list if call.args[1] == self.voice_id]
        self.assertEqual(voice_calls, [])
        self.assert_no_models()

    def test_old_short_automatic_excerpt_upgrades_without_changing_saved_reference(self):
        short = "Then someone was here."
        source = {"kind": "book_excerpt", "segment_ids": ["old-short-quote"]}
        saved = self.saved_voice(text=short, source=source)
        before = Path(saved["reference_audio"]).read_bytes()
        row = self.session.rows()[0]
        self.assertEqual(row["reference_text"], EXCERPT)
        self.assertEqual(row["sample_source"]["replaces_reference_text"], short)
        self.session.prepare_samples()
        self.assertEqual(self.session.store.load(self.voice_id)["reference_text"], short)
        self.assertEqual(Path(saved["reference_audio"]).read_bytes(), before)
        self.assert_no_models()

    def test_joined_and_creative_scripts_remain_stable_in_persisted_drafts(self):
        for kind in ("book_excerpts", "casting_script"):
            self.sample["source"]["kind"] = kind
            self.session._samples.clear()
            self.session.prepare_samples()
            original = read_json(self.session._draft_path(self.voice_id))
            self.assertEqual(original["sample_source"]["kind"], kind)
            later = self.new_session()
            with patch.object(later, "_sample_default", side_effect=AssertionError("Existing script replaced")):
                self.assertEqual(later.rows()[0]["reference_text"], EXCERPT)
            self.session._draft_path(self.voice_id).unlink()
            # The other character's draft is deliberately retained.
        self.assert_no_models()

    def test_unprepared_listing_does_not_fall_back_to_generic_generation(self):
        from epub_tts_narrator.renderer.samples import MissingSampleScript
        self.selector.side_effect = MissingSampleScript(self.voice_id)
        rows = self.session.rows()
        self.assertEqual(rows[0]["reference_text"], "")
        self.assertEqual(rows[0]["sample_source"]["kind"], "unprepared")
        with self.assertRaises(MissingSampleScript):
            self.session.generate(self.voice_id)
        self.assert_no_models()

    def test_fallback_can_acquire_an_excerpt_when_a_later_book_has_suitable_dialogue(self):
        self.selector.return_value = {
            "reference_text": self.config["voices"]["reference_text"],
            "source": {"kind": "fallback", "reason": "insufficient_character_dialogue"},
        }
        self.session.prepare_samples()
        self.assertEqual(read_json(self.session._draft_path(self.voice_id))["sample_source"]["kind"],
                         "fallback")
        self.selector.return_value = self.sample
        session = self.new_session()
        session.prepare_samples()
        record = read_json(session._draft_path(self.voice_id))
        self.assertEqual(record["reference_text"], EXCERPT)
        self.assertEqual(record["sample_source"]["kind"], "book_excerpt")
        self.assert_no_models()

    def test_version_two_source_is_covered_by_integrity_checksum(self):
        self.session.prepare_samples()
        path = self.session._draft_path(self.voice_id)
        record = read_json(path)
        record["sample_source"]["segment_ids"] = ["a-different-character"]
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.new_session().rows()
        self.assert_no_models()

    def test_unknown_version_two_source_kind_is_rejected_even_with_valid_checksum(self):
        self.session.prepare_samples()
        path = self.session._draft_path(self.voice_id)
        record = read_json(path)
        record["sample_source"]["kind"] = "invented"
        record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "sample source"):
            self.new_session().rows()
        self.assert_no_models()


if __name__ == "__main__":
    unittest.main()
