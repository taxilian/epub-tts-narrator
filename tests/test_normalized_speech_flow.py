"""Approved spelling reaches real render/cache gates without losing source text."""

from copy import deepcopy
from pathlib import Path
import tempfile
from threading import Lock
import unittest

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.performance import apply_bracket_delivery
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.workflow import _render_chunk
from epub_tts_narrator.speech_normalization import normalize_document_speech, normalization_pending
from test_contracts import valid_document
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_recovery_integration import RecoveryEngine, RecoveryVerifier
from test_renderer_workflow import FFMPEG, FFPROBE
from test_speech_normalization import Requests


def normalized_document():
    doc = valid_document()
    text = " Wheerrre issss sheeee?"
    doc["source"]["blocks"][1]["text"] = text
    doc["segments"][-1].update(text=text, start=0, end=len(text))
    doc = apply_bracket_delivery(doc)
    return normalize_document_speech(doc, Requests({"Wheerrre": "Where", "issss": "is", "sheeee": "she"}))


class NormalizedSpeechContractTests(unittest.TestCase):
    def test_reapplying_brackets_preserves_normalization_and_source(self):
        doc = normalized_document()
        revised = apply_bracket_delivery(doc)
        self.assertEqual(validate_document(revised), [])
        self.assertEqual(revised, doc)
        self.assertFalse(normalization_pending(revised))
        self.assertEqual(revised["segments"][-1]["spoken_text"], " Where is she?")

    def test_normalized_text_cannot_disguise_source_changes_or_forged_spoken_text(self):
        doc = normalized_document()
        self.assertEqual(validate_document(doc), [])
        changed = deepcopy(doc)
        changed["segments"][-1]["spoken_text"] = "Where she?"
        self.assertTrue(any("spoken_text differs" in error for error in validate_document(changed)))
        changed = deepcopy(doc)
        changed["segments"][-1]["text"] = " Where is she?"
        self.assertTrue(any("exactly match source" in error for error in validate_document(changed)))

    def test_real_failure_checks_normal_words_and_still_rejects_missing_words(self):
        row = normalized_document()["segments"][-1]
        # Raw legacy spellings may reach independent pronunciation review;
        # prepared analysis gives the verifier an exact lexical target.
        self.assertTrue(speech_coverage(row["text"], "Where is she?")["passed"])
        self.assertTrue(speech_coverage(row["spoken_text"], "Where is she?")["passed"])
        for heard in ("Where she?", "Where is?", "Where is he?", "Where is she now?"):
            with self.subTest(heard=heard):
                self.assertFalse(speech_coverage(row["spoken_text"], heard)["passed"])

    def test_saved_unapplied_styles_cannot_masquerade_as_directed_speech(self):
        voice = reference()
        request = speech_request(voice, "Where is she?")
        request["chunk"]["segment"]["delivery"]["direction"] = "Draw out the vowels."
        voices = {voice["asset_hash"]: voice}
        with self.assertRaisesRegex(ValueError, "does not confirm"):
            generation_identity(request, voices, engine_metadata={"synthesis_text": "Where is she?"})
        identity = generation_identity(request, voices, engine_metadata={
            "synthesis_text": "Where is she?", "applied_control": "style_prefix",
            "direction": "Draw out the vowels."})
        self.assertEqual(identity["text"], "[Draw out the vowels.] Where is she?")


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe needed for waveform/cache integration")
class NormalizedSpeechFlowTests(unittest.TestCase):
    def test_export_render_verify_and_reuse_normalized_passage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = normalized_document()
            target = root / "bundle"
            export_bundle(document, target)
            data = load_bundle(target)
            config = deepcopy(DEFAULTS)
            config["quality"] = validate_policy({})
            config["render"].update(max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE)
            plan = make_plan(data, config)
            chunk = next(item for chapter in plan["chapters"] for item in chapter["chunks"]
                         if item["segment"]["block_id"] == "b2")
            registry = {}
            engine, verifier = RecoveryEngine(registry), RecoveryVerifier(registry)
            voice = reference(chunk["segment"]["voice_id"])
            voices = {voice["voice_id"]: voice}
            work = root / "work"
            messages = []
            locks, guard = {}, Lock()

            def render(value):
                return _render_chunk(value, 1, 1, voices, engine, engine.identity,
                    verifier, config, work, locks, guard, notify=messages.append)

            rendered, accepted = render(chunk)
            self.assertTrue(accepted["report"]["passed"])
            self.assertEqual(engine.calls[0]["text"], "Where is she?")
            self.assertEqual(engine.calls[0]["direction"], "Sustain the vowels and sibilants naturally.")
            self.assertEqual(rendered["segment"]["text"], " Wheerrre issss sheeee?")
            self.assertEqual(rendered["source_ranges"], [{"start": 0, "end": 23}])
            moved = deepcopy(chunk)
            moved["chunk_id"] = "another-position"
            render(moved)
            self.assertEqual(len(engine.calls), 1, "Position changes must reuse the verified performance")
            changed_style = deepcopy(chunk)
            changed_style["segment"]["speech_normalization"]["edits"][0]["direction"] = "Speak with a hiss."
            render(changed_style)
            self.assertEqual(len(engine.calls), 2, "Changed audible instructions need a new performance")


if __name__ == "__main__":
    unittest.main()
