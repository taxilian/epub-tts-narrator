"""A model-selected acting change must produce new, fully verified speech."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.delivery_recovery import validate_delivery_marker
from epub_tts_narrator.renderer.engines import _directed_text
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.recurring_resolution import RecurringResolution
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget, SpeechVerificationFailed, _render_chunk
from test_renderer_recurring_resolution import PlanClient
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_recovery_integration import RecoveryVerifier, pcm
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


TEXT = "Arrgh! The pain has returned."
DIRECTION = "pained, breaking into a cry"


class DirectedFixtureEngine:
    identity = {"engine": "fish_http", "model": "offline-drama-3", "temperature": 1.0,
                "top_p": 0.9, "top_k": 30}

    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        self.omit_in_plain = False
        self.heard_override = None

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.calls.append({"text": text, "voice": deepcopy(voice), "direction": direction, "seed": seed})
        write_tone(path, frequency=220 + len(self.calls) * 31)
        if self.heard_override is not None:
            heard = self.heard_override(text, direction, len(self.calls))
        elif direction:
            heard = (text.replace("Arrgh", "Oh"), text.replace("Arrgh", ""),
                     text.replace("Arrgh", "Are"))[(len(self.calls) - 1) % 3]
        elif self.omit_in_plain:
            heard = text.replace("pain ", "")
        else:
            heard = text
        self.registry[hashlib.sha256(pcm(path)).hexdigest()] = heard
        directed, metadata = _directed_text(text, direction)
        return {"fixture": True, "synthesis_text": directed, **metadata}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for plain-delivery integration")
class PlainDeliveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["quality"] = validate_policy({})
        self.config["render"].update(max_attempts=3, sample_rate=44100, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.registry = {}
        self.engine = DirectedFixtureEngine(self.registry)
        self.base = RecoveryVerifier(self.registry)
        self.client = PlanClient([{"action": "plain_delivery", "edits": [],
                                   "reason": "The acting direction may be obscuring the words; try plain speech."}])
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.voice = reference("character")
        voice_path = self.work / "reference.wav"
        write_tone(voice_path)
        self.voice.update(reference_audio=str(voice_path), reference_sha256=file_hash(voice_path))
        self.chunk = speech_request(self.voice, TEXT)["chunk"]
        self.chunk["segment"]["delivery"]["direction"] = DIRECTION
        self.messages = []

    def render_chunk(self, budget=None):
        return _render_chunk(self.chunk, 1, 1, {self.voice["voice_id"]: self.voice},
                             self.engine, self.engine.identity, self.verifier, self.config,
                             self.work, {}, Lock(), notify=self.messages.append,
                             retry_budget=budget or SpeechRetryBudget())

    def accepted_record(self, result):
        return read_json(Path(result["wav_path"]).parent / "accepted.json")

    def test_varied_same_span_failures_choose_plain_then_verify_original_and_reuse_cache(self):
        original = deepcopy(self.chunk)
        budget = SpeechRetryBudget()
        result, accepted = self.render_chunk(budget)
        self.assertEqual(self.chunk, original)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["expected_text"], TEXT)
        self.assertEqual(accepted["report"]["transcript"], TEXT)
        self.assertEqual(accepted["report"]["metrics"]["wer"], 0)
        self.assertNotEqual(accepted["report"].get("text_confirmation_method"), "recurring_resolution",
                            "Choosing a performance cannot accept the old failed audio")
        self.assertEqual([call["direction"] for call in self.engine.calls], [DIRECTION] * 3 + [""])
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 4)
        self.assertTrue(all(call["voice"] == self.voice for call in self.engine.calls))
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(self.client.reviews, [])
        self.assertIn("plain_delivery", self.client.plans[0]["permitted_actions"])
        self.assertEqual([row["attempt"] for row in self.client.plans[0]["failures"]], [1, 2, 3])
        record = self.accepted_record(result)
        marker = record["engine_metadata"]["delivery_recovery"]
        self.assertEqual(marker["original_direction"], DIRECTION)
        self.assertEqual(marker["replacement_direction"], "")
        plain = validate_delivery_marker(record["request"], marker)
        self.assertEqual(plain["chunk"]["spoken_text"], TEXT)
        self.assertEqual(plain["voice_asset_hash"], self.voice["asset_hash"])
        assets = {self.voice["asset_hash"]: self.voice}
        self.assertEqual(generation_identity(record["request"], assets),
                         generation_identity(record["request"], assets, engine_metadata=record["engine_metadata"]))
        before = len(self.engine.calls), len(self.base.calls), len(self.client.plans)
        self.assertEqual(self.render_chunk(budget), (result, accepted))
        self.assertEqual((len(self.engine.calls), len(self.base.calls), len(self.client.plans)), before)

    def test_plain_delivery_still_rejects_missing_complete_words(self):
        self.config["verification"]["max_resolution_attempts"] = 4
        self.engine.omit_in_plain = True
        with self.assertRaises(SpeechVerificationFailed) as failure:
            self.render_chunk()
        self.assertEqual([call["direction"] for call in self.engine.calls], [DIRECTION] * 3 + [""])
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 4)
        self.assertEqual(failure.exception.report["reasons"], ["text_mismatch"])
        self.assertFalse(list((self.work / "clips").glob("*/accepted.json")))
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))
        record = next((self.work / "clips").glob("*/attempt-004.json"))
        self.assertIn("delivery_recovery", read_json(record)["engine_metadata"])

    def test_persisted_plain_decision_resumes_without_repeating_llm_or_directed_takes(self):
        save = RecurringResolution._save
        interrupted = False

        def interrupt_after_decision(session):
            nonlocal interrupted
            save(session)
            if (not interrupted and session.state["history"]
                    and session.state["history"][-1]["audit"].get("proposal", {}).get("action") == "plain_delivery"):
                interrupted = True
                raise KeyboardInterrupt("Fixture crash after the decision was saved")

        with patch.object(RecurringResolution, "_save", interrupt_after_decision):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        self.assertTrue(interrupted)
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.plans), 1)
        state_path = next((self.work / "clips").glob("*/speech-resolution.json"))
        prior_history = read_json(state_path)["history"]
        result, accepted = self.render_chunk()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual([call["direction"] for call in self.engine.calls], [DIRECTION] * 3 + [""])
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(read_json(state_path)["history"], prior_history)
        self.assertIn("delivery_recovery", self.accepted_record(result)["engine_metadata"])

    def test_plain_decision_at_whole_strategy_boundary_gets_one_real_test(self):
        text = "Arrgh the pain has returned. Everyone waited outside."
        self.chunk["spoken_text"] = text
        self.chunk["spoken_end"] = len(text)
        self.chunk["source_ranges"] = [{"start": 0, "end": len(text)}]
        self.config["verification"]["max_resolution_attempts"] = 6
        result, accepted = self.render_chunk()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual([call["text"] for call in self.engine.calls], [text] * 4)
        self.assertEqual([call["direction"] for call in self.engine.calls], [DIRECTION] * 3 + [""])
        self.assertEqual(len(self.client.plans), 1)
        self.assertFalse(list((self.work / "clips").glob("*/speech-recovery.json")),
                         "A valid new recovery proposal must be tried before splitting the passage")
        self.assertIn("delivery_recovery", self.accepted_record(result)["engine_metadata"])

    def test_later_review_cannot_select_already_active_plain_delivery_again(self):
        self.config["verification"]["max_resolution_attempts"] = 7
        self.engine.omit_in_plain = True
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual(len(self.client.plans), 2)
        self.assertIn("plain_delivery", self.client.plans[0]["permitted_actions"])
        self.assertNotIn("plain_delivery", self.client.plans[1]["permitted_actions"])
        self.assertEqual([call["direction"] for call in self.engine.calls], [DIRECTION] * 3 + [""] * 4)
        state = read_json(next((self.work / "clips").glob("*/speech-resolution.json")))
        self.assertEqual([row["audit"]["decision"] for row in state["history"]], ["resolved", "blocked"])
        self.assertFalse(list((self.work / "clips").glob("*/accepted.json")))

    def test_saved_rewrite_and_plain_decisions_resume_together_with_both_audits(self):
        text = "Insang felt the pain."
        alternate = "Insaan felt the pain."
        self.chunk["spoken_text"] = text
        self.chunk["spoken_end"] = len(text)
        self.chunk["source_ranges"] = [{"start": 0, "end": len(text)}]
        self.client.actions = [
            {"action": "rewrite", "edits": [{"start": 0, "end": 6, "source": "Insang",
                                                 "replacement": "Insaan"}],
             "reason": "Try a pronunciation spelling."},
            {"action": "plain_delivery", "edits": [], "reason": "The acting still obscures a word."},
        ]
        self.engine.heard_override = lambda given, direction, number: (
            given.replace("Insang", "Insane").replace("felt ", "") if direction else given)
        save = RecurringResolution._save

        def interrupt_after_plain(session):
            save(session)
            if (session.state["history"]
                    and session.state["history"][-1]["audit"].get("proposal", {}).get("action") == "plain_delivery"):
                raise KeyboardInterrupt("Fixture crash after saving both repairs")

        with patch.object(RecurringResolution, "_save", interrupt_after_plain):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        self.assertEqual([call["text"] for call in self.engine.calls], [text] * 3 + [alternate] * 3)
        self.assertEqual(len(self.client.plans), 2)
        self.assertEqual(len(self.client.reviews), 3)
        result, accepted = self.render_chunk()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["expected_text"], text)
        self.assertEqual(accepted["report"]["transcript"], alternate)
        self.assertEqual(len(self.client.plans), 2)
        self.assertEqual(len(self.engine.calls), 7)
        self.assertEqual(self.engine.calls[-1]["text"], alternate)
        self.assertEqual(self.engine.calls[-1]["direction"], "")
        record = self.accepted_record(result)
        metadata = record["engine_metadata"]
        self.assertIn("delivery_recovery", metadata)
        self.assertIn("speech_resolution", metadata)
        self.assertEqual(metadata["synthesis_text"], alternate)
        self.assertEqual(metadata["speech_resolution"]["original_text"], text)
        self.assertEqual(metadata["delivery_recovery"]["original_text"], text)
        self.assertEqual(metadata["delivery_recovery"]["original_direction"], DIRECTION)
        self.assertEqual(self.render_chunk(), (result, accepted))
        self.assertEqual(len(self.engine.calls), 7)


if __name__ == "__main__":
    unittest.main()
