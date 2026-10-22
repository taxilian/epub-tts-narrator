"""Real PCM/cache integration for recurring recovery, without models or APIs."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.common import file_hash, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.recurring_resolution import RecurringResolution, failure_signatures
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, SpeechVerificationFailed, VerificationUnavailable,
    _render_chunk, verified_take,
)
import epub_tts_narrator.renderer.workflow as workflow
from test_renderer_speech_recovery_integration import RecoveryEngine, RecoveryVerifier, pcm
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


TEXT = "Insang waited beside the old gate while everyone watched the river."
HEARD = TEXT.replace("Insang", "Insane")


class PlanClient:
    identity = {"model": "offline-resolution-fixture", "base_url": "https://fixture.invalid/v1"}

    def __init__(self, actions=None):
        self.actions = list(actions or [{"action": "accept", "edits": [], "reason": "Plausible sound spelling."}])
        self.plans, self.reviews = [], []
        self.unavailable = False

    def complete(self, prompt, payload):
        if "pairs" in payload:
            self.reviews.append(deepcopy(payload))
            return {"pairs": [{"id": pair["id"], "plausible": True,
                                "reason": "Plausible pronunciation spelling."} for pair in payload["pairs"]]}
        self.plans.append(deepcopy(payload))
        if self.unavailable:
            raise TimeoutError("Fixture model unavailable")
        position = min(len(self.plans) - 1, len(self.actions) - 1)
        return deepcopy(self.actions[position])


class ResolutionEngine(RecoveryEngine):
    identity = {"engine": "recurring-resolution-fixture", "model": "offline-v1"}

    def __init__(self, registry):
        super().__init__(registry)
        self.heard = lambda text, number: text.replace("Insang", "Insane")
        self.same_audio = False

    def _write(self, text, path, seed):
        if self.silent:
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(88200))
        else:
            write_tone(path, frequency=220 if self.same_audio else 220 + len(self.calls) * 31)
        self.registry[hashlib.sha256(pcm(path)).hexdigest()] = self.heard(text, len(self.calls))
        return {"fixture": True, "seed": seed, "synthesis_text": text}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for recurring recovery integration")
class RecurringResolutionIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["quality"] = validate_policy({})
        self.config["render"].update(max_attempts=3, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.registry = {}
        self.engine = ResolutionEngine(self.registry)
        self.base = RecoveryVerifier(self.registry)
        self.client = PlanClient()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.voice = {"asset_hash": "fixture-reference-asset", "kind": "reference",
                      "reference_text": "Fixture reference voice.", "reference_sha256": "1" * 64}
        self.text = TEXT
        self.request = {"kind": "speech", "text": self.text, "engine": self.engine.identity}
        self.chunk = {"chunk_id": "a1" * 16, "spoken_text": self.text,
                      "silent_control": False,
                      "source_ranges": [{"block_id": "b000123", "start": 7, "end": 7 + len(TEXT)}],
                      "segment": {"id": "segment-2", "role_id": "insang", "voice_id": "voice-1",
                                  "delivery": {"direction": "startled"}}}
        self.messages = []

    def take(self, budget=None):
        def generate(path, seed):
            return self.engine.synthesize(self.text, self.voice, path, seed=seed)
        def repaired(text, path, seed):
            return self.engine.synthesize(text, self.voice, path, seed=seed)
        return verified_take(self.text, generate, self.work / "takes", self.request,
                             self.verifier, self.config, notify=self.messages.append,
                             retry_budget=budget or SpeechRetryBudget(), generate_repaired=repaired)

    def render_chunk(self, budget=None):
        return _render_chunk(self.chunk, 1, 1, {"voice-1": self.voice}, self.engine,
                             self.engine.identity, self.verifier, self.config, self.work,
                             {}, Lock(), notify=self.messages.append,
                             retry_budget=budget or SpeechRetryBudget())

    @staticmethod
    def rewrite(replacement):
        return {"action": "rewrite", "reason": "Try a pronunciation spelling.",
                "edits": [{"start": 0, "end": 6, "source": "Insang", "replacement": replacement}]}

    def test_nonconsecutive_a_b_a_c_a_resolves_on_fifth_take(self):
        variants = [HEARD, TEXT.replace("old ", ""), HEARD, TEXT + " Today.", HEARD]
        self.engine.heard = lambda text, number: variants[number - 1]
        wav, accepted = self.take()
        self.assertEqual(len(self.engine.calls), 5)
        self.assertEqual(wav.name, "attempt-005.wav")
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(len(self.client.reviews), 3)
        self.assertEqual([item["attempt"] for item in self.client.plans[0]["failures"]], [1, 2, 3, 4, 5])
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["transcript"], HEARD)
        self.assertEqual(accepted["report"]["expected_text"], TEXT)
        self.assertGreater(accepted["report"]["metrics"]["wer"], 0)
        self.assertEqual(accepted["report"]["previous_reasons"], ["text_mismatch"])
        state = read_json(self.work / "takes" / "speech-resolution.json")
        self.assertEqual(list(state["history"][0]["trigger_counts"].values()), [3])

    def test_distinct_generations_with_identical_audio_count_once_per_take(self):
        self.engine.same_audio = True
        _, accepted = self.take()
        failures = self.client.plans[0]["failures"]
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual([failure["attempt"] for failure in failures], [1, 2, 3])
        self.assertEqual(len({failure["audio_sha256"] for failure in failures}), 1)
        report = {"passed": False, "reasons": ["text_mismatch"], "transcript": HEARD,
                  "secondary_transcript": HEARD, "corroboration": {"transcript": HEARD}}
        self.assertEqual(len(failure_signatures(TEXT, report)), 1,
                         "Repeated ASR evidence for one waveform is one occurrence")
        self.assertEqual(accepted["attempt"], 3)

    def test_rewrite_changes_only_tts_input_and_result_reuses_normal_cache(self):
        self.client.actions = [self.rewrite("Insaan")]
        before = deepcopy(self.chunk)
        result, acceptance = self.render_chunk()
        alternate = TEXT.replace("Insang", "Insaan")
        self.assertEqual([row["text"] for row in self.engine.calls], [TEXT] * 3 + [alternate])
        self.assertEqual(self.chunk, before)
        self.assertEqual(result, {**before, "wav_path": result["wav_path"]})
        self.assertTrue(acceptance["report"]["passed"])
        self.assertEqual(acceptance["report"]["expected_text"], TEXT)
        self.assertEqual(acceptance["report"]["transcript"], alternate)
        self.assertEqual(acceptance["report"]["text_confirmation_method"], "pronunciation_rewrite")
        saved = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertEqual(saved["request"]["chunk"], before)
        self.assertEqual(saved["engine_metadata"]["synthesis_text"], alternate)
        self.assertEqual(saved["engine_metadata"]["speech_resolution"]["original_text"], TEXT)
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(len(self.client.reviews), 3)
        counts = len(self.engine.calls), len(self.base.calls), len(self.client.plans)
        repeated = self.render_chunk()
        self.assertEqual(repeated, (result, acceptance))
        self.assertEqual((len(self.engine.calls), len(self.base.calls), len(self.client.plans)), counts)
        for call in self.engine.calls:
            self.assertEqual(call["voice"], self.voice)
            self.assertEqual(call["direction"], "startled")

    def test_pending_resolver_outage_resumes_saved_candidate_without_new_synthesis(self):
        self.client.unavailable = True
        with self.assertRaises(VerificationUnavailable):
            self.take()
        self.assertEqual(len(self.engine.calls), 3)
        pending = read_json(self.work / "takes" / "attempt-003.json")
        self.assertTrue(pending["report"]["resolution_pending"])
        wav = self.work / "takes" / "attempt-003.wav"
        original = wav.read_bytes()
        old_asr_calls = len(self.base.calls)
        self.client.unavailable = False
        resumed, accepted = self.take()
        self.assertEqual(resumed, wav)
        self.assertEqual(wav.read_bytes(), original)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.engine.calls), 3)
        self.assertGreater(len(self.base.calls), old_asr_calls)
        self.assertEqual(len(self.client.plans), 2)
        self.assertEqual([row["attempt"] for row in self.client.plans[-1]["failures"]], [1, 2, 3])

    def test_interrupted_rewritten_take_keeps_metadata_and_resumes_without_new_tts(self):
        self.client.actions = [self.rewrite("Insaan")]
        validate_audio = workflow.validate_audio

        def interrupt_fourth_take(path, *args, **kwargs):
            if len(self.engine.calls) == 4:
                raise KeyboardInterrupt("Interrupted after rewritten audio was normalized")
            return validate_audio(path, *args, **kwargs)

        with patch.object(workflow, "validate_audio", side_effect=interrupt_fourth_take):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        self.assertEqual([row["text"] for row in self.engine.calls],
                         [TEXT] * 3 + [TEXT.replace("Insang", "Insaan")])
        pending_path = next((self.work / "clips").glob("*/attempt-004.json"))
        pending = read_json(pending_path)
        wav = pending_path.parent / pending["audio_file"]
        self.assertEqual(pending["audio_sha256"], file_hash(wav))
        self.assertTrue(pending["report"]["retry_on_resume"])
        self.assertEqual(pending["engine_metadata"]["synthesis_text"], TEXT.replace("Insang", "Insaan"))
        self.assertEqual(pending["engine_metadata"]["speech_resolution"]["original_text"], TEXT)
        before = wav.read_bytes()
        result, acceptance = self.render_chunk()
        self.assertEqual(result["wav_path"], str(wav))
        self.assertEqual(wav.read_bytes(), before)
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(acceptance["report"]["text_confirmation_method"], "pronunciation_rewrite")
        self.assertEqual(acceptance["report"]["expected_text"], TEXT)
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual(len(self.engine.calls), 4)

    def interrupt_after_saved_decision(self, action):
        save = RecurringResolution._save

        def save_then_interrupt(session):
            save(session)
            if session.state["history"][-1]["audit"]["proposal"]["action"] == action:
                raise KeyboardInterrupt("Decision saved before its result was published")

        with patch.object(RecurringResolution, "_save", save_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()

    def test_saved_accept_decision_replays_after_interrupted_publication(self):
        self.interrupt_after_saved_decision("accept")
        self.assertEqual(len(self.engine.calls), 3)
        self.assertFalse(list((self.work / "clips").glob("*/accepted.json")))
        state_path = next((self.work / "clips").glob("*/speech-resolution.json"))
        saved_state = state_path.read_bytes()
        result, acceptance = self.render_chunk()
        self.assertEqual(Path(result["wav_path"]).name, "attempt-003.wav")
        self.assertTrue(acceptance["report"]["passed"])
        self.assertEqual(acceptance["report"]["text_confirmation_method"], "recurring_resolution")
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(len(self.client.reviews), 3)
        self.assertEqual(state_path.read_bytes(), saved_state,
                         "Replay must not increment counts, spend a model round, or replace its audit")
        self.assertTrue(any("Resuming saved speech recovery decision: accept" in row for row in self.messages))

    def test_saved_accept_is_not_replayed_when_fresh_transcription_changes(self):
        self.interrupt_after_saved_decision("accept")
        candidate = next((self.work / "clips").glob("*/attempt-003.wav"))
        # The waveform is unchanged, but independent recognition now reports a
        # missing complete word. The old decision must not approve that result.
        self.registry[hashlib.sha256(pcm(candidate)).hexdigest()] = HEARD.replace("old ", "")
        self.engine.heard = lambda text, number: text
        result, acceptance = self.render_chunk()
        self.assertEqual(Path(result["wav_path"]).name, "attempt-004.wav")
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(acceptance["report"]["metrics"]["wer"], 0)
        self.assertNotEqual(acceptance["report"].get("text_confirmation_method"), "recurring_resolution")

    def test_saved_split_decision_replays_before_spending_more_whole_takes(self):
        original = self.configure_split_rewrite()
        self.interrupt_after_saved_decision("split")
        self.assertEqual(len(self.engine.calls), 3)
        self.assertFalse(list((self.work / "clips").glob("*/speech-recovery.json")))
        state_path = next((self.work / "clips").glob("*/speech-resolution.json"))
        saved_state = state_path.read_bytes()
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance["report"]["passed"])
        self.assertEqual(acceptance["report"]["expected_text"], original)
        self.assertEqual(len(self.engine.calls), 8)
        self.assertEqual([row["text"] for row in self.engine.calls[:3]], [original] * 3)
        self.assertTrue(all(row["text"] != original for row in self.engine.calls[3:]))
        self.assertEqual(len(self.client.plans), 2, "Only the child rewrite should need a new model call")
        self.assertEqual(state_path.read_bytes(), saved_state)
        self.assertTrue(any("Resuming saved speech recovery decision: split" in row for row in self.messages))
        self.assertTrue(Path(result["wav_path"]).is_file())

    def test_rewrites_and_split_parts_share_the_twelve_generation_budget(self):
        text = "Insang waited beside the old gate. Everyone watched the river."
        self.chunk["spoken_text"] = text
        self.chunk["source_ranges"][0]["end"] = 7 + len(text)
        self.config["verification"]["max_resolution_rounds"] = 5
        self.client.actions = [self.rewrite("Insaan"), self.rewrite("Eensang"),
                               {"action": "split", "edits": [], "reason": "Try complete short sentences."},
                               {"action": "retry", "edits": [], "reason": "Try a fresh take."}]
        self.engine.heard = lambda given, number: given.replace("old ", "")
        budget = SpeechRetryBudget()
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk(budget)
        self.assertEqual(len(self.engine.calls), 12)
        self.assertEqual([row["text"] for row in self.engine.calls[:9]],
                         [text] * 3 + [text.replace("Insang", "Insaan")] * 3
                         + [text.replace("Insang", "Eensang")] * 3)
        self.assertEqual([row["text"] for row in self.engine.calls[9:]],
                         ["Insang waited beside the old gate."] * 3)
        self.assertTrue(list((self.work / "clips").glob("*/speech-recovery.json")))
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk(budget)
        self.assertEqual(len(self.engine.calls), 12, "Reentering a split plan cannot reset its parent budget")

    def test_omission_cannot_be_accepted_even_if_planner_and_sound_judges_say_yes(self):
        self.config["verification"]["max_resolution_attempts"] = 6
        self.engine.heard = lambda text, number: text.replace("old ", "")
        with self.assertRaises(SpeechVerificationFailed):
            self.take()
        self.assertEqual(len(self.engine.calls), 6)
        self.assertEqual(len(self.client.plans), 2)
        self.assertEqual(self.client.reviews, [], "A missing word must never reach subjective acceptance")
        self.assertFalse((self.work / "takes" / "accepted.json").exists())

    def test_acoustic_failure_cannot_be_accepted_by_model(self):
        self.config["verification"]["max_resolution_attempts"] = 6
        self.engine.silent = True
        with self.assertRaises(SpeechVerificationFailed) as error:
            self.take()
        self.assertIn("silent_audio", error.exception.report["reasons"])
        self.assertEqual(len(self.engine.calls), 6)
        self.assertEqual(len(self.client.plans), 2)
        self.assertTrue(all("accept" not in row["permitted_actions"] for row in self.client.plans))
        self.assertEqual(self.client.reviews, [])
        self.assertEqual(self.base.calls, [])
        self.assertFalse((self.work / "takes" / "accepted.json").exists())

    def configure_split_rewrite(self):
        text = "Insang waited beside the old gate. Everyone watched the river."
        self.chunk["spoken_text"] = text
        self.chunk["source_ranges"][0]["end"] = 7 + len(text)
        self.client.actions = [
            {"action": "split", "edits": [], "reason": "Try shorter complete sentences."},
            self.rewrite("Insaan"),
        ]
        return text

    def test_split_assembly_carries_verified_child_pronunciation_to_original_text(self):
        original = self.configure_split_rewrite()
        before = deepcopy(self.chunk)
        result, acceptance = self.render_chunk()
        self.assertEqual(self.chunk, before)
        self.assertEqual(len(self.engine.calls), 8)
        report = acceptance["report"]
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["expected_text"], original)
        self.assertEqual(report["transcript"], original.replace("Insang", "Insaan"))
        self.assertEqual(report["text_confirmation_method"], "verified_recovery_parts")
        self.assertEqual(report["metrics"]["duration_seconds"], 2)
        record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertEqual(record["request"]["chunk"], before)
        proof = record["engine_metadata"]["recovery"]["parts"]
        self.assertEqual(len(proof), 2)
        children = []
        for part in proof:
            child_file = Path(result["wav_path"]).parent / part["audio_file"]
            accepted_path = child_file.parent / "accepted.json"
            self.assertEqual(part["accepted_sha256"], file_hash(accepted_path))
            children.append(read_json(accepted_path))
        self.assertEqual(children[0]["report"]["text_confirmation_method"], "pronunciation_rewrite")
        self.assertEqual(children[0]["report"]["expected_text"], "Insang waited beside the old gate. ")
        self.assertEqual(children[0]["report"]["transcript"], "Insaan waited beside the old gate.")
        self.assertEqual(children[1]["report"]["metrics"]["wer"], 0)
        counts = len(self.engine.calls), len(self.base.calls), len(self.client.plans)
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual((len(self.engine.calls), len(self.base.calls), len(self.client.plans)), counts)

    def test_split_assembly_with_extra_words_fails_despite_accepted_pronunciation_parts(self):
        original = self.configure_split_rewrite()
        self.base.fail_assembly = True
        with self.assertRaises(SpeechVerificationFailed) as error:
            self.render_chunk()
        self.assertEqual(error.exception.report["expected_text"], original)
        self.assertTrue(error.exception.report["transcript"].endswith("Thank you."))
        self.assertEqual(error.exception.report["reasons"], ["text_mismatch"])
        self.assertEqual(len(self.engine.calls), 8)
        self.assertEqual(len(list((self.work / "clips").glob("*/parts/*/accepted.json"))), 2)
        self.assertFalse(list((self.work / "clips").glob("*/accepted.json")))
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))

    def test_split_assembly_cannot_use_changed_child_acceptance_record(self):
        self.configure_split_rewrite()
        transcribe = self.base.transcribe
        changed = []

        def tamper_after_assembly(path):
            result = transcribe(path)
            with wave.open(str(path), "rb") as audio:
                assembled = audio.getnframes() > 44100
            if assembled and not changed:
                child = next(Path(path).parent.glob("parts/*/accepted.json"))
                record = read_json(child)
                record["diagnostic"] = "Altered after assembly evidence was bound"
                write_json(child, record)
                changed.append(child)
            return result

        self.base.transcribe = tamper_after_assembly
        with self.assertRaises(SpeechVerificationFailed) as error:
            self.render_chunk()
        self.assertEqual(len(changed), 1)
        self.assertEqual(error.exception.report["reasons"], ["text_mismatch"])
        self.assertEqual(len(self.engine.calls), 8)
        self.assertFalse(list((self.work / "clips").glob("*/accepted.json")))
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
