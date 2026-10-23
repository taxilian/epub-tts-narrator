"""A standalone chapter worker can recover one uniform paragraph without replanning."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
from epub_tts_narrator.renderer.chapter_client import _validate_timing_sources
from epub_tts_narrator.renderer.common import digest, file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.engines import format_dialogue
from epub_tts_narrator.renderer.paragraphs import _group
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
from test_renderer_recurring_resolution import PlanClient
from test_renderer_speech_identity import reference
from test_renderer_speech_recovery_integration import RecoveryEngine, RecoveryVerifier, TEXT, FIRST, LAST
from test_renderer_workflow import FFMPEG, FFPROBE, timed_transcript, write_tone


class UniformEngine(RecoveryEngine):
    identity = {"engine": "fish_http", "model": "offline-paragraph-recovery"}
    capabilities = {"multi_speaker": True}

    def synthesize_dialogue(self, turns, path, *, seed):
        self.dialogue_calls.append(deepcopy(turns))
        metadata = self._write("".join(turn["text"] for turn in turns), path, seed)
        formatted, _, formatting = format_dialogue(turns)
        return {**metadata, "synthesis_text": formatted, **formatting}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class UniformParagraphRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["quality"] = validate_policy({})
        self.config["render"].update(max_attempts=3, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        registry = {}
        self.engine, self.verifier = UniformEngine(registry), RecoveryVerifier(registry)
        self.voice = reference("voice")
        reference_path = self.work / "reference.wav"
        write_tone(reference_path)
        self.voice.update(reference_audio=str(reference_path), reference_sha256=file_hash(reference_path))
        turns = []
        boundary = TEXT.index(" What") + 1
        for index, (start, end) in enumerate(((0, boundary), (boundary, len(TEXT)))):
            text = TEXT[start:end]
            row = {"id": f"segment-{index}", "chapter_id": "chapter-0001", "block_id": "b000001",
                   "start": start, "end": end, "text": text, "spoken_text": text,
                   "role_id": "character", "voice_id": "voice", "delivery": {"direction": "", "evidence": []}}
            turns.append({"chunk_id": digest(index)[:32], "spoken_text": text, "spoken_start": 0,
                          "spoken_end": len(text), "silent_control": False, "segment": row,
                          "source_ranges": [{"start": start, "end": end}]})
        self.chunk = _group(turns)[0]
        self.chapter = {"id": "chapter-0001", "number": 1, "title": "Fixture", "chunks": [self.chunk]}
        self.executor = ChapterExecutor(self.config, engine=self.engine, verifier=self.verifier)
        self.addCleanup(self.executor.close)
        self.spec = {"version": 1, "book": {"id": "fixture", "title": "Fixture"},
                     "source_fingerprint": digest("source"), "chapter": self.chapter, "total_chapters": 1,
                     "voice_bindings": {"voice": self.voice["asset_hash"]}, "render": {},
                     "renderer_fingerprint": digest(self.executor.identity)}
        self.messages = []
        self.voices = {"voice": self.voice}

    def render(self):
        return self.executor.synthesize(self.spec, self.voices, self.work, self.messages.append)

    def adaptive(self, actions):
        client = PlanClient(actions)
        self.executor._verifier = RecurringSpeechVerifier(self.verifier, SpeechResolver(client))
        return client

    def test_remote_failure_splits_preserves_one_parent_assembles_and_resumes(self):
        original = deepcopy(self.spec)
        prepared = self.render()
        self.assertEqual(self.spec, original)
        self.assertEqual(len(self.engine.dialogue_calls), 3)
        self.assertEqual([call["text"] for call in self.engine.calls], [FIRST, LAST])
        self.assertEqual(len(prepared["clips"]), 1)
        self.assertEqual(len(prepared["acceptance"]), 1)
        clip = prepared["clips"][0]
        self.assertEqual(clip, dict(self.chunk, wav_path=clip["wav_path"]))
        accepted = read_json(Path(clip["wav_path"]).parent / "accepted.json")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["expected_text"], TEXT)
        self.assertEqual(accepted["report"]["transcript"], TEXT)
        audit = accepted["engine_metadata"]["recovery"]
        self.assertEqual("".join(part["text"] for part in audit["parts"]), TEXT)
        assets = {self.voice["asset_hash"]: self.voice}
        identity = generation_identity(accepted["request"], assets)
        self.assertEqual(generation_identity(accepted["request"], assets,
                                            engine_metadata=accepted["engine_metadata"]), identity)
        for part in audit["parts"]:
            child = part["acceptance"]
            path = Path(clip["wav_path"]).parent / part["audio_file"]
            self.assertEqual(file_hash(path), child["audio_sha256"])
            self.assertEqual(file_hash(path.parent / "accepted.json"), part["accepted_sha256"])
            self.assertEqual(child["request"]["synthesis_request"]["voice_asset_hash"], self.voice["asset_hash"])
        artifacts = self.executor.assemble(self.spec, prepared, self.work, self.messages.append)
        self.assertIn("chapter.m4a", artifacts)
        self.assertIn("chapter.flac", artifacts)
        metadata = read_json(artifacts["chapter.json"])
        _validate_timing_sources(metadata, self.chapter["chunks"])
        self.assertEqual(metadata["segments"][0]["precision"], "paragraph")
        before = len(self.engine.calls), len(self.engine.dialogue_calls), len(self.verifier.calls)
        self.assertEqual(self.render(), prepared)
        self.assertEqual(self.executor.assemble(self.spec, prepared, self.work, self.messages.append), artifacts)
        self.assertEqual((len(self.engine.calls), len(self.engine.dialogue_calls), len(self.verifier.calls)), before)

    def test_bad_full_join_is_not_cached(self):
        self.verifier.fail_assembly = True
        with self.assertRaisesRegex(SpeechVerificationFailed, "final full-text verification"):
            self.render()
        self.assertEqual(len(self.engine.calls), 2)
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))

    def test_cold_index_rebuild_reuses_verified_parent_without_synthesis(self):
        prepared = self.render()
        for path in (self.work / "speech-cache").glob("*.json"):
            path.unlink()
        before = len(self.engine.calls), len(self.engine.dialogue_calls), len(self.verifier.calls)
        self.assertEqual(self.render(), prepared)
        self.assertEqual((len(self.engine.calls), len(self.engine.dialogue_calls), len(self.verifier.calls)), before)

    def test_interrupted_child_resumes_first_part_and_advances_failed_part_seed(self):
        self.engine.interrupt_part = LAST
        with self.assertRaises(KeyboardInterrupt):
            self.render()
        saved = {path: file_hash(path) for path in (self.work / "clips").glob("*/parts/*/accepted.json")}
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(self.engine.dialogue_calls), 3)
        self.engine.interrupt_part = None
        prepared = self.render()
        self.assertTrue(prepared["acceptance"][0]["report"]["passed"])
        self.assertEqual(len(self.engine.dialogue_calls), 3)
        self.assertEqual([call["text"] for call in self.engine.calls], [FIRST, LAST, LAST])
        self.assertNotEqual(self.engine.calls[1]["seed"], self.engine.calls[2]["seed"])
        self.assertTrue(all(file_hash(path) == checksum for path, checksum in saved.items()))

    def test_ordinary_accepted_paragraph_keeps_identity_and_never_splits(self):
        with patch.object(self.verifier, "transcribe", return_value=timed_transcript(TEXT)):
            prepared = self.render()
        accepted = read_json(Path(prepared["clips"][0]["wav_path"]).parent / "accepted.json")
        self.assertNotIn("recovery", accepted["request"])
        self.assertNotIn("recovery", accepted["engine_metadata"])
        self.assertEqual(len(self.engine.dialogue_calls), 1)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.render(), prepared)
        self.assertEqual(len(self.engine.dialogue_calls), 1)
        self.assertEqual(self.engine.calls, [])

    def test_mixed_voice_or_style_paragraph_cannot_use_uniform_recovery(self):
        for change in ("style", "voice"):
            with self.subTest(change=change):
                changed = deepcopy(self.spec)
                last = changed["chapter"]["chunks"][0]["paragraph_turns"][-1]
                voices = dict(self.voices)
                if change == "style":
                    last["segment"]["delivery"]["direction"] = "whisper"
                else:
                    # Even identical PCM under distinct saved asset bindings
                    # does not authorize assuming the characters are one voice.
                    other = dict(self.voice, asset_hash=digest("other"), voice_id="other")
                    voices["other"] = other
                    last["segment"]["voice_id"] = "other"
                    changed["voice_bindings"]["other"] = other["asset_hash"]
                with self.assertRaises(SpeechVerificationFailed):
                    self.executor.synthesize(changed, voices, self.work / change, self.messages.append)
                self.assertFalse(list((self.work / change / "clips").glob("*/speech-recovery.json")))
        self.assertEqual(self.engine.calls, [])

    def test_recovery_audit_rejects_changed_inputs_even_when_rehashed(self):
        prepared = self.render()
        accepted = read_json(Path(prepared["clips"][0]["wav_path"]).parent / "accepted.json")
        assets = {self.voice["asset_hash"]: self.voice}
        for change in ("text", "style", "voice", "audio", "offset", "metadata", "missing_metadata"):
            request, metadata = deepcopy(accepted["request"]), deepcopy(accepted["engine_metadata"])
            part = metadata["recovery"]["parts"][0]
            child = part["acceptance"]
            if change == "text":
                child["request"]["synthesis_request"]["chunk"]["spoken_text"] = "Different words."
            elif change == "style":
                child["request"]["synthesis_request"]["chunk"]["segment"]["delivery"]["direction"] = "whisper"
            elif change == "voice":
                child["request"]["synthesis_request"]["voice_asset_hash"] = digest("another voice")
            elif change == "audio":
                child["audio_sha256"] = digest("different audio")
            elif change == "offset":
                part["start"] += 1
            elif change == "metadata":
                child["engine_metadata"]["synthesis_text"] = "Unrelated speech."
            else:
                child["engine_metadata"].pop("synthesis_text")
            part["accepted_sha256"] = hashlib.sha256(
                (json.dumps(child, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                generation_identity(request, assets, engine_metadata=metadata)

    def test_old_stop_is_reconsidered_with_new_executable_split_capability(self):
        client = self.adaptive([{"action": "stop", "reason": "No usable action.", "edits": []},
                                {"action": "split", "reason": "Use complete sentences.", "edits": []}])
        with patch("epub_tts_narrator.renderer.speech_recovery.split_recovery_text", return_value=None):
            with self.assertRaises(SpeechVerificationFailed):
                self.render()
        self.assertNotIn("split", client.plans[0]["permitted_actions"])
        self.assertIn("rewrite", client.plans[0]["permitted_actions"])
        whole_calls = len(self.engine.dialogue_calls)
        saved = {path: file_hash(path) for path in (self.work / "clips").glob("*/attempt-*.wav")}
        prepared = self.render()
        self.assertIn("split", client.plans[-1]["permitted_actions"])
        self.assertEqual(len(self.engine.dialogue_calls), whole_calls)
        self.assertTrue(prepared["acceptance"][0]["report"]["passed"])
        self.assertTrue(list((self.work / "clips").glob("*/speech-resolution-*.json")))
        self.assertTrue(all(file_hash(path) == checksum for path, checksum in saved.items()))

    def test_exhausted_shared_budget_resumes_parts_without_new_whole_takes(self):
        self.config["verification"]["max_resolution_attempts"] = 3
        self.executor.config = deepcopy(self.config)
        self.adaptive([{"action": "split", "reason": "Use complete sentences.", "edits": []}])
        with self.assertRaises(SpeechVerificationFailed):
            self.render()
        self.assertEqual(len(self.engine.dialogue_calls), 3)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(len(list((self.work / "clips").glob("*/speech-recovery.json"))), 1)
        prepared = self.render()
        self.assertEqual(len(self.engine.dialogue_calls), 3)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertTrue(prepared["acceptance"][0]["report"]["passed"])


if __name__ == "__main__":
    unittest.main()
