"""Actual FFmpeg chapter jobs using offline synthesis and recognition doubles."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.assembly import probe_audio
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor, _pcm_data_hash
from epub_tts_narrator.renderer.common import digest, file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import RenderBlocked, make_render_plan
from test_renderer_paragraph_workflow import ParagraphFixtureEngine
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureVerifier, write_tone


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class ChapterExecutorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.work = self.root / "work"
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(ffmpeg=FFMPEG, ffprobe=FFPROBE, max_attempts=1,
                                     normalize_loudness=False, paragraph_synthesis=True)
        self.config["quality"] = validate_policy({})
        self.engine = ParagraphFixtureEngine()
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.executor = ChapterExecutor(self.config, engine=self.engine, verifier=self.verifier)
        self.addCleanup(self.executor.close)
        bundle = load_bundle(EXAMPLE / "bundle")
        # A same-role paragraph still exercises remote grouped rendering and
        # paragraph sync; mixed-role passages must travel as separate requests.
        for row in bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        plan = make_render_plan(bundle, self.config)
        self.spec = {
            "version": 1, "book": bundle["source"]["book"],
            "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "chapter": plan["chapters"][0], "total_chapters": len(bundle["chapters"]),
            "voice_bindings": {key: digest({"voice_id": key}) for key in plan["voice_ids"]},
            "render": {}, "renderer_fingerprint": digest(self.executor.identity),
        }
        reference = self.root / "reference.wav"
        write_tone(reference)
        self.voices = {key: {"asset_hash": value, "reference_audio": str(reference),
                             "reference_text": "A saved reference."}
                       for key, value in self.spec["voice_bindings"].items()}
        self.messages = []

    def synthesize(self, spec=None):
        return self.executor.synthesize(spec or self.spec, self.voices, self.work, self.messages.append)

    def assemble(self, prepared, spec=None):
        return self.executor.assemble(spec or self.spec, prepared, self.work, self.messages.append)

    def test_verified_chapter_exports_only_compressed_audio_and_honest_timing(self):
        prepared = self.synthesize()
        # The CPU assembly phase must remain independent of all model state.
        with patch.object(self.executor, "_components", side_effect=AssertionError("model access during assembly")):
            artifacts = self.assemble(prepared)
        self.assertEqual(set(artifacts), {"chapter.m4a", "chapter.flac", "chapter.json", "sync.json",
                                         "verification.json", "manifest.json"})
        self.assertTrue(all(path.suffix != ".wav" for path in artifacts.values()))
        chapter = read_json(artifacts["chapter.json"])
        self.assertEqual(chapter["spec_hash"], digest(self.spec))
        self.assertEqual(chapter["voice_bindings"], self.spec["voice_bindings"])
        self.assertEqual(chapter["encoded_sha256"], file_hash(artifacts["chapter.m4a"]))
        decoded = self.root / "decoded.wav"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                        "-i", str(artifacts["chapter.flac"]), "-c:a", "pcm_s16le", str(decoded)],
                       check=True, timeout=30)
        self.assertEqual(chapter["pcm_data_sha256"], _pcm_data_hash(decoded))
        self.assertFalse((self.work / "chapters").exists())
        self.assertEqual(list((self.work / "chapter-exports").rglob("*.wav")), [])
        for name, codec in (("chapter.m4a", "aac"), ("chapter.flac", "flac")):
            streams = probe_audio(artifacts[name], ffprobe=FFPROBE)["streams"]
            self.assertEqual(next(stream for stream in streams if stream["codec_type"] == "audio")["codec_name"], codec)
        sync = read_json(artifacts["sync.json"])
        self.assertEqual(sync["precision"], "paragraph")
        self.assertEqual(sync["segments"], chapter["segments"])
        self.assertTrue(any("source_segments" in segment for segment in sync["segments"]))
        self.assertFalse(any("words" in segment for segment in sync["segments"]))
        verification = read_json(artifacts["verification.json"])
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["acceptance"], prepared["acceptance"])
        self.assertEqual(self.engine.voice_calls, [])

    def test_resume_and_spacing_change_reuse_accepted_speech(self):
        prepared = self.synthesize()
        first = self.assemble(prepared)
        counts = len(self.engine.speech_calls), len(self.engine.dialogue_calls), len(self.verifier.calls)
        changed = deepcopy(self.spec)
        changed["render"]["paragraph_pause_ms"] = 750
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("regenerated speech")), \
             patch.object(self.engine, "synthesize_dialogue", side_effect=AssertionError("regenerated paragraph")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified accepted speech")):
            self.assertEqual(self.assemble(self.synthesize()), first)
            second = self.assemble(self.synthesize(changed), changed)
        self.assertNotEqual(first["chapter.m4a"].parent, second["chapter.m4a"].parent)
        self.assertNotEqual(read_json(first["chapter.json"])["frames"], read_json(second["chapter.json"])["frames"])
        self.assertEqual(counts, (len(self.engine.speech_calls), len(self.engine.dialogue_calls), len(self.verifier.calls)))

    def test_changed_voice_regenerates_only_its_separate_passage(self):
        self.synthesize()
        before = len(self.engine.speech_calls), len(self.engine.dialogue_calls)
        changed = deepcopy(self.spec)
        changed["voice_bindings"]["voice_lena"] = digest("new-lena")
        self.voices["voice_lena"]["asset_hash"] = changed["voice_bindings"]["voice_lena"]
        self.synthesize(changed)
        self.assertEqual(len(self.engine.speech_calls) - before[0], 1)
        self.assertEqual(len(self.engine.dialogue_calls), before[1])

    def test_failed_verification_never_publishes_chapter(self):
        self.verifier.wrong_texts.add(self.spec["chapter"]["chunks"][0]["spoken_text"])
        with self.assertRaises(RenderBlocked):
            self.synthesize()
        self.assertFalse((self.work / "chapter-exports").exists())

    def test_missing_or_changed_voice_is_rejected_before_generation(self):
        voice_id = next(iter(self.voices))
        self.voices[voice_id]["asset_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Missing or changed registered voice"):
            self.synthesize()
        self.assertFalse(self.engine.speech_calls or self.engine.dialogue_calls)

    def test_prepared_audio_or_spec_tampering_blocks_publication(self):
        prepared = self.synthesize()
        changed = deepcopy(self.spec)
        changed["chapter"]["title"] += " changed"
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.assemble(prepared, changed)
        wav = Path(next(clip["wav_path"] for clip in prepared["clips"] if clip["wav_path"]))
        wav.write_bytes(wav.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed before assembly"):
            self.assemble(prepared)

    def test_corrupt_compressed_export_is_not_silently_reused(self):
        prepared = self.synthesize()
        artifacts = self.assemble(prepared)
        artifacts["chapter.flac"].write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "artifact failed integrity"):
            self.assemble(prepared)

    def test_empty_chapter_is_explicit_and_does_not_invent_audio(self):
        empty = deepcopy(self.spec)
        empty["chapter"]["chunks"] = []
        empty["voice_bindings"] = {}
        artifacts = self.assemble(self.synthesize(empty), empty)
        self.assertEqual(set(artifacts), {"chapter.json", "sync.json", "verification.json", "manifest.json"})
        chapter = read_json(artifacts["chapter.json"])
        self.assertEqual(chapter["status"], "empty")
        self.assertEqual(chapter["frames"], 0)
        self.assertIsNone(chapter["audio_file"])
        self.assertFalse(self.engine.speech_calls or self.engine.dialogue_calls)

    def test_client_cannot_override_worker_paths_or_executables(self):
        changed = deepcopy(self.spec)
        changed["render"]["ffmpeg"] = "/arbitrary/program"
        with self.assertRaisesRegex(ValueError, "unsupported settings"):
            self.synthesize(changed)
        self.assertFalse(self.engine.speech_calls or self.engine.dialogue_calls)

    def test_capabilities_match_backend_and_reject_unsupported_paragraphs(self):
        self.assertTrue(self.executor.multi_speaker)
        self.engine.capabilities = {"multi_speaker": False}
        self.assertFalse(self.executor.multi_speaker)
        with self.assertRaisesRegex(ValueError, "cannot render multi-speaker"):
            self.synthesize()
        self.assertFalse(self.engine.speech_calls or self.engine.dialogue_calls)

    def test_export_failure_removes_only_its_temporary_chapter_pcm(self):
        prepared = self.synthesize()
        before = {path: file_hash(path) for path in (self.work / "clips").rglob("*.wav")}
        with patch("epub_tts_narrator.renderer.chapter_executor._flac_master", side_effect=ValueError("compress failed")):
            with self.assertRaisesRegex(ValueError, "compress failed"):
                self.assemble(prepared)
        self.assertEqual(list((self.work / "chapter-exports").iterdir()), [])
        self.assertEqual(before, {path: file_hash(path) for path in (self.work / "clips").rglob("*.wav")})
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("regenerated speech")), \
             patch.object(self.engine, "synthesize_dialogue", side_effect=AssertionError("regenerated paragraph")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified accepted speech")):
            artifacts = self.assemble(self.synthesize())
        self.assertTrue(artifacts["chapter.m4a"].is_file())

    def test_parallel_assemblies_keep_shared_accepted_clips_intact(self):
        changed = deepcopy(self.spec)
        changed["render"]["paragraph_pause_ms"] = 750
        first, second = self.synthesize(), self.synthesize(changed)
        before = {path: file_hash(path) for path in (self.work / "clips").rglob("*.wav")}
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(self.assemble, first)
            b = pool.submit(self.assemble, second, changed)
            exports = a.result(), b.result()
        self.assertNotEqual(exports[0]["chapter.m4a"].parent, exports[1]["chapter.m4a"].parent)
        self.assertEqual(before, {path: file_hash(path) for path in (self.work / "clips").rglob("*.wav")})
        self.assertEqual(list((self.work / "chapter-exports").rglob("*.wav")), [])


if __name__ == "__main__":
    unittest.main()
