"""Accepted history flows through real chapter transactions and the local adapter."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import re
import tempfile
import unittest

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.engines import Drama3MLXEngine
from epub_tts_narrator.renderer.workflow import _render_chapter_chunks
from shared_fixtures import renderer_config, timed_transcript
from test_renderer_paragraphs import passages, plan, chunks_of
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone
from epub_tts_narrator.renderer.paragraphs import group_paragraphs


@dataclass(frozen=True)
class Reference:
    text: str
    codes: object


class Model:
    sample_rate = 44100

    def __init__(self):
        self.calls = []
        self.transcripts = {}
        self.model = SimpleNamespace(config=SimpleNamespace(
            audio_decoder=SimpleNamespace(num_codebooks=10, vocab_size=1024)))

    def reference(self, audio, rate, text):
        import numpy as np
        return Reference(text, np.zeros((10, 10), dtype=np.int64))

    def generate(self, text, **options):
        import numpy as np
        index = len(self.calls) + 1
        self.calls.append((text, deepcopy(options)))
        frequency = 200 + 20 * index
        self.transcripts[frequency] = re.sub(r'<[^>]*>|\[[^]]*\]', '', text).strip()
        audio = (0.15 * np.sin(2 * np.pi * frequency * np.arange(44100) / 44100)).astype(np.float32)
        yield SimpleNamespace(text=text, codes=np.full((10, 20), index, dtype=np.int64),
                              audio=audio, finish_reason="eos")


class Verifier:
    identity = {"backend": "fixture"}

    def __init__(self, model):
        self.model = model
        self.reject = set()

    def transcribe(self, path):
        import numpy as np
        import soundfile
        audio, rate = soundfile.read(path)
        frequency = round(np.argmax(abs(np.fft.rfft(audio))) * rate / len(audio))
        text = self.model.transcripts[frequency]
        if frequency in self.reject:
            text = "Unrelated invented words."
        return timed_transcript(text)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ContinuityTests(unittest.TestCase):
    def setUp(self):
        import soundfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = renderer_config(self.root)
        self.config["render"]["continuity"] = "chapter"
        self.engine = Drama3MLXEngine(self.root)
        self.engine._identity = {"engine": "drama3_mlx", "model": "fixture", "revision": "one"}
        self.model = self.engine._model = Model()
        self.engine._soundfile = soundfile
        self.verifier = Verifier(self.model)
        reference = self.root / "voice.wav"
        write_tone(reference)
        self.voices = {"v1": {"kind": "reference", "voice_id": "v1", "asset_hash": "a" * 64,
            "reference_audio": str(reference), "reference_sha256": file_hash(reference),
            "reference_text": "A fixed reference."}}
        self.chunks = passages(["First words.", "Second words.", "Third words.", "Fourth words."])
        for chunk in self.chunks:
            chunk["segment"]["delivery"]["direction"] = ""

    def run_chapter(self, chunks=None):
        return _render_chapter_chunks({"number": 1, "chunks": chunks or self.chunks}, self.voices,
            self.engine, self.engine.identity, self.verifier, self.config, self.root / "work",
            notify=lambda message: None)

    def test_fixed_reference_and_first_latest_survive_resume(self):
        original = deepcopy(self.voices)
        self.run_chapter()
        histories = [[text for text, codes in options.get("previous_takes", [])]
                     for text, options in self.model.calls]
        self.assertEqual(histories, [[], ["First words."], ["First words.", "Second words."],
                                    ["First words.", "Third words."]])
        self.assertTrue(all(options["references"][0].text == "A fixed reference."
                            for text, options in self.model.calls))
        self.assertEqual(self.voices, original)
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 4)

    def test_rejected_take_does_not_advance_history(self):
        self.verifier.reject.add(240)
        self.run_chapter()
        histories = [[text for text, codes in options.get("previous_takes", [])]
                     for text, options in self.model.calls]
        self.assertEqual(histories[1:3], [["First words."], ["First words."]])
        self.assertEqual(self.model.calls[3][1]["previous_takes"][-1][1][0, 0], 3)

    def test_predecessor_change_invalidates_following_takes(self):
        self.run_chapter()
        self.chunks[0]["spoken_text"] = "Changed words."
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 8)

    def test_single_voice_paragraph_participates(self):
        grouped = chunks_of(group_paragraphs(plan(self.chunks[:2])))
        self.run_chapter(grouped + self.chunks[2:])
        self.assertTrue(self.model.calls[1][1].get("previous_takes"))

    def test_unsupported_backend_fails_before_synthesis(self):
        from test_renderer_workflow import FixtureEngine
        self.engine = FixtureEngine({})
        with self.assertRaisesRegex(ValueError, "continuity"):
            self.run_chapter()
        self.assertEqual(self.engine.speech_calls, [])

    def test_off_keeps_legacy_generation_without_codes(self):
        self.config["render"]["continuity"] = "off"
        self.run_chapter()
        self.assertTrue(all("previous_takes" not in options for text, options in self.model.calls))
        self.assertEqual(list((self.root / "work").rglob("*.codes.json")), [])

    def test_corrupt_codes_block_resume_without_regenerating_or_reencoding(self):
        self.run_chapter()
        artifact = next((self.root / "work").rglob("*.codes.json"))
        artifact.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "codes artifact"):
            self.run_chapter()
        self.assertEqual(len(self.model.calls), 4)

    def test_missing_codes_are_not_reconstructed_from_accepted_wav(self):
        self.run_chapter()
        next((self.root / "work").rglob("*.codes.json")).unlink()
        with self.assertRaises((ValueError, OSError)):
            self.run_chapter()
        self.assertEqual(len(self.model.calls), 4)

    def test_chapter_and_silent_control_reset_history(self):
        silent = deepcopy(self.chunks[1])
        silent.update(silent_control=True, spoken_text="")
        self.run_chapter([self.chunks[0], silent, self.chunks[2]])
        self.run_chapter([self.chunks[3]])
        self.assertEqual([options["previous_takes"] for text, options in self.model.calls], [[], [], []])

    def test_mixed_paragraph_is_an_explicit_history_barrier(self):
        self.voices["v2"] = dict(self.voices["v1"], voice_id="v2", asset_hash="b" * 64)
        mixed = passages(["Two voices. ", "One paragraph."], voices=["v1", "v2"])
        grouped = chunks_of(group_paragraphs(plan(mixed), multi_speaker=True))
        self.run_chapter([self.chunks[0], grouped[0], self.chunks[3]])
        self.assertEqual([options["previous_takes"] for text, options in self.model.calls], [[], [], []])

    def test_codec_ranges_and_shapes_are_rejected_even_with_matching_checksum(self):
        from epub_tts_narrator.renderer.continuity_codes import load_codes
        from epub_tts_narrator.renderer.common import write_json
        self.run_chapter([self.chunks[0]])
        record = read_json(next((self.root / "work").rglob("accepted.json")))
        marker = record["engine_metadata"]["continuity_codes"]
        path = next((self.root / "work").rglob("*.codes.json"))
        original = read_json(path)
        cases = [[], [[1]], [[-1] * 20] * 10, [[1024] * 20] * 10,
                 [[1.5] * 20] * 10, [[True] * 20] * 10]
        for codes in cases:
            with self.subTest(codes=codes[:1]):
                write_json(path, dict(original, codes=codes))
                with self.assertRaisesRegex(ValueError, "dimensions or range"):
                    load_codes(path, dict(marker, sha256=file_hash(path)))

    def test_alias_cannot_hide_changed_history_or_runtime(self):
        from epub_tts_narrator.renderer.speech_identity import generation_identity
        self.run_chapter([self.chunks[0]])
        record = read_json(next((self.root / "work").rglob("accepted.json")))
        request = record["request"]
        request["engine"]["cache_model_alias"] = "same-alias"
        assets = {voice["asset_hash"]: voice for voice in self.voices.values()}
        old = generation_identity(request, assets)
        changed = deepcopy(request)
        changed["continuity"]["runtime"] = "b" * 64
        self.assertNotEqual(old, generation_identity(changed, assets))

    def test_prompt_overflow_never_sheds_selected_history(self):
        from unittest.mock import patch
        self.run_chapter([self.chunks[0]])
        generate = self.model.generate
        seen = []
        def bounded(text, **options):
            seen.append(options["previous_takes"])
            if options["previous_takes"]:
                raise ValueError("prompt has more than 8192 rows")
            return generate(text, **options)
        with patch.object(self.model, "generate", side_effect=bounded):
            with self.assertRaises(ValueError):
                self.run_chapter(self.chunks[:2])
        self.assertTrue(seen)
        self.assertTrue(all(len(history) == 1 for history in seen))

    def test_simultaneous_lanes_stay_independent_and_reset_following_history(self):
        self.voices["v2"] = dict(self.voices["v1"], voice_id="v2", asset_hash="b" * 64)
        simultaneous = deepcopy(self.chunks[1])
        simultaneous["segment"].update(voice_id=None, role_id=None,
            simultaneous_speakers=[{"role_id": "r1", "voice_id": "v1"},
                                   {"role_id": "r2", "voice_id": "v2"}])
        clips, accepted = self.run_chapter([self.chunks[0], simultaneous, self.chunks[3]])
        self.assertEqual(len(accepted[1]["speaker_lanes"]), 2)
        self.assertIn("mix", clips[1])
        self.assertTrue(all(not options["previous_takes"] for text, options in self.model.calls))

    def test_accepted_repair_is_a_barrier_not_a_lexical_rejection(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity
        self.run_chapter([self.chunks[0]])
        record = read_json(next((self.root / "work").rglob("accepted.json")))
        request = deepcopy(record["request"])
        request.pop("continuity")
        request.pop("synthesis_identity")
        state = ChapterContinuity(self.engine.identity)
        for barrier in ("recovery", "clip_repair", "exhaustion_selection"):
            with self.subTest(barrier=barrier):
                state.history["a" * 64] = [{"identity": {"previous": True}}]
                with state.transaction(request):
                    repaired = deepcopy(record)
                    repaired["engine_metadata"][barrier] = {}
                    state.accept(Path("not-read-for-a-barrier"), repaired)
                with state.transaction(request) as bound:
                    self.assertEqual(bound["continuity"]["takes"], [])

    def test_off_request_identity_and_remote_projection_are_unchanged(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError, _safe_render, validate_remote_plan
        from epub_tts_narrator.renderer.continuity import enabled
        before = deepcopy(self.config["render"])
        before.pop("continuity")
        self.assertEqual(_safe_render(before), _safe_render(dict(before, continuity="off")))
        with self.assertRaisesRegex(ValueError, "continuity"):
            enabled({"render": {"continuity": "unknown"}})
        with self.assertRaisesRegex(ChapterClientError, "continuity"):
            validate_remote_plan({}, self.config, {})

    def test_interrupted_chapter_replays_only_accepted_predecessors(self):
        from unittest.mock import patch
        generate = self.model.generate
        def interrupt(text, **options):
            if text == "Second words.":
                raise KeyboardInterrupt("fixture interruption")
            return generate(text, **options)
        with patch.object(self.model, "generate", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_chapter()
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 4)
        self.assertEqual([text for text, codes in self.model.calls[1][1]["previous_takes"]], ["First words."])
