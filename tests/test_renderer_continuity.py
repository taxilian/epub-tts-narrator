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
from shared_fixtures import renderer_config, route_mlx_takes_to_fake_generate, timed_transcript
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


class PromptTooLongError(ValueError):
    """Same class name as the pinned runtime's prompt overflow error."""


def tagged(text, speaker=0):
    return f"<|speaker:{speaker}|> {text}"


def histories(model):
    return [[text for text, codes in options["previous_takes"]] for text, options in model.calls]


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ContinuityTests(unittest.TestCase):
    def setUp(self):
        import soundfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = renderer_config(self.root)
        self.engine = Drama3MLXEngine(self.root)
        self.engine._identity = {"engine": "drama3_mlx", "model": "fixture", "revision": "one"}
        self.model = self.engine._model = Model()
        route_mlx_takes_to_fake_generate(self)
        self.engine._soundfile = soundfile
        self.verifier = Verifier(self.model)
        reference = self.root / "voice.wav"
        write_tone(reference)
        self.voices = {"v1": {"kind": "reference", "voice_id": "v1", "asset_hash": "a" * 64,
            "reference_audio": str(reference), "reference_sha256": file_hash(reference),
            "reference_text": "A fixed reference."}}
        self.voices["v2"] = dict(self.voices["v1"], voice_id="v2", asset_hash="b" * 64,
                                 reference_text="Another fixed reference.")
        self.chunks = passages(["First words.", "Second words.", "Third words.", "Fourth words."])
        for chunk in self.chunks:
            chunk["segment"]["delivery"]["direction"] = ""
            chunk["segment"]["kind"] = "narration"

    def run_chapter(self, chunks=None):
        return _render_chapter_chunks({"number": 1, "chunks": chunks or self.chunks}, self.voices,
            self.engine, self.engine.identity, self.verifier, self.config, self.root / "work",
            notify=lambda message: None)

    def test_pinned_first_narration_and_latest_with_fixed_references_survive_resume(self):
        original = deepcopy(self.voices)
        self.run_chapter()
        self.assertEqual(histories(self.model), [
            [], [tagged("First words.")], [tagged("First words."), tagged("Second words.")],
            [tagged("First words."), tagged("Third words.")]])
        self.assertTrue(all([reference.text for reference in options["references"]]
                            == [tagged("A fixed reference.")] for text, options in self.model.calls))
        self.assertEqual(self.voices, original)
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 4)

    def test_fish_sampling_defaults_apply_to_every_take(self):
        self.run_chapter(self.chunks[:1])
        options = self.model.calls[0][1]
        self.assertEqual((options["temperature"], options["top_p"], options["top_k"]), (0.7, 0.7, 30))

    def test_rejected_take_does_not_advance_history(self):
        self.verifier.reject.add(240)
        self.run_chapter()
        self.assertEqual(histories(self.model)[1:3], [[tagged("First words.")], [tagged("First words.")]])
        self.assertEqual(self.model.calls[3][1]["previous_takes"][-1][1][0, 0], 3)

    def test_strict_history_reuse_makes_the_rest_of_the_chapter_stale_after_a_change(self):
        self.config["render"]["history_reuse"] = "strict"
        self.run_chapter()
        self.chunks[2]["spoken_text"] = "Changed words."
        self.chunks[2]["segment"]["text"] = "Changed words."
        self.run_chapter()
        self.assertEqual([text for text, options in self.model.calls[4:]],
                         [tagged("Changed words."), tagged("Fourth words.")])
        self.chunks[0]["spoken_text"] = "Other words."
        self.chunks[0]["segment"]["text"] = "Other words."
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 10)

    def test_tolerant_history_reuse_regenerates_only_the_changed_take(self):
        self.assertEqual(self.config["render"]["history_reuse"], "tolerant")
        self.run_chapter()
        self.chunks[2]["spoken_text"] = self.chunks[2]["segment"]["text"] = "Changed words."
        self.run_chapter()
        # Only the changed line is generated, hearing the current history;
        # the later take that heard the old version is kept.
        self.assertEqual([text for text, options in self.model.calls[4:]], [tagged("Changed words.")])
        self.assertEqual(histories(self.model)[4], [tagged("First words."), tagged("Second words.")])
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 5)
        # A changed pinned first take regenerates only itself as well.
        self.chunks[0]["spoken_text"] = self.chunks[0]["segment"]["text"] = "Other words."
        self.run_chapter()
        self.assertEqual([text for text, options in self.model.calls[5:]], [tagged("Other words.")])
        # A take generated afterwards hears the takes the chapter now has.
        self.chunks[3]["spoken_text"] = self.chunks[3]["segment"]["text"] = "Final words."
        self.run_chapter()
        self.assertEqual([text for text, options in self.model.calls[6:]], [tagged("Final words.")])
        self.assertEqual(histories(self.model)[6], [tagged("Other words."), tagged("Changed words.")])

    def test_tolerant_history_reuse_keeps_only_ordinarily_verified_takes(self):
        from unittest.mock import patch
        from epub_tts_narrator.renderer import speech_cache
        self.run_chapter()
        self.chunks[2]["spoken_text"] = self.chunks[2]["segment"]["text"] = "Changed words."
        ordinary = speech_cache.ordinarily_verified

        def best_effort_fourth(record):
            # Treat the fourth line's take as a best-effort exhaustion choice.
            return ordinary(record) and record["request"]["chunk"]["spoken_text"] != "Fourth words."

        with patch.object(speech_cache, "ordinarily_verified", side_effect=best_effort_fourth):
            self.run_chapter()
        # Its history changed, so the best-effort take is not kept: it gets a
        # fresh chance at an ordinarily verified take.
        self.assertEqual([text for text, options in self.model.calls[4:]],
                         [tagged("Changed words."), tagged("Fourth words.")])
        self.assertTrue(ordinary({"report": {"passed": True}, "engine_metadata": {}}))
        self.assertFalse(ordinary({"report": {"passed": True, "text_confirmation_method": "llm_exhaustion_selection"}}))
        self.assertFalse(ordinary({"report": {"passed": True}, "engine_metadata": {"exhaustion_selection": {}}}))

    def test_tolerant_history_reuse_never_moves_a_take_to_another_line(self):
        self.run_chapter()
        # The second line now says what the fourth said; the fourth line's
        # take heard a different history at another position, so it is not used.
        self.chunks[1]["spoken_text"] = self.chunks[1]["segment"]["text"] = "Fourth words."
        self.run_chapter()
        self.assertEqual([text for text, options in self.model.calls[4:]], [tagged("Fourth words.")])

    def test_heading_never_becomes_the_pinned_take(self):
        self.chunks[0]["segment"]["kind"] = "heading"
        self.run_chapter()
        self.assertEqual(histories(self.model), [
            [], [tagged("First words.")], [tagged("Second words.")],
            [tagged("Second words."), tagged("Third words.")]])
    def test_single_voice_paragraph_participates(self):
        grouped = chunks_of(group_paragraphs(plan(self.chunks[:2])))
        self.run_chapter(grouped + self.chunks[2:])
        self.assertEqual(histories(self.model)[1], [self.model.calls[0][0]])

    def test_other_voices_never_enter_a_lines_history(self):
        for chunk, voice in zip(self.chunks, ["v1", "v2", "v1", "v2"]):
            chunk["segment"].update(voice_id=voice, role_id=f"role-{voice}")
        self.run_chapter()
        self.assertEqual(histories(self.model), [[], [], [tagged("First words.")], [tagged("Second words.")]])
        self.assertTrue(all(len(options["references"]) == 1 for text, options in self.model.calls))
        self.assertEqual([options["references"][0].text for text, options in self.model.calls],
                         [tagged("A fixed reference."), tagged("Another fixed reference.")] * 2)

    def test_linked_voices_sharing_one_recording_keep_separate_histories(self):
        # The narrator is often linked to a character's voice (voice-aliases):
        # both clone the same reference, but narration and that character's
        # dialogue are different performances and never share history.
        self.voices["sal"] = dict(self.voices["v1"], voice_id="sal")
        for chunk, voice in zip(self.chunks, ["v1", "sal", "v1", "sal"]):
            chunk["segment"].update(voice_id=voice, role_id=f"role-{voice}")
        self.chunks[1]["segment"]["kind"] = self.chunks[3]["segment"]["kind"] = "dialogue"
        self.run_chapter()
        self.assertEqual(histories(self.model), [[], [], [tagged("First words.")], [tagged("Second words.")]])
        self.assertTrue(all([reference.text for reference in options["references"]]
                            == [tagged("A fixed reference.")] for text, options in self.model.calls))
        # Resume replays the same per-voice histories and regenerates nothing.
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 4)

    def test_mixed_paragraph_hears_only_turns_by_its_own_voices(self):
        mixed = passages(["Two voices. ", "One paragraph."], voices=["v1", "v2"], block="b2", offset=100)
        for chunk in mixed:
            chunk["segment"]["delivery"]["direction"] = ""
            chunk["segment"]["kind"] = "dialogue"
        grouped = chunks_of(group_paragraphs(plan(mixed), multi_speaker=True))
        reply = deepcopy(self.chunks[3])
        reply["segment"].update(voice_id="v2", role_id="role-v2", kind="dialogue")
        self.run_chapter([self.chunks[0], grouped[0], reply])
        self.assertEqual(self.model.calls[1][0], "<|speaker:0|> Two voices.\n<|speaker:1|> One paragraph.")
        self.assertEqual(histories(self.model)[1], [tagged("First words.")])
        # The v2 reply cannot hear the paragraph, which also contains v1.
        self.assertEqual(histories(self.model)[2], [])
    def test_silent_controls_keep_history_and_chapters_start_fresh(self):
        silent = deepcopy(self.chunks[1])
        silent.update(silent_control=True, spoken_text="")
        self.run_chapter([self.chunks[0], silent, self.chunks[2]])
        self.run_chapter([self.chunks[3]])
        self.assertEqual(histories(self.model), [[], [tagged("First words.")], []])

    def test_engine_without_history_support_fails_before_synthesis(self):
        from test_renderer_workflow import FixtureEngine
        self.engine = FixtureEngine({})
        self.engine.capabilities = {"reference_voice": True, "voice_design": True}
        with self.assertRaisesRegex(ValueError, "continuity"):
            self.run_chapter()
        self.assertEqual(self.engine.speech_calls, [])

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

    def test_alias_cannot_hide_changed_history(self):
        from epub_tts_narrator.renderer.speech_identity import generation_identity
        self.run_chapter([self.chunks[0]])
        record = read_json(next((self.root / "work").rglob("accepted.json")))
        request = record["request"]
        request["engine"]["cache_model_alias"] = "same-alias"
        assets = {voice["asset_hash"]: voice for voice in self.voices.values()}
        old = generation_identity(request, assets)
        changed = deepcopy(request)
        changed["continuity"]["takes"] = [{"codes_sha256": "b" * 64}]
        self.assertNotEqual(old, generation_identity(changed, assets))

    def test_prompt_overflow_follows_fish_fallback_order(self):
        from unittest.mock import patch
        generate = self.model.generate
        def bounded(text, **options):
            if len(options["previous_takes"]) > 1:
                raise PromptTooLongError("prompt has more than 8192 rows")
            yield from generate(text, **options)
        with patch.object(self.model, "generate", side_effect=bounded):
            self.run_chapter(self.chunks[:3])
        self.assertEqual(histories(self.model), [[], [tagged("First words.")], [tagged("First words.")]])
        record = [read_json(path) for path in (self.root / "work").rglob("accepted.json")
                  if read_json(path)["engine_metadata"].get("history_available") == 2]
        self.assertEqual([item["engine_metadata"]["history_turns"] for item in record], [1])

    def test_prompt_overflow_without_history_fails(self):
        from unittest.mock import patch
        def overflow(text, **options):
            raise PromptTooLongError("prompt has more than 8192 rows")
            yield
        with patch.object(self.model, "generate", side_effect=overflow):
            with self.assertRaisesRegex(Exception, "too long even without chapter history"):
                self.run_chapter(self.chunks[:1])

    def test_simultaneous_lanes_use_their_own_voice_history_without_advancing_it(self):
        simultaneous = deepcopy(self.chunks[1])
        simultaneous["segment"].update(voice_id=None, role_id=None,
            simultaneous_speakers=[{"role_id": "r1", "voice_id": "v1"},
                                   {"role_id": "r2", "voice_id": "v2"}])
        clips, accepted = self.run_chapter([self.chunks[0], simultaneous, self.chunks[3]])
        self.assertEqual(len(accepted[1]["speaker_lanes"]), 2)
        self.assertIn("mix", clips[1])
        self.assertEqual(histories(self.model), [[], [tagged("First words.")], [], [tagged("First words.")]])
    def test_accepted_repair_keeps_existing_history(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity
        self.run_chapter([self.chunks[0]])
        record = read_json(next((self.root / "work").rglob("accepted.json")))
        request = deepcopy(record["request"])
        request.pop("continuity")
        request.pop("synthesis_identity")
        state = ChapterContinuity(self.engine.identity, self.voices)
        previous = {"identity": {"previous": True, "speakers": ["a" * 64]}, "pinnable": True, "voices": []}
        for barrier in ("recovery", "clip_repair", "exhaustion_selection"):
            with self.subTest(barrier=barrier):
                state.entries = [previous]
                with state.transaction(request):
                    repaired = deepcopy(record)
                    repaired["engine_metadata"][barrier] = {}
                    state.accept(Path("not-read-for-a-barrier"), repaired)
                self.assertEqual(state.entries, [previous])
    def test_interrupted_chapter_replays_only_accepted_predecessors(self):
        from unittest.mock import patch
        generate = self.model.generate
        def interrupt(text, **options):
            if text == tagged("Second words."):
                raise KeyboardInterrupt("fixture interruption")
            return generate(text, **options)
        with patch.object(self.model, "generate", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_chapter()
        self.run_chapter()
        self.assertEqual(len(self.model.calls), 4)
        self.assertEqual(histories(self.model)[1], [tagged("First words.")])


class ContinuityPolicyTests(unittest.TestCase):
    def test_each_chunk_waits_only_for_earlier_chunks_of_its_own_voices(self):
        from epub_tts_narrator.renderer.continuity import history_dependencies
        chunks = passages(["Narration.", "Said.", "More narration.", "Said again.", "Both."],
                          voices=["n", "c", "n", "c", "n"])
        both = chunks[4]
        both["paragraph_turns"] = [deepcopy(chunks[0]), deepcopy(chunks[1])]
        self.assertEqual(history_dependencies(chunks), [
            frozenset(), frozenset(), frozenset({0}), frozenset({1}),
            # A paragraph with both voices waits for every earlier take of either.
            frozenset({0, 1, 2, 3})])

    def test_history_comes_from_earlier_lines_even_when_a_later_one_finished_first(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity
        state = ChapterContinuity({"engine": "fixture"}, {})
        def entry(position, name):
            return {"voices": ["n"], "pinnable": True, "position": position,
                    "identity": {"request_hash": name, "audio_sha256": name}}
        # The take at position 5 was accepted before the one at position 2.
        state.entries = [entry(1, "first"), entry(5, "later"), entry(2, "second")]
        names = lambda takes: [take["identity"]["request_hash"] for take in takes]
        self.assertEqual(names(state.takes(frozenset({"n"}), before=3)), ["first", "second"])
        self.assertEqual(names(state.takes(frozenset({"n"}), before=2)), ["first"])
        self.assertEqual(names(state.takes(frozenset({"n"}), before=6)), ["first", "later"])

    def test_only_heading_only_takes_cannot_be_pinned(self):
        from epub_tts_narrator.renderer.continuity import is_pinnable
        chunks = passages(["Chapter 4", "Narration.", "Said."])
        for chunk, kind in zip(chunks, ["heading", "narration", "dialogue"]):
            chunk["segment"]["kind"] = kind
        self.assertEqual([is_pinnable(chunk) for chunk in chunks], [False, True, True])
    def test_candidates_drop_latest_then_pinned_and_respect_speaker_limit(self):
        from epub_tts_narrator.renderer.continuity import candidates
        references = [{"speaker": index} for index in range(17)]
        pinned = ("<|speaker:1|> pinned", [[0]])
        latest = (" ".join(f"<|speaker:{index}|> x" for index in range(2, 17)), [[0]])
        options = candidates("<|speaker:0|> new", references, [pinned, latest])
        self.assertEqual([len(kept) for selected, kept in options], [1, 0])
        self.assertEqual([[reference["speaker"] for reference in selected] for selected, kept in options],
                         [[0, 1], [0]])
