"""Accepted history flows through real chapter transactions and the local adapter."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import re
import tempfile
import unittest

from epub_tts_narrator.renderer.assembly import compose_chapter
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

    def test_scene_divider_control_is_silent_between_paragraphs_and_on_resume(self):
        first = passages(["First words."], block="b1")[0]
        divider = passages(["* * *"], block="b2")[0]
        divider["silent_control"] = True
        third = passages(["Third words."], block="b3")[0]
        for chunk in (first, divider, third):
            chunk["segment"]["delivery"]["direction"] = ""
            chunk["segment"]["kind"] = "narration"
        chunks = [first, divider, third]

        clips, acceptance = self.run_chapter(chunks)

        self.assertIsNone(clips[1]["wav_path"])
        self.assertEqual(len(acceptance), 2)
        self.assertEqual([text for text, options in self.model.calls],
                         [tagged("First words."), tagged("Third words.")])
        self.assertEqual(histories(self.model), [[], [tagged("First words.")]])
        with_divider = compose_chapter(clips, self.root / "chapter-with-divider.wav")
        without_divider = compose_chapter([clips[0], clips[2]], self.root / "chapter-without-divider.wav")
        self.assertEqual(with_divider["frames"], without_divider["frames"])
        self.assertEqual((self.root / "chapter-with-divider.wav").read_bytes(),
                         (self.root / "chapter-without-divider.wav").read_bytes())
        timing = with_divider["segments"][1]
        self.assertTrue(timing["no_speech"])
        self.assertEqual(timing["chapter_start_frame"], timing["chapter_end_frame"])
        self.assertEqual(timing["source_text"], "* * *")
        self.assertEqual(timing["source_ranges"], [{"start": 0, "end": 5}])

        self.run_chapter(chunks)
        self.assertEqual(len(self.model.calls), 2)

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
        with self.subTest(barrier="silence_stall_recovery"):
            # A take kept despite one long pause never becomes history.
            state.entries = [previous]
            with state.transaction(request):
                stalled = deepcopy(record)
                stalled["report"]["silence_stall_recovery"] = {"version": 1, "decision": "accepted"}
                state.accept(Path("not-read-for-a-barrier"), stalled)
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


READING = "S EH2 N EH1 D R AH0"


def chosen(chunk, name="Ce’Nedra", phonemes=READING):
    """Give a passage a person-chosen CMU reading of ``name``."""
    from epub_tts_narrator.performance_plan import text_sha256
    text = chunk["spoken_text"]
    start = text.index(name)
    chunk["expressive_direction"] = ""
    chunk["expressive_performance"] = {"version": 1, "text_sha256": text_sha256(text), "controls": [
        {"kind": "pronunciation", "start": start, "end": start + len(name), "text": name,
         "alphabet": "cmu_arpabet", "phonemes": phonemes}]}
    return chunk


class NameModel(Model):
    """Heard as written, with each phoneme span heard as the name it reads."""

    def __init__(self, heard="Ce’Nedra"):
        super().__init__()
        self.heard = heard

    def generate(self, text, **options):
        for result in super().generate(text, **options):
            frequency = max(self.transcripts)
            spoken = re.sub(r"<\|phoneme_start\|>.*?<\|phoneme_end\|>", self.heard, text)
            self.transcripts[frequency] = re.sub(r'<[^>]*>|\[[^]]*\]', '', spoken).strip()
            yield result


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ChosenReadingHistoryTests(unittest.TestCase):
    """A line with a chosen pronunciation hears an earlier take saying it right."""

    setUp = ContinuityTests.setUp
    run_chapter = ContinuityTests.run_chapter

    def name_chapter(self):
        self.model = self.engine._model = NameModel()
        self.verifier = Verifier(self.model)
        chunks = passages(["Chapter One", "The road ran on.", "Ce’Nedra objected.", "Garion rode ahead.",
                           "Ce’Nedra laughed."])
        for chunk in chunks:
            chunk["segment"]["delivery"]["direction"] = ""
            chunk["segment"]["kind"] = "narration"
        chunks[0]["segment"]["kind"] = "heading"
        chosen(chunks[2]), chosen(chunks[4])
        return chunks

    def test_the_latest_turn_is_an_earlier_take_saying_the_reading(self):
        self.run_chapter(self.name_chapter())
        named = tagged(f"<|phoneme_start|>{READING}<|phoneme_end|> objected.")
        road = tagged("The road ran on.")
        # A line without a chosen reading keeps Fish's first + latest.
        self.assertEqual(histories(self.model)[3], [road, named])
        # The name's line hears the take that said it, not the line before.
        self.assertEqual(histories(self.model)[4], [road, named])
        self.assertEqual(len(self.model.calls), 5)

    def test_a_take_heard_rearranged_is_never_an_example(self):
        from epub_tts_narrator.renderer.continuity import _readings_heard_right
        chunk = chosen(passages(["Ce’Nedra objected."])[0])
        record = lambda heard: {"request": {"kind": "speech", "chunk": chunk},
                                "report": {"passed": True, "expected_text": chunk["spoken_text"], "transcript": heard}}
        self.assertEqual(_readings_heard_right(record("Sinedra objected.")), {READING})
        self.assertEqual(_readings_heard_right(record("Ce’Nedra objected.")), {READING})
        self.assertEqual(_readings_heard_right(record("Sendera objected.")), frozenset())
        self.assertEqual(_readings_heard_right(record("She objected.")), frozenset())


class ChosenReadingPolicyTests(unittest.TestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity
        self.state = ChapterContinuity({"engine": "fixture"}, {})

    def entry(self, position, name, readings=(), voices=("n",), pinnable=True):
        return {"voices": list(voices), "pinnable": pinnable, "position": position, "readings": frozenset(readings),
                "identity": {"request_hash": name, "audio_sha256": name}}

    def names(self, **options):
        return [take["identity"]["request_hash"] for take in self.state.takes(frozenset({"n"}), **options)]

    def test_the_most_recent_earlier_example_replaces_the_latest_turn(self):
        self.state.entries = [self.entry(1, "first"), self.entry(2, "said"), self.entry(3, "said again", ["A"]),
                              self.entry(4, "said once more", ["A"]), self.entry(5, "plain"), self.entry(7, "after", ["A"])]
        self.assertEqual(self.names(before=6, readings={"A"}), ["first", "said once more"])
        # Without a chosen reading, or with no example of it, history is Fish's.
        self.assertEqual(self.names(before=6), ["first", "plain"])
        self.assertEqual(self.names(before=6, readings={"B"}), ["first", "plain"])

    def test_history_already_saying_the_reading_is_kept(self):
        self.state.entries = [self.entry(1, "first", ["A"]), self.entry(2, "example", ["A"]), self.entry(3, "plain")]
        self.assertEqual(self.names(before=4, readings={"A"}), ["first", "plain"])
        self.state.entries = [self.entry(1, "first"), self.entry(2, "example", ["A"]), self.entry(3, "latest", ["A"])]
        self.assertEqual(self.names(before=4, readings={"A"}), ["first", "latest"])

    def test_the_example_saying_the_most_chosen_readings_wins(self):
        self.state.entries = [self.entry(1, "first"), self.entry(2, "both", ["A", "B"]), self.entry(3, "one", ["A"]),
                              self.entry(4, "plain")]
        self.assertEqual(self.names(before=5, readings={"A", "B"}), ["first", "both"])

    def test_a_turn_saying_one_reading_is_never_traded_for_one_saying_another(self):
        # Book 4: the latest turn said Ce'Nedra right; Polgara was missing, and the
        # turn saying only Polgara (with "Sendaria" just before) replaced it.
        self.state.entries = [self.entry(1, "first"), self.entry(2, "polgara", ["P"], pinnable=False),
                              self.entry(3, "cenedra", ["C"], pinnable=False)]
        self.assertEqual(self.names(before=4, readings={"C", "P"}), ["first", "cenedra"])
        # One saying both still wins.
        self.state.entries.insert(1, self.entry(2, "both", ["C", "P"], pinnable=False))
        self.assertEqual(self.names(before=4, readings={"C", "P"}), ["first", "both"])

    def test_an_example_by_another_voice_is_never_heard(self):
        self.state.entries = [self.entry(1, "first"), self.entry(2, "other voice", ["A"], voices=("c",)),
                              self.entry(3, "plain")]
        self.assertEqual(self.names(before=4, readings={"A"}), ["first", "plain"])

    def test_a_request_names_the_chosen_readings_it_uses(self):
        from epub_tts_narrator.renderer.continuity import _chosen_readings
        chunk = chosen(passages(["They saw Ce’Nedra’s face."])[0], "Ce’Nedra’s", READING + " Z")
        self.assertEqual(_chosen_readings({"kind": "speech", "chunk": chunk}), {READING})
        self.assertEqual(_chosen_readings({"kind": "speech", "chunk": passages(["Plain words."])[0]}), frozenset())



class SilentModel(Model):
    """Its ``silent`` call speaks one second, then goes silent until the watch stops it."""

    def __init__(self, silent):
        super().__init__()
        self.silent = silent

    def generate(self, text, **options):
        import numpy as np
        for result in super().generate(text, **options):
            if len(self.calls) == self.silent:
                index = len(self.calls)
                tone = (0.15 * np.sin(2 * np.pi * (200 + 20 * index) * np.arange(44100) / 44100)).astype(np.float32)
                result = SimpleNamespace(text=result.text, codes=np.full((10, 20), index, dtype=np.int64),
                                         audio=np.concatenate([tone, np.zeros(3 * 44100, dtype=np.float32)]),
                                         finish_reason="silence")
            yield result


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class SilenceStoppedHistoryTests(unittest.TestCase):
    setUp = ContinuityTests.setUp
    run_chapter = ContinuityTests.run_chapter

    def test_a_kept_stopped_take_is_cut_used_and_never_heard_as_history(self):
        import soundfile
        self.model = self.engine._model = SilentModel(silent=2)
        self.verifier = Verifier(self.model)
        self.run_chapter(self.chunks[:3])
        # The third line hears the first, not the take that went silent.
        self.assertEqual(histories(self.model)[2], [tagged("First words.")])
        accepted = [read_json(path) for path in (self.root / "work" / "clips").glob("*/accepted.json")]
        stopped = [record for record in accepted if record["engine_metadata"].get("finish_reason") == "silence"]
        self.assertEqual(len(stopped), 1)
        self.assertTrue(stopped[0]["report"]["passed"])
        self.assertLess(stopped[0]["engine_metadata"]["duration_seconds"], 1.6)

    def test_outside_a_chapter_a_stopped_take_is_still_a_failure(self):
        from epub_tts_narrator.renderer.engines import EngineError
        self.model = self.engine._model = SilentModel(silent=1)
        path = self.root / "preview-take.wav"
        with self.assertRaisesRegex(EngineError, "did not finish the text: 'silence'"):
            self.engine.synthesize("Preview words.", self.voices["v1"], path, seed=1)
        self.assertFalse(path.exists())


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class OtherChapterExampleTests(unittest.TestCase):
    """A chapter's first line with a chosen reading hears another chapter's take saying it right."""

    setUp = ContinuityTests.setUp

    def chapter(self, texts, chapter_id, first_sequence, block):
        chunks = passages(texts, block=block)
        for number, chunk in enumerate(chunks):
            chunk["segment"].update(chapter_id=chapter_id, sequence=first_sequence + number, kind="narration")
            chunk["segment"]["delivery"]["direction"] = ""
            if "Ce’Nedra" in chunk["spoken_text"]:
                chosen(chunk)
        return chunks

    def run(self, result=None):
        from epub_tts_narrator.renderer import reading_examples
        reading_examples._INDEXES.clear()
        return super().run(result)

    def render(self, chunks, number):
        return _render_chapter_chunks({"number": number, "chunks": chunks}, self.voices, self.engine,
            self.engine.identity, self.verifier, self.config, self.root / "work", notify=lambda message: None)

    def test_the_first_mention_in_a_chapter_hears_an_earlier_chapter_saying_it(self):
        self.model = self.engine._model = NameModel()
        self.verifier = Verifier(self.model)
        self.render(self.chapter(["The road ran on.", "Ce’Nedra objected."], "ch1", 1, "b1"), 1)
        self.render(self.chapter(["Garion rode ahead.", "Ce’Nedra laughed."], "ch2", 10, "b2"), 2)
        named = tagged(f"<|phoneme_start|>{READING}<|phoneme_end|> objected.")
        # Chapter two's own first line keeps Fish's history; its name line hears chapter one's take.
        self.assertEqual(histories(self.model)[3], [tagged("Garion rode ahead."), named])

    def test_a_take_heard_rearranged_or_retired_is_never_an_example(self):
        self.model = self.engine._model = NameModel(heard="Sendera")
        self.verifier = Verifier(self.model)
        # A misread take is rejected, so nothing in chapter one says the name right.
        self.config["render"]["max_attempts"] = 1
        with self.assertRaises(Exception):
            self.render(self.chapter(["The road ran on.", "Ce’Nedra objected."], "ch1", 1, "b1"), 1)
        from epub_tts_narrator.renderer.reading_examples import for_work
        self.assertEqual(for_work(self.root / "work").candidates(), [])


class ExampleChoiceTests(unittest.TestCase):
    def test_a_line_with_two_chosen_names_hears_an_example_saying_both(self):
        from pathlib import Path
        from unittest.mock import patch
        from epub_tts_narrator.renderer import common, continuity_codes, name_phonemes
        from epub_tts_narrator.renderer.reading_examples import ReadingExamples

        def item(name, readings, sequence):
            marker = {"file": f"{name}.codes.json", "sha256": name * 8, "text": name, "speakers": ["v"]}
            return {"accepted": Path(f"/fixture/{name}/accepted.json"), "record_audio": name, "request_hash": name,
                    "marker": marker, "readings": frozenset(readings), "voices": ["v"], "sequence": sequence,
                    "chapter": "ch1"}
        examples = ReadingExamples("/fixture")
        examples._candidates = [item("both", {"A", "B"}, 3), item("only", {"A"}, 5)]
        saved = {str(row["accepted"]): {"audio_sha256": row["record_audio"],
                                        "engine_metadata": {"continuity_codes": row["marker"]}}
                 for row in examples._candidates}
        with patch.object(common, "read_json", side_effect=lambda path: saved[str(path)]), \
                patch.object(continuity_codes, "load_codes"), patch.object(name_phonemes, "judge", return_value=None):
            chosen = examples.example({"v"}, {"A"}, sequence=10, chapter="ch2", voices_by_asset={"v": {}},
                                      wanted={"A", "B"})
            nearest = examples.example({"v"}, {"A"}, sequence=10, chapter="ch2", voices_by_asset={"v": {}})
        self.assertEqual(chosen["marker"]["text"], "both")
        # The missing reading still decides first; without other wanted readings the nearest wins.
        self.assertEqual(nearest["marker"]["text"], "only")


class RetiredExampleTests(OtherChapterExampleTests):
    test_the_first_mention_in_a_chapter_hears_an_earlier_chapter_saying_it = None
    test_a_take_heard_rearranged_or_retired_is_never_an_example = None

    def test_an_example_retired_after_the_index_was_read_is_not_heard(self):
        from epub_tts_narrator.renderer.reading_examples import for_work
        self.model = self.engine._model = NameModel()
        self.verifier = Verifier(self.model)
        self.render(self.chapter(["The road ran on.", "Ce’Nedra objected."], "ch1", 1, "b1"), 1)
        [example] = for_work(self.root / "work").candidates()
        example["accepted"].rename(example["accepted"].with_name("accepted.judged-bad-fixture.json"))
        self.render(self.chapter(["Garion rode ahead.", "Ce’Nedra laughed."], "ch2", 10, "b2"), 2)
        self.assertEqual(histories(self.model)[3], [tagged("Garion rode ahead.")])
