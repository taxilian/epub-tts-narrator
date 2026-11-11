"""The one-vs-several reference clip experiment keeps its controls honest offline."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_renderer_samples import bundle as sample_bundle, row

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import compare_reference_clips as experiment
finally:
    sys.path.remove(str(SCRIPTS))

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed


def sentence(number, words=14):
    filler = " ".join(f"word{number}x{index}" for index in range(words - 2))
    return f"Line {number} {filler}."


def speech_bundle(lines=12, words=20, chapters=3):
    rows = [row(sentence(number, words), ident=f"s{number}", sequence=number) for number in range(lines)]
    data = sample_bundle(rows)
    data["chapters"] = [{"entry": {"title": f"Chapter {index + 1}", "path": f"scripts/chapter-{index + 1:04d}.jsonl"},
                         "rows": [item for position, item in enumerate(rows) if position % chapters == index]}
                        for index in range(chapters)]
    for chapter_index, chapter in enumerate(data["chapters"]):
        for item in chapter["rows"]:
            item["chapter_id"] = f"ch{chapter_index + 1}"
    data["cast"]["characters"] = [{"id": "c1", "name": "Mara", "aliases": ["The Smith"], "voice_id": "v1"},
                                  {"id": "c2", "name": "Tobin", "aliases": [], "voice_id": "v2"}]
    data["cast"]["roles"].append({"id": "narrator_main", "voice_id": "vn", "kind": "narration"})
    return data


def same(value):
    return value


class SelectionTests(unittest.TestCase):
    def test_clips_never_share_a_segment_and_test_lines_avoid_every_clip(self):
        plan = experiment.make_plan(speech_bundle(), "v1", same, clips=3, lines=3,
                                    shared_text="All of us say this.", description="Female, warm.")
        segments = [set(script["source"]["segment_ids"]) for script in plan["scripts"]]
        self.assertEqual(len(segments), 3)
        self.assertFalse(segments[0] & segments[1] or segments[0] & segments[2] or segments[1] & segments[2])
        used = set().union(*segments)
        own = [line for line in plan["lines"] if not line.get("shared")]
        self.assertEqual(len(own), 3)
        self.assertFalse({line["segment_id"] for line in own} & used)
        self.assertEqual(plan["lines"][0], {"text": "All of us say this.", "shared": True})

    def test_too_little_book_speech_for_every_clip_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, r"Mara has book speech for only \d separate audition script"):
            experiment.make_plan(speech_bundle(lines=4), "v1", same, clips=3, lines=0,
                                 shared_text="Hello there.", description="Female.")

    def test_voices_resolve_by_id_name_alias_and_narrator_but_never_by_guess(self):
        data = speech_bundle()
        for selector, expected in (("v2", "v2"), ("mara", "v1"), ("the smith", "v1"), ("Narrator", "vn")):
            with self.subTest(selector=selector):
                self.assertEqual(experiment.resolve_voice(data, selector), expected)
        with self.assertRaisesRegex(ValueError, "No cast voice"):
            experiment.resolve_voice(data, "Mar")
        data["cast"]["characters"].append({"id": "c3", "name": "Mara", "aliases": [], "voice_id": "v2"})
        with self.assertRaisesRegex(ValueError, "more than one voice"):
            experiment.resolve_voice(data, "Mara")


class FakeDesigner:
    """Gemini's boundary: one hosted voice per output folder, written as a tiny WAV."""

    def __init__(self):
        self.folders, self.released = [], []
        self.identity = {"engine": "gemini_voice_design", "model": "fake"}

    def create_voice(self, description, text, output_path, *, seed):
        self.folders.append(Path(output_path).parent)
        Path(output_path).write_bytes(b"RIFF" + text.encode())
        return {"description_applied": True, "reference_text": text}

    def release_voice(self, directory):
        self.released.append(Path(directory))
        return []


def fake_verified_take(text, generate, directory, request, verifier, config, *, notify, purpose):
    """Stands in for generation plus verification: one take, accepted as spoken."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    wav = Path(directory) / f"clip-{request['clip']}-accepted.wav"
    if not wav.exists():
        generate(wav, 7)
    return wav, {"report": {"passed": True, "reference_text": text}}


class ClipTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / "mara"
        self.folder.mkdir()
        self.plan = experiment.make_plan(speech_bundle(), "v1", same, clips=3, lines=1,
                                         shared_text="Hello.", description="Female, warm, low.")
        self.designer = FakeDesigner()
        self.session = SimpleNamespace(_get_designer=lambda: self.designer, _get_verifier=lambda: object(),
                                       config={"render": {"seed": 11}})

    def prepare(self):
        return experiment.prepare_clips(self.session, self.plan, self.folder, notify=lambda message: None)

    def test_every_clip_comes_from_one_hosted_voice_which_is_deleted_when_all_are_verified(self):
        with patch("epub_tts_narrator.renderer.workflow.verified_take", fake_verified_take):
            clips = self.prepare()
        self.assertEqual([clip["index"] for clip in clips], [1, 2, 3])
        self.assertEqual(set(self.designer.folders), {self.folder / "hosted-voice"})
        self.assertEqual(self.designer.released, [self.folder / "hosted-voice"])
        for clip, script in zip(clips, self.plan["scripts"]):
            self.assertEqual(clip["reference_text"], script["text"])
            self.assertEqual(clip["reference_sha256"], file_hash(clip["reference_audio"]))
        self.assertEqual(list((self.folder / "hosted-voice").iterdir()), [])

    def test_saved_clips_are_reused_without_the_designer_and_refused_once_changed(self):
        with patch("epub_tts_narrator.renderer.workflow.verified_take", fake_verified_take):
            first = self.prepare()
        self.session._get_designer = lambda: self.fail("saved clips need no designer")
        self.assertEqual(self.prepare(), first)
        Path(first[1]["reference_audio"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Saved experiment clip changed"):
            self.prepare()

    def test_a_clip_that_fails_verification_deletes_the_hosted_voice_but_an_interruption_keeps_it(self):
        def failing(text, generate, directory, request, *args, **kwargs):
            if request["clip"] == 2:
                raise SpeechVerificationFailed("bad", request=request, report={})
            return fake_verified_take(text, generate, directory, request, *args, **kwargs)

        with patch("epub_tts_narrator.renderer.workflow.verified_take", failing), \
                self.assertRaises(SpeechVerificationFailed):
            self.prepare()
        self.assertEqual(self.designer.released, [self.folder / "hosted-voice"])
        self.assertFalse((self.folder / "clips.json").exists())

        self.designer.released.clear()

        def interrupted(text, generate, directory, request, *args, **kwargs):
            if request["clip"] == 3:
                raise KeyboardInterrupt
            return fake_verified_take(text, generate, directory, request, *args, **kwargs)

        with patch("epub_tts_narrator.renderer.workflow.verified_take", interrupted), \
                self.assertRaises(KeyboardInterrupt):
            self.prepare()
        self.assertEqual(self.designer.released, [])

    def test_a_designer_without_one_lasting_hosted_voice_is_refused_before_any_request(self):
        self.designer.identity = {"engine": "elevenlabs_voice_design"}
        with self.assertRaisesRegex(ValueError, "needs the Gemini voice designer"):
            self.prepare()
        self.assertEqual(self.designer.folders, [])


class FakeServeClient:
    """The serve client's boundary: records each request's reference and seed."""

    identity = {"engine": "fish_http", "fake": True}

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def synthesize(self, text, voice, output_path, *, seed):
        self.calls.append({"text": text, "reference": voice["reference_sha256"], "seed": seed})
        if self.fail_first:
            self.fail_first -= 1
            raise EngineError("Drama 3 returned no audio frames")
        Path(output_path).write_bytes(text.encode())
        return {"finish_reason": "eos", "codes": [[1, 2]], "history_turns": 0}


def write_tone(path, seconds, rate=24000, value=0.25):
    import numpy as np
    import soundfile
    soundfile.write(path, np.full(int(rate * seconds), value, dtype=np.float32), rate, subtype="PCM_16")
    return {"reference_audio": str(path), "reference_text": f" words of {Path(path).stem} ",
            "reference_sha256": file_hash(path)}


class JoinTests(unittest.TestCase):
    def test_clips_join_in_order_with_half_second_pauses_and_their_transcripts_in_order(self):
        import soundfile
        with tempfile.TemporaryDirectory() as temp:
            clips = [write_tone(Path(temp) / f"clip-{index}.wav", index, value=0.1 * index) for index in (1, 2, 3)]
            joined = experiment.join_clips(clips, temp, "Mara")
            audio, rate = soundfile.read(joined["reference_audio"], dtype="float32")
            self.assertEqual(len(audio), rate * (1 + 2 + 3) + 2 * rate // 2)
            self.assertAlmostEqual(float(audio[rate + rate // 4]), 0.0)
            self.assertAlmostEqual(float(audio[-1]), 0.3, places=3)
            self.assertEqual(joined["reference_text"], "words of clip-1 words of clip-2 words of clip-3")
            self.assertIn("Mara", Path(joined["reference_audio"]).name)
            self.assertEqual(soundfile.info(joined["reference_audio"]).subtype, "PCM_16")
            again = experiment.join_clips(clips, temp, "Mara")
            self.assertEqual(again, joined)

    def test_clips_of_different_sample_rates_are_refused_instead_of_joined(self):
        with tempfile.TemporaryDirectory() as temp:
            clips = [write_tone(Path(temp) / "clip-1.wav", 1), write_tone(Path(temp) / "clip-2.wav", 1, rate=44100)]
            with self.assertRaisesRegex(ValueError, "one sample rate"):
                experiment.join_clips(clips, temp, "Mara")


class RenderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.plan = experiment.make_plan(speech_bundle(), "v1", same, clips=3, lines=2,
                                         shared_text="Hello there, friend.", description="Female.")
        voice = {"kind": "reference", "reference_audio": "/a.wav"}
        self.clips = [dict(voice, index=index, reference_text=f"clip {index}", reference_sha256=str(index))
                      for index in (1, 2, 3)]
        self.references = experiment.condition_references(
            dict(voice, reference_text="saved", reference_sha256="s"), self.clips,
            dict(voice, reference_text="clip 1 clip 2 clip 3", reference_sha256="joined"))

    def render(self, engine):
        return experiment.render_voice(engine, self.plan, self.folder, self.references, seed_base=5,
                                       sampling={}, notify=lambda message: None)

    def test_each_line_uses_one_seed_in_every_condition_with_only_the_reference_changing(self):
        engine = FakeServeClient()
        results = self.render(engine)
        self.assertEqual(len(engine.calls), 3 * len(self.plan["lines"]))
        by_text = {}
        for call in engine.calls:
            by_text.setdefault(call["text"], []).append(call)
        for calls in by_text.values():
            self.assertEqual(len({call["seed"] for call in calls}), 1)
            self.assertEqual([call["reference"] for call in calls], ["s", "1", "joined"])
        # Codes and other bulky metadata are not kept in the render record.
        self.assertNotIn("codes", results[("all", 1)])
        self.assertEqual(results[("all", 1)]["finish_reason"], "eos")
        page = experiment.write_index(self.folder, [{"plan": self.plan, "folder": str(self.folder),
                                                     "saved": self.references["saved"], "clips": self.clips}])
        text = page.read_text()
        self.assertIn("renders/all/", text)
        self.assertNotIn(str(self.folder), text)

    def test_finished_renders_are_reused_and_renders_from_other_inputs_are_refused(self):
        self.render(FakeServeClient())
        resumed = FakeServeClient()
        self.render(resumed)
        self.assertEqual(resumed.calls, [])
        self.references["all"] = dict(self.references["all"], reference_sha256="other")
        with self.assertRaisesRegex(ValueError, "made from other inputs"):
            self.render(FakeServeClient())

    def test_a_failed_take_retries_with_the_next_seed_and_records_it(self):
        engine = FakeServeClient(fail_first=1)
        results = self.render(engine)
        self.assertEqual(engine.calls[1]["seed"], engine.calls[0]["seed"] + 1)
        record = read_json(next((self.folder / "renders" / "saved").glob("*line-01*.json")))
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(results[("saved", 1)]["attempt"], 2)


class ScoreTests(unittest.TestCase):
    def test_fidelity_compares_each_condition_with_its_own_designer_audio_and_separation_uses_the_shared_line(self):
        with tempfile.TemporaryDirectory() as temp:
            voices, vectors = [], {}
            for number, axis in ((1, 0), (2, 1)):
                plan = {"name": f"Voice {number}", "lines": [{"text": "Shared.", "shared": True}]}
                folder = Path(temp) / f"voice-{number}"
                saved = folder / "saved.wav"
                clips = [folder / f"clip-{index}.wav" for index in (1, 2)]
                vectors[str(saved)] = [1.0 if axis == 0 else 0.0, 1.0 if axis == 1 else 0.0, 0.0]
                for clip in clips:
                    vectors[str(clip)] = vectors[str(saved)]
                for condition in experiment.CONDITIONS:
                    wav = folder / "renders" / condition / experiment.render_name(plan, condition, 1)
                    # Renders converge on a shared third axis, more without several clips.
                    weight = 0.5 if condition == "all" else 1.0
                    base = vectors[str(saved)]
                    vectors[str(wav)] = [base[0], base[1], weight]
                voices.append({"plan": plan, "folder": str(folder), "saved": {"reference_audio": str(saved)},
                               "clips": [{"reference_audio": str(clip)} for clip in clips]})
            scores = experiment.summarize_scores(voices, lambda path: vectors[str(path)])
        self.assertAlmostEqual(scores["separation"]["designer"], 0.0)
        self.assertGreater(scores["separation"]["one"], scores["separation"]["all"])
        self.assertGreater(scores["fidelity"]["Voice 1"]["all"], scores["fidelity"]["Voice 1"]["one"])


if __name__ == "__main__":
    unittest.main()
