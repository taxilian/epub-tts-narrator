"""The one-vs-several reference clip experiment keeps its controls honest offline."""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FakeReference:
    """The vendored ``Reference`` shape: a dataclass of transcript and codes."""

    text: str
    codes: str


class FakeEngine:
    """Drama 3's boundary: records each request's references and seed."""

    identity = {"engine": "drama3_mlx", "fake": True}

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def _load(self):
        return SimpleNamespace(sample_rate=44100)

    def _reference(self, voice):
        return FakeReference(text=voice["reference_text"], codes=voice["reference_sha256"])

    def _directing_segment(self, model, text, references, previous, seed):
        self.calls.append({"text": text, "references": [item.text for item in references],
                           "previous": previous, "seed": seed})
        if self.fail_first:
            self.fail_first -= 1
            return None
        return SimpleNamespace(finish_reason="eos", prompt_rows=100 * len(references), text=text)

    def _save_segment(self, segment, model, expected, output):
        output.write_bytes(expected.encode())


class RenderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.plan = experiment.make_plan(speech_bundle(), "v1", same, clips=3, lines=2,
                                         shared_text="Hello there, friend.", description="Female.")
        voice = {"kind": "reference", "reference_audio": "/a.wav"}
        self.references = experiment.condition_references(
            dict(voice, reference_text="saved", reference_sha256="s"),
            [dict(voice, index=index, reference_text=f"clip {index}", reference_sha256=str(index))
             for index in (1, 2, 3)])

    def render(self, engine):
        return experiment.render_voice(engine, self.plan, self.folder, self.references, seed_base=5,
                                       sampling={}, notify=lambda message: None)

    def test_each_line_uses_one_seed_in_every_condition_with_only_the_references_changing(self):
        engine = FakeEngine()
        self.render(engine)
        self.assertEqual(len(engine.calls), 3 * len(self.plan["lines"]))
        by_text = {}
        for call in engine.calls:
            by_text.setdefault(call["text"], []).append(call)
            self.assertEqual(call["previous"], [])
            self.assertTrue(call["text"].startswith("<|speaker:0|> "))
            self.assertTrue(all(item.startswith("<|speaker:0|> ") for item in call["references"]))
        for calls in by_text.values():
            self.assertEqual(len({call["seed"] for call in calls}), 1)
            self.assertEqual([len(call["references"]) for call in calls], [1, 1, 3])
        page = experiment.write_index(self.folder, [{"plan": self.plan, "folder": str(self.folder),
                                                     "saved": self.references["saved"][0], "clips": [
                                                         dict(item, index=i) for i, item in
                                                         enumerate(self.references["all"], 1)]}])
        text = page.read_text()
        self.assertIn("renders/all/", text)
        self.assertNotIn(str(self.folder), text)

    def test_finished_renders_are_reused_and_renders_from_other_inputs_are_refused(self):
        self.render(FakeEngine())
        resumed = FakeEngine()
        self.render(resumed)
        self.assertEqual(resumed.calls, [])
        self.references["all"] = self.references["all"][:2]
        with self.assertRaisesRegex(ValueError, "made from other inputs"):
            self.render(FakeEngine())

    def test_a_take_without_audio_retries_with_the_next_seed_and_records_it(self):
        engine = FakeEngine(fail_first=1)
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
