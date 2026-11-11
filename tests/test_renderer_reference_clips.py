"""Several clips of one designed voice become its joined, verified reference."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer import reference_clips, workflow
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
from shared_fixtures import pcm_hash
from test_renderer_automatic_gemini_cleanup import FixtureGemini
from test_renderer_samples import bundle as sample_bundle, row
from test_renderer_voice_design_integration import (
    FFMPEG, FFPROBE, ReferencePlaybackEngine, TranscriptVerifier,
)


GEMINI = {"engine": "gemini_voice_design", "model": "fixture"}


def write_tone(path, seconds, rate=24000, value=1000):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(value.to_bytes(2, "little", signed=True) * int(rate * seconds))
    return Path(path)


def speech_bundle(lines=9):
    text = "I have seen the road ahead and it is long, but we will walk it together before the snow arrives here."
    return sample_bundle([row(text.replace("road", f"road {number}"), ident=f"s{number}", sequence=number)
                          for number in range(lines)])


class PlanTests(unittest.TestCase):
    def test_extra_scripts_avoid_the_first_clip_and_each_other(self):
        data = speech_bundle()
        scripts = reference_clips.plan_scripts({"casting": {"reference_clips": 3}}, GEMINI, data, "v1",
                                               lambda value: value, {"segment_ids": ["s0", "s1"]})
        self.assertEqual(len(scripts), 2)
        segments = [segment for script in scripts for segment in script["source"]["segment_ids"]]
        self.assertEqual(len(segments), len(set(segments)))
        self.assertFalse({"s0", "s1"} & set(segments))

    def test_no_extra_scripts_for_one_clip_another_designer_or_too_little_speech(self):
        data = speech_bundle()
        cases = (({"casting": {"reference_clips": 1}}, GEMINI, data),
                 ({"casting": {"reference_clips": 3}}, {"engine": "elevenlabs_voice_design"}, data),
                 ({"casting": {"reference_clips": 3}}, GEMINI, speech_bundle(lines=1)),
                 ({"casting": {"reference_clips": 3}}, GEMINI, {"manifest": {}}))
        for config, designer, bundle in cases:
            with self.subTest(designer=designer["engine"], clips=config["casting"]["reference_clips"]):
                self.assertEqual(reference_clips.plan_scripts(config, designer, bundle, "v1", lambda value: value,
                                                              {"segment_ids": ["s0"]}), [])


class JoinTests(unittest.TestCase):
    def test_clips_join_in_order_with_half_second_pauses(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [write_tone(Path(temp) / f"clip-{index}.wav", index, value=100 * index) for index in (1, 2)]
            joined = reference_clips.join_wavs(paths, Path(temp) / "out" / "joined.wav")
            with wave.open(str(joined), "rb") as audio:
                frames = audio.readframes(audio.getnframes())
                rate = audio.getframerate()
        samples = [int.from_bytes(frames[index:index + 2], "little", signed=True) for index in range(0, len(frames), 2)]
        self.assertEqual(len(samples), rate * 3 + rate // 2)
        self.assertEqual((samples[0], samples[rate + rate // 4], samples[-1]), (100, 0, 200))

    def test_clips_of_different_rates_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [write_tone(Path(temp) / "a.wav", 1), write_tone(Path(temp) / "b.wav", 1, rate=44100)]
            with self.assertRaisesRegex(ValueError, "one sample format and rate"):
                reference_clips.join_wavs(paths, Path(temp) / "joined.wav")


class AddClipsTests(unittest.TestCase):
    """The clip loop with generation and verification stood in by ``verified_take``."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.first = write_tone(self.directory / "first.wav", 1)
        self.first_accepted = {"report": {"passed": True, "reference_text": "first words"},
                               "engine_metadata": {"description_applied": True}, "request_hash": "h1"}
        self.request = {"voice_id": "v1", "text": "first words", "sample_source": {"kind": "user"}}
        self.scripts = [{"text": "second words", "source": {"segment_ids": ["s2"]}},
                        {"text": "third words", "source": {"segment_ids": ["s3"]}}]
        self.failing = set()
        self.calls = []

    def fake_take(self, text, generate, directory, request, verifier, config, *, notify, purpose):
        self.calls.append((text, config["render"]["max_attempts"]))
        if text in self.failing:
            raise SpeechVerificationFailed("failed", request=request, report={})
        Path(directory).mkdir(parents=True, exist_ok=True)
        wav = Path(directory) / "accepted-take.wav"
        if "joined_clips" in request:
            generate(wav, 1)
        else:
            write_tone(wav, 1)
        return wav, {"report": {"passed": True, "reference_text": text}, "request_hash": "h-" + text,
                     "engine_metadata": {"description_applied": True}}

    def add(self):
        with patch.object(reference_clips, "verified_take", self.fake_take):
            return reference_clips.add_clips(
                self.first, self.first_accepted, self.scripts, designer=None, prompt="p", directory=self.directory,
                request=self.request, verifier=None, config={"render": {"max_attempts": 4}}, label="Mara",
                notify=lambda message: None)

    def test_every_clip_is_joined_and_the_joined_recording_is_verified_once(self):
        wav, accepted, clips = self.add()
        self.assertEqual(accepted["report"]["reference_text"], "first words second words third words")
        self.assertEqual([clip["index"] for clip in clips], [1, 2, 3])
        self.assertEqual(self.calls[-1], ("first words second words third words", 1))
        with wave.open(str(wav), "rb") as audio:
            self.assertEqual(audio.getnframes(), 3 * 24000 + 2 * 12000)

    def test_an_extra_clip_that_fails_verification_is_left_out(self):
        self.failing.add("second words")
        _wav, accepted, clips = self.add()
        self.assertEqual([clip["index"] for clip in clips], [1, 3])
        self.assertEqual(accepted["report"]["reference_text"], "first words third words")

    def test_a_joined_recording_that_fails_verification_leaves_the_first_clip_alone(self):
        self.failing.add("first words second words third words")
        wav, accepted, clips = self.add()
        self.assertEqual((wav, accepted, clips), (self.first, self.first_accepted, []))

    def test_missing_clips_after_the_hosted_voice_was_removed_need_a_fresh_voice(self):
        (self.directory / "gemini-voice-cleanup.json").write_text("{}")
        with self.assertRaises(reference_clips.FreshVoiceRequired):
            self.add()
        self.assertEqual(self.calls, [])


@unittest.skipUnless(FFMPEG, "FFmpeg is required for reference verification")
class AutomaticCreationTests(unittest.TestCase):
    """Render-time voice creation with the real Gemini request-state and cleanup lifecycle."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = deepcopy(DEFAULTS)
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="fixture-series")
        self.config["render"].update(max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.config["casting"]["reference_clips"] = 3
        self.work = self.root / "work"
        self.transcripts = {}
        self.engine = ReferencePlaybackEngine(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        self.remote = FixtureGemini(self.transcripts)
        self.store = VoiceStore(self.root / "voices", "fixture-series")
        self.bundle = speech_bundle()
        self.profile = {"id": "v1", "description": "A warm, clear adult voice.", "traits": []}
        original = reference_clips.join_wavs

        def join_and_register(paths, output, **kwargs):
            joined = original(paths, output, **kwargs)
            self.transcripts[pcm_hash(joined)] = " ".join(self.transcripts[pcm_hash(path)] for path in paths)
            return joined
        joining = patch.object(reference_clips, "join_wavs", side_effect=join_and_register)
        joining.start()
        self.addCleanup(joining.stop)

    def ensure(self):
        lock = self.store.lock()
        try:
            return workflow.ensure_voice(self.store, "v1", self.profile, self.engine, self.verifier,
                                         self.config, self.work, designer=self.remote, notify=lambda message: None,
                                         bundle=self.bundle)
        finally:
            lock.rmdir()

    def test_render_time_creation_saves_clips_of_one_hosted_voice_joined(self):
        selected = self.ensure()
        clips = selected["creation"]["reference_clips"]
        self.assertGreaterEqual(len(clips), 2)
        self.assertEqual(self.remote.events, [("create", "voice_fixture_1")]
                         + [("synthesize", "voice_fixture_1")] * len(clips) + [("delete", "voice_fixture_1")])
        self.assertEqual(selected["reference_text"], " ".join(clip["reference_text"] for clip in clips))

    def test_an_interruption_between_clips_starts_again_with_one_fresh_voice(self):
        original, made = self.remote.create_voice, []

        def interrupted(prompt, text, path, *, seed):
            made.append(text)
            if len(made) == 2:
                raise KeyboardInterrupt
            return original(prompt, text, path, seed=seed)
        with patch.object(self.remote, "create_voice", interrupted), self.assertRaises(KeyboardInterrupt):
            self.ensure()
        self.assertIn(("delete", "voice_fixture_1"), self.remote.events)
        self.remote.events.clear()
        selected = self.ensure()
        clips = selected["creation"]["reference_clips"]
        # Every saved clip comes from the one fresh voice; none from the deleted one.
        self.assertEqual(self.remote.events, [("create", "voice_fixture_2")]
                         + [("synthesize", "voice_fixture_2")] * len(clips) + [("delete", "voice_fixture_2")])
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_fixture_2")


if __name__ == "__main__":
    unittest.main()
