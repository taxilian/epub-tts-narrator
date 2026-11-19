"""Crowd murmur: proposed from narration, built from verified takes, unintelligible, mixed under a span."""

from array import array
from copy import deepcopy
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.group_proposals import propose_groups
from epub_tts_narrator.renderer import murmur
from test_character_merge import build


def murmur_document():
    document = build("murmur", "Murmur", [("mara", "Mara", ()), ("eunuchs", "eunuch chorus", ())], [
        [("“Welcome,” ", "eunuchs"), ("the chorus replied in unison.", None)],
        [("Behind them the eunuchs knelt, murmuring their praises.", None)],
        [("Mara walked on. ", None), ("“Quickly,” ", "mara"), ("she said.", None)],
        [("The halls were quiet.", None)],
        [("Somewhere a door closed.", None)],
    ])
    return document


class MurmurFormatTests(unittest.TestCase):
    def test_a_murmur_names_a_group_and_a_later_segment_of_the_same_chapter(self):
        document = murmur_document()
        row = next(item for item in document["segments"] if item["id"] == "b2:0")
        row["murmur"] = {"group": "eunuchs", "until": "b3:2", "shape": "fade", "basis": "explicit",
                         "evidence": [{"block_id": "b2", "quote": "murmuring their praises"}]}
        self.assertEqual(validate_document(document), [])
        for change, message in (({"until": "b1:0"}, "later one"), ({"group": "nobody"}, "character"),
                                ({"shape": "roar"}, "shape"), ({"evidence": []}, "evidence"),
                                ({"from_line": "b2:0"}, "spoken segment")):
            broken = deepcopy(document)
            next(item for item in broken["segments"] if item["id"] == "b2:0")["murmur"].update(change)
            with self.subTest(change=change):
                self.assertTrue(any(message in error for error in validate_document(broken)),
                                validate_document(broken))


class MurmurProposalTests(unittest.TestCase):
    def test_narration_that_says_a_groups_voices_keep_sounding_proposes_a_span(self):
        candidate, changes = propose_groups(murmur_document())
        self.assertEqual(validate_document(candidate), [])
        row = next(item for item in candidate["segments"] if item["id"] == "b2:0")
        self.assertEqual((row["murmur"]["group"], row["murmur"]["basis"]), ("eunuchs", "explicit"))
        self.assertEqual(row["murmur"]["evidence"][0]["quote"], "Behind them the eunuchs knelt, murmuring their praises.")
        self.assertIn(row["murmur"]["until"], {"b3:2", "b4:0", "b5:0"})
        self.assertTrue(any("murmur" in change for change in changes))
        again, more = propose_groups(candidate)
        self.assertEqual(more, [], "idempotent")

    def test_a_groups_own_tag_and_a_single_speakers_sound_are_no_murmur(self):
        document = build("murmur", "Murmur", [("eunuchs", "eunuch chorus", ())], [
            [("“Welcome,” ", "eunuchs"), ("the chorus replied in unison.", None)],
            [("“Praise her,” ", "eunuchs"), ("the eunuchs murmured.", None)],
            [("The leader of the eunuchs began chanting.", None)],
            [("Far overhead the trees murmured to each other.", None)]])
        candidate, _ = propose_groups(document)
        self.assertFalse(any("murmur" in row for row in candidate["segments"]))


class MurmurBedTests(unittest.TestCase):
    def test_a_run_of_three_heard_murmur_words_makes_a_bed_intelligible(self):
        texts = ["Keep your voice down, they can hear us."]
        self.assertEqual(murmur.heard_run("mumble your voice down mumble".split(), texts), 3)
        self.assertEqual(murmur.heard_run("mumble keep mumble hear".split(), texts), 1)

    def test_loops_are_deterministic_full_length_and_levelled(self):
        rate = 8000
        tone = array("h", (round(4000 * math.sin(2 * math.pi * 200 * index / rate)) for index in range(rate)))
        loop = murmur.build_loop([[tone], [tone, tone]], "key", rate, 2800.0)
        self.assertEqual(len(loop), int(murmur.LOOP_SECONDS * rate))
        self.assertEqual(loop, murmur.build_loop([[tone], [tone, tone]], "key", rate, 2800.0))
        self.assertNotEqual(loop, murmur.build_loop([[tone], [tone, tone]], "other", rate, 2800.0))

    def test_mixing_under_a_span_changes_no_frames_and_ducks_under_speech(self):
        rate = 8000
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        work = Path(temporary.name)
        chapter = work / "chapter.wav"
        speech = [round(6000 * math.sin(2 * math.pi * 300 * index / rate)) for index in range(rate)]
        samples = array("h", speech + [0] * rate)
        with wave.open(str(chapter), "wb") as stream:
            stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(samples.tobytes())
        folder = work / "murmurs" / "k"
        folder.mkdir(parents=True)
        bed = array("h", [2000] * (rate // 2))
        murmur._write(folder / "murmur-bed.wav", bed, rate)
        from epub_tts_narrator.renderer.stored_audio import audio_hash
        from epub_tts_narrator.renderer.common import write_json
        write_json(folder / "bed.json", {"status": "ok", "audio_file": "murmur-bed.wav",
                                         "audio_sha256": audio_hash(folder / "murmur-bed.wav")})
        segments = [{"segment_id": "a", "chapter_start_frame": 0, "chapter_end_frame": rate},
                    {"segment_id": "b", "chapter_start_frame": rate, "chapter_end_frame": 2 * rate}]
        murmurs = [{"start": "a", "until": "b", "group": "g", "shape": "steady", "level": "low",
                    "bed": {"key": "k", "audio_sha256": audio_hash(folder / "murmur-bed.wav")}}]
        applied = murmur.apply_beds(chapter, segments, murmurs, work, rate)
        self.assertEqual(applied[0]["chapter_end_frame"], 2 * rate)
        with wave.open(str(chapter), "rb") as stream:
            self.assertEqual(stream.getnframes(), 2 * rate)
            mixed = array("h", stream.readframes(stream.getnframes()))
        under_speech = abs(mixed[rate // 2] - samples[rate // 2])
        in_silence = abs(mixed[rate + rate // 2])
        self.assertGreater(in_silence, 0, "the murmur is heard where nobody speaks")
        self.assertLess(under_speech, in_silence, "and ducks under speech")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
class MurmurWorkflowTests(unittest.TestCase):
    def test_a_cast_groups_murmur_is_recorded_verified_and_mixed_under_its_span(self):
        from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
        from epub_tts_narrator.renderer.common import read_json
        from epub_tts_narrator.renderer.workflow import render
        from audition_fixtures import prepare_bundle
        from test_renderer_ensembles import EnsembleWorkflowTests, ensemble_bundle
        from test_renderer_simultaneous import LaneEngine, LaneVerifier
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bundle = ensemble_bundle(root, {"members": ["voice_chorus_1", "voice_chorus_2"], "mode": "unison",
                                        "basis": "user"})
        tag = next(row for chapter in bundle["chapters"] for row in chapter["rows"]
                   if row["text"] == " the chorus said in unison.")
        tag["murmur"] = {"group": "chorus", "until": tag["id"], "shape": "steady", "basis": "user"}
        config = EnsembleWorkflowTests.config(type("Fixture", (), {"root": root})())

        class QuietVerifier(LaneVerifier):
            def transcribe(self, path):
                if "murmur-bed" in str(path):
                    return {"text": "mm hm", "segments": []}
                return super().transcribe(path)
        engine = LaneEngine()
        plan = make_plan(bundle, config)
        self.assertEqual(plan["collective_voices"]["murmurs"], 1)
        render(bundle, config, engine, QuietVerifier(engine), notify=lambda _: None)
        beds = list((root / "work" / "murmurs").iterdir())
        self.assertEqual(len(beds), 1)
        self.assertEqual(read_json(beds[0] / "bed.json")["status"], "ok")
        murmur_lines = {call["text"] for call in engine.speech_calls} & set(murmur.SENTENCES)
        self.assertTrue(murmur_lines, "the group's voices recorded murmur sentences")
        composed = [read_json(path) for path in (root / "work" / "chapters").glob("*/chapter.json")]
        mixed = [record for record in composed if record["identity"].get("murmurs")]
        self.assertEqual(len(mixed), 1, "the chapter was composed with its bed")
        self.assertEqual(mixed[0]["identity"]["murmurs"][0]["bed"]["key"], beds[0].name)


if __name__ == "__main__":
    unittest.main()
