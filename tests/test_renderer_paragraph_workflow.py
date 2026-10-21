"""Paragraph rendering keeps speaker provenance and accepted-audio resume behavior."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from audiobook_renderer.bundle import load_bundle
from audiobook_renderer.common import file_hash, read_json
from audiobook_renderer.config import DEFAULTS
from audiobook_renderer.quality import validate_policy
from audiobook_renderer.voices import VoiceStore
from audiobook_renderer.workflow import RenderBlocked, assemble, render
from epub_audiobook.cli import export_bundle
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


class ParagraphFixtureEngine(FixtureEngine):
    """Record each dialogue call as one PCM fixture with its complete transcript."""

    capabilities = {**FixtureEngine.capabilities, "multi_speaker": True}

    def __init__(self):
        super().__init__({})
        self.dialogue_calls = []
        self.interrupt_dialogue_number = None

    def synthesize_dialogue(self, turns, path, *, seed):
        self.dialogue_calls.append({
            "turns": [{"text": turn["text"], "voice_hash": turn["voice"]["asset_hash"],
                       "voice_id": turn["voice"]["voice_id"], "direction": turn["direction"]}
                      for turn in turns],
            "seed": seed,
        })
        if self.interrupt_dialogue_number == len(self.dialogue_calls):
            raise KeyboardInterrupt("fixture paragraph interruption")
        return self._write("".join(turn["text"] for turn in turns), path, seed)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for paragraph integration")
class ParagraphWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "unused-tts")
        self.config["verification"]["model"] = str(self.root / "unused-asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="paragraph-fixture")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE,
            paragraph_synthesis=True, max_paragraph_chars=1000,
        )
        self.config["quality"] = validate_policy({})
        self.engine = ParagraphFixtureEngine()
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.messages = []

    def chapters(self):
        return render(self.bundle, self.config, self.engine, self.verifier,
                      chapters_only=True, notify=self.messages.append)

    def package(self):
        return assemble(self.bundle, self.config, notify=self.messages.append)

    def accepted_clips(self):
        return {path: read_json(path) for path in (self.root / "work" / "clips").glob("*/accepted.json")}

    def grouped_clips(self):
        return {path: record for path, record in self.accepted_clips().items()
                if record["request"]["kind"] == "paragraph_speech"}

    def test_fresh_paragraph_uses_one_performance_and_one_verification_with_source_sync(self):
        first = self.chapters()
        self.assertEqual(first["status"], "chapters_complete")
        # Nine source segments become six performances: two grouped paragraphs
        # (three Mara turns, then narrator/Lena) plus four standalone passages.
        self.assertEqual(len(self.engine.dialogue_calls), 2)
        self.assertEqual(len(self.engine.speech_calls), 4)
        self.assertEqual(len(self.verifier.calls) - len(self.engine.voice_calls), 6)
        multi = [call for call in self.engine.dialogue_calls
                 if len({turn["voice_id"] for turn in call["turns"]}) > 1]
        self.assertEqual(len(multi), 1)
        self.assertEqual([turn["voice_id"] for turn in multi[0]["turns"]], ["voice_mara", "voice_lena"])
        rows = [row for row in self.bundle["chapters"][0]["rows"] if row["block_id"] == "b000004"]
        expected = "".join(row["text"] for row in rows)
        self.assertEqual("".join(turn["text"] for turn in multi[0]["turns"]), expected)
        grouped = self.grouped_clips()
        self.assertEqual(len(grouped), 2)
        path, accepted = next((path, record) for path, record in grouped.items()
                              if record["request"]["chunk"]["segment"]["block_id"] == "b000004")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["request"]["chunk"]["spoken_text"], expected)
        self.assertEqual(len(list(path.parent.glob("attempt-*.json"))), 1)
        self.assertEqual(set(accepted["request"]["voice_assets"]), {"voice_mara", "voice_lena"})

        # The completed chapter restores without TTS, ASR, or re-encoding.
        counts = (len(self.engine.dialogue_calls), len(self.engine.speech_calls), len(self.verifier.calls))
        with patch("audiobook_renderer.workflow.verified_take", side_effect=AssertionError("reverified accepted audio")), \
             patch("audiobook_renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded chapter")):
            resumed = self.chapters()
            self.assertEqual(self.package()["status"], "complete")
        self.assertEqual(first["chapters"], resumed["chapters"])
        self.assertEqual(counts, (len(self.engine.dialogue_calls), len(self.engine.speech_calls), len(self.verifier.calls)))
        sync = read_json(self.root / "output" / "sync.json")
        paragraph = next(row for row in sync["segments"] if row["block_id"] == "b000004")
        self.assertEqual(paragraph["precision"], "paragraph")
        self.assertEqual(paragraph["source_text"], expected)
        self.assertIsNone(paragraph["voice_id"])
        self.assertEqual([part["segment_id"] for part in paragraph["source_segments"]], [row["id"] for row in rows])
        self.assertEqual([part["voice_id"] for part in paragraph["source_segments"]], [row["voice_id"] for row in rows])
        self.assertEqual([part["source_ranges"] for part in paragraph["source_segments"]],
                         [[{"start": row["start"], "end": row["end"]}] for row in rows])
        self.assertEqual("".join(part["spoken_text"] for part in paragraph["source_segments"]), expected)
        self.assertTrue(all("book_start_frame" not in part for part in paragraph["source_segments"]))
        self.assertGreater(paragraph["book_end_frame"], paragraph["book_start_frame"])

    def test_recast_one_character_invalidates_only_its_paragraph(self):
        self.chapters()
        before = self.accepted_clips()
        old_group = next(record for record in before.values()
                         if record["request"]["kind"] == "paragraph_speech"
                         and "voice_lena" in record["request"]["voice_assets"])
        store = VoiceStore(self.config["voices"]["directory"], self.config["voices"]["series_id"])
        old = store.load("voice_lena")
        reference = self.root / "replacement.wav"
        self.engine._write(old["reference_text"], reference, 9182)
        store.publish_candidate("voice_lena", old["profile"], reference, old["reference_text"],
                                self.engine.identity, {"passed": True}, {"fixture": True},
                                candidate_id="a" * 32)
        store.select_candidate("voice_lena", "a" * 32)
        with self.assertRaisesRegex(RenderBlocked, "is stale"):
            self.package()
        counts = (len(self.engine.dialogue_calls), len(self.engine.speech_calls), len(self.verifier.calls))
        self.chapters()
        self.assertEqual(len(self.engine.dialogue_calls) - counts[0], 1)
        self.assertEqual(len(self.engine.speech_calls), counts[1])
        self.assertEqual(len(self.verifier.calls) - counts[2], 1)
        after = self.accepted_clips()
        self.assertEqual(len(after), len(before) + 1)
        self.assertTrue(all(read_json(path) == record for path, record in before.items()))
        new_group = next(record for path, record in after.items() if path not in before)
        self.assertEqual(new_group["request"]["chunk"], old_group["request"]["chunk"])
        self.assertNotEqual(new_group["request"]["voice_assets"]["voice_lena"],
                            old_group["request"]["voice_assets"]["voice_lena"])
        self.assertEqual(new_group["request"]["voice_assets"]["voice_mara"],
                         old_group["request"]["voice_assets"]["voice_mara"])

    def test_accepted_individual_clips_survive_enabling_paragraph_mode(self):
        self.config["render"]["paragraph_synthesis"] = False
        first = self.chapters()
        accepted = self.accepted_clips()
        self.assertEqual(len(accepted), 9)
        self.assertEqual(self.engine.dialogue_calls, [])
        self.config["render"]["paragraph_synthesis"] = True
        with patch("audiobook_renderer.workflow.verified_take", side_effect=AssertionError("regenerated accepted individual clips")), \
             patch("audiobook_renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded unchanged chapter")):
            resumed = self.chapters()
        self.assertEqual(first["chapters"], resumed["chapters"])
        self.assertEqual(self.accepted_clips(), accepted)
        self.assertEqual(self.engine.dialogue_calls, [])

    def test_interruption_reuses_the_complete_accepted_paragraph_on_resume(self):
        self.engine.interrupt_dialogue_number = 2
        with self.assertRaisesRegex(KeyboardInterrupt, "paragraph interruption"):
            self.chapters()
        grouped = self.grouped_clips()
        self.assertEqual(len(grouped), 1)
        path, accepted = next(iter(grouped.items()))
        self.assertEqual(len(accepted["request"]["chunk"]["paragraph_turns"]), 3)
        wav = path.parent / accepted["audio_file"]
        old_hash = file_hash(wav)
        calls = len(self.verifier.calls)
        self.assertFalse((self.root / "work" / ".render.lock").exists())
        self.engine.interrupt_dialogue_number = None
        resumed = self.chapters()
        self.assertEqual(resumed["status"], "chapters_complete")
        self.assertEqual(len(self.engine.dialogue_calls), 3)  # two successes plus interruption
        self.assertEqual(len(self.engine.speech_calls), 4)
        self.assertEqual(len(self.verifier.calls) - calls, 3)  # second paragraph and final two passages
        self.assertEqual(read_json(path), accepted)
        self.assertEqual(file_hash(wav), old_hash)
        self.assertEqual(len(self.grouped_clips()), 2)

    def test_partially_rendered_legacy_paragraph_keeps_its_boundaries(self):
        self.config["render"]["paragraph_synthesis"] = False
        self.engine.interrupt_speech_number = 4
        with self.assertRaises(KeyboardInterrupt):
            self.chapters()
        before = self.accepted_clips()
        self.assertEqual(len(before), 3)
        # The first short quotation in b000003 is accepted. Keep that entire
        # paragraph as individual turns, while grouping the untouched b000004.
        self.assertEqual(sum(record["request"]["chunk"]["segment"]["block_id"] == "b000003"
                             for record in before.values()), 1)
        self.config["render"]["paragraph_synthesis"] = True
        self.engine.interrupt_speech_number = None
        self.chapters()
        self.assertEqual(len(self.engine.dialogue_calls), 1)
        self.assertEqual(len(self.engine.speech_calls), 8)  # seven accepted plus the interrupted call
        self.assertTrue(all(read_json(path) == record for path, record in before.items()))
        accepted = self.accepted_clips()
        self.assertEqual(len(accepted), 8)
        self.assertEqual(sum(record["request"]["kind"] == "speech"
                             and record["request"]["chunk"]["segment"]["block_id"] == "b000003"
                             for record in accepted.values()), 3)
        self.assertEqual(len(self.grouped_clips()), 1)


if __name__ == "__main__":
    unittest.main()
