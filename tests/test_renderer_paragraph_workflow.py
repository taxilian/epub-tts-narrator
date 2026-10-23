"""Paragraph rendering keeps speaker provenance and accepted-audio resume behavior."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import digest, file_hash, read_json
from epub_tts_narrator.renderer.paragraphs import group_paragraphs
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import (
    RenderBlocked, _render_chapter_chunks, assemble, make_render_plan, render,
)
from epub_tts_narrator.cli import export_bundle
from shared_fixtures import renderer_config
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
        # Exercise a legitimate same-performer group. Role/voice changes are
        # covered separately; a narrator portraying the same character must
        # retain a separate request even when their saved voice is shared.
        for row in self.bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        self.config = renderer_config(self.root, series_id="paragraph-fixture",
                                      paragraph_synthesis=True, max_paragraph_chars=1000)
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

    def test_single_voice_paragraph_backend_retains_groups_and_resume(self):
        original_plan = make_render_plan(self.bundle, self.config)
        self.engine.capabilities = {**self.engine.capabilities, "multi_speaker": False,
                                    "single_speaker_paragraph": True}
        restricted_plan = make_render_plan(self.bundle, self.config, multi_speaker=False,
                                          single_speaker_paragraph=True)
        self.assertEqual(original_plan, restricted_plan)
        first = self.chapters()
        self.assertTrue(self.grouped_clips())
        calls = len(self.engine.dialogue_calls), len(self.engine.speech_calls)
        with patch.object(self.engine, "synthesize_dialogue", side_effect=AssertionError("regenerated paragraph")), \
             patch.object(self.engine, "synthesize", side_effect=AssertionError("regenerated speech")):
            resumed = self.chapters()
        self.assertEqual(first["chapters"], resumed["chapters"])
        self.assertEqual(calls, (len(self.engine.dialogue_calls), len(self.engine.speech_calls)))

    def test_fresh_same_voice_paragraph_groups_but_mixed_voices_remain_separate(self):
        first = self.chapters()
        self.assertEqual(first["status"], "chapters_complete")
        # Three Mara turns form one performance. The narrator/Lena paragraph
        # stays separate so a model cannot ignore its speaker-change tag.
        self.assertEqual(len(self.engine.dialogue_calls), 1)
        self.assertEqual(len(self.engine.speech_calls), 6)
        self.assertEqual(len(self.verifier.calls) - len(self.engine.voice_calls), 7)
        multi = [call for call in self.engine.dialogue_calls
                 if len({turn["voice_id"] for turn in call["turns"]}) > 1]
        self.assertEqual(multi, [])
        rows = [row for row in self.bundle["chapters"][0]["rows"] if row["block_id"] == "b000003"]
        expected = "".join(row["text"] for row in rows)
        self.assertEqual("".join(turn["text"] for turn in self.engine.dialogue_calls[0]["turns"]), expected)
        grouped = self.grouped_clips()
        self.assertEqual(len(grouped), 1)
        path, accepted = next((path, record) for path, record in grouped.items()
                              if record["request"]["chunk"]["segment"]["block_id"] == "b000003")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["request"]["chunk"]["spoken_text"], expected)
        self.assertEqual(len(list(path.parent.glob("attempt-*.json"))), 1)
        self.assertEqual(set(accepted["request"]["voice_assets"]), {"voice_mara"})

        # The completed chapter restores without TTS, ASR, or re-encoding.
        counts = (len(self.engine.dialogue_calls), len(self.engine.speech_calls), len(self.verifier.calls))
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("reverified accepted audio")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded chapter")):
            resumed = self.chapters()
            self.assertEqual(self.package()["status"], "complete")
        self.assertEqual(first["chapters"], resumed["chapters"])
        self.assertEqual(counts, (len(self.engine.dialogue_calls), len(self.engine.speech_calls), len(self.verifier.calls)))
        sync = read_json(self.root / "output" / "sync.json")
        paragraph = next(row for row in sync["segments"] if row["block_id"] == "b000003")
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
        mixed = [row for row in sync["segments"] if row["block_id"] == "b000004"]
        self.assertEqual([row["voice_id"] for row in mixed], ["voice_mara", "voice_lena"])
        self.assertTrue(all(row.get("precision") != "paragraph" for row in mixed))

    def test_recast_one_character_invalidates_only_its_own_passage(self):
        self.chapters()
        before = self.accepted_clips()
        old_clip = next(record for record in before.values()
                        if record["request"]["chunk"]["segment"]["voice_id"] == "voice_lena")
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
        self.assertEqual(len(self.engine.dialogue_calls), counts[0])
        self.assertEqual(len(self.engine.speech_calls) - counts[1], 1)
        self.assertEqual(len(self.verifier.calls) - counts[2], 1)
        after = self.accepted_clips()
        self.assertEqual(len(after), len(before) + 1)
        self.assertTrue(all(read_json(path) == record for path, record in before.items()))
        new_clip = next(record for path, record in after.items() if path not in before)
        self.assertEqual(new_clip["request"]["chunk"], old_clip["request"]["chunk"])
        self.assertNotEqual(new_clip["request"]["voice_asset_hash"],
                            old_clip["request"]["voice_asset_hash"])

    def test_dialogue_narrator_dialogue_uses_three_separate_voice_requests(self):
        from test_renderer_paragraphs import passages, plan
        members = passages(['"Wow, this is amazing", ', 'she said, ',
                            '"Why should we do that again?"'],
                           voices=["voice_lena", "voice_mara", "voice_lena"])
        grouped = group_paragraphs(plan(members))["chapters"][0]["chunks"]
        chapter = dict(make_render_plan(self.bundle, self.config)["chapters"][0], chunks=grouped)
        voices = {voice_id: {"voice_id": voice_id, "asset_hash": digest(voice_id)}
                  for voice_id in ("voice_lena", "voice_mara")}
        clips, accepted = _render_chapter_chunks(
            chapter, voices, self.engine, self.engine.identity, self.verifier,
            self.config, self.root / "work", notify=self.messages.append,
        )
        self.assertEqual(self.engine.dialogue_calls, [])
        self.assertEqual([call["text"] for call in self.engine.speech_calls],
                         ['Wow, this is amazing",', 'she said,', 'Why should we do that again?'])
        self.assertEqual([call["voice_hash"] for call in self.engine.speech_calls],
                         [digest("voice_lena"), digest("voice_mara"), digest("voice_lena")])
        self.assertEqual(len(accepted), 3)
        self.assertEqual([clip["source_ranges"] for clip in clips],
                         [member["source_ranges"] for member in members])

    def test_speech_trims_only_edges_and_retains_source_and_verification_records(self):
        from test_renderer_paragraphs import passages
        source = ' \t“ ‘He said, "Don’t go," and waited.’ ”\n'
        members = passages([source], voices=["voice_lena"], offset=17)
        original = deepcopy(members)
        chapter = dict(make_render_plan(self.bundle, self.config)["chapters"][0], chunks=members)
        voices = {"voice_lena": {"voice_id": "voice_lena", "asset_hash": "lena"}}

        clips, accepted = _render_chapter_chunks(
            chapter, voices, self.engine, self.engine.identity, self.verifier,
            self.config, self.root / "work", notify=self.messages.append,
        )

        self.assertEqual(self.engine.speech_calls[0]["text"], 'He said, "Don’t go," and waited.')
        self.assertEqual(members, original)
        self.assertEqual(clips[0]["source_ranges"], [{"start": 17, "end": 17 + len(source)}])
        self.assertEqual(clips[0]["spoken_text"], source)
        record = next(iter(self.accepted_clips().values()))
        self.assertEqual(record["request"]["chunk"], original[0])
        self.assertEqual(record["report"]["expected_text"], source)
        self.assertEqual(record["engine_metadata"]["synthesis_text"], self.engine.speech_calls[0]["text"])
        self.assertTrue(accepted[0]["report"]["passed"])

    def test_old_accepted_quoted_speech_is_reused_after_edge_trimming_changes(self):
        from test_renderer_paragraphs import passages
        members = passages([' “Keep the lantern burning.” '], voices=["voice_lena"])
        chapter = dict(make_render_plan(self.bundle, self.config)["chapters"][0], chunks=members)
        voices = {"voice_lena": {"voice_id": "voice_lena", "asset_hash": "lena"}}

        # Reproduce a valid take generated before edge trimming was introduced.
        with patch("epub_tts_narrator.renderer.workflow.trim_speech_text", side_effect=lambda text: text):
            first = _render_chapter_chunks(
                chapter, voices, self.engine, self.engine.identity, self.verifier,
                self.config, self.root / "work", notify=self.messages.append,
            )
        before = self.accepted_clips()
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("regenerated accepted speech")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified accepted speech")):
            resumed = _render_chapter_chunks(
                chapter, voices, self.engine, self.engine.identity, self.verifier,
                self.config, self.root / "work", notify=self.messages.append,
            )
        self.assertEqual(resumed, first)
        self.assertEqual(self.accepted_clips(), before)

    def test_parenthetical_thought_and_narration_keep_separate_source_assignments(self):
        from test_renderer_paragraphs import passages, plan
        members = passages(['(Lie, ', 'I thought fiercely). '],
                           voices=["voice_lena", "voice_mara"], offset=178)
        original = deepcopy(members)
        grouped = group_paragraphs(plan(members))["chapters"][0]["chunks"]
        chapter = dict(make_render_plan(self.bundle, self.config)["chapters"][0], chunks=grouped)
        voices = {voice_id: {"voice_id": voice_id, "asset_hash": "shared-voice"}
                  for voice_id in ("voice_lena", "voice_mara")}

        clips, accepted = _render_chapter_chunks(
            chapter, voices, self.engine, self.engine.identity, self.verifier,
            self.config, self.root / "work", notify=self.messages.append,
        )

        self.assertEqual(self.engine.dialogue_calls, [])
        self.assertEqual([call["text"] for call in self.engine.speech_calls], ['Lie,', 'I thought fiercely.'])
        self.assertEqual(members, original)
        self.assertEqual([clip["source_ranges"] for clip in clips],
                         [member["source_ranges"] for member in members])
        self.assertEqual([clip["segment"]["voice_id"] for clip in clips], ["voice_lena", "voice_mara"])
        self.assertTrue(all(record["report"]["passed"] for record in accepted))
        self.assertEqual([record["report"]["expected_text"] for record in accepted],
                         ['(Lie, ', 'I thought fiercely). '])

    def test_migration_replaces_only_mixed_cached_paragraph_and_rebuilds_chapter(self):
        def legacy_groups(*args, **kwargs):
            return group_paragraphs(*args, **kwargs, multi_speaker=True)

        # Reproduce a chapter made by the old planner, including accepted mixed
        # speech. The production default must reject that group's saved anchor.
        with patch("epub_tts_narrator.renderer.paragraphs.group_paragraphs", side_effect=legacy_groups):
            self.chapters()
        before = self.accepted_clips()
        hashes = {path: file_hash(path.parent / record["audio_file"])
                  for path, record in before.items()}
        old_mixed = next(record for record in before.values()
                         if record["request"]["kind"] == "paragraph_speech"
                         and len(record["request"]["voice_assets"]) > 1)
        checkpoint_path = next((self.root / "work" / "completed-chapters").glob("*.json"))
        old_checkpoint = read_json(checkpoint_path)
        with self.assertRaisesRegex(RenderBlocked, "is stale"):
            self.package()
        counts = (len(self.engine.dialogue_calls), len(self.engine.speech_calls), len(self.verifier.calls))
        self.chapters()
        self.assertEqual(len(self.engine.dialogue_calls), counts[0])
        self.assertEqual(len(self.engine.speech_calls) - counts[1], 2)
        self.assertEqual(len(self.verifier.calls) - counts[2], 2)
        new = {path: record for path, record in self.accepted_clips().items() if path not in before}
        self.assertEqual(len(new), 2)
        ordered = sorted(new.values(), key=lambda row: row["request"]["chunk"]["segment"]["start"])
        self.assertEqual([record["request"]["chunk"] for record in ordered],
                         old_mixed["request"]["chunk"]["paragraph_turns"])
        for path, record in before.items():
            self.assertEqual(read_json(path), record)
            self.assertEqual(file_hash(path.parent / record["audio_file"]), hashes[path])
        checkpoint = read_json(checkpoint_path)
        self.assertNotEqual(checkpoint["context"]["chapter_hash"], old_checkpoint["context"]["chapter_hash"])
        self.assertNotIn(old_mixed["request"]["chunk"]["chunk_id"],
                         [entry["chunk_id"] for entry in checkpoint["acceptance"]])
        self.assertEqual(self.package()["status"], "complete")

    def test_accepted_individual_clips_survive_enabling_paragraph_mode(self):
        self.config["render"]["paragraph_synthesis"] = False
        first = self.chapters()
        accepted = self.accepted_clips()
        self.assertEqual(len(accepted), 9)
        self.assertEqual(self.engine.dialogue_calls, [])
        self.config["render"]["paragraph_synthesis"] = True
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("regenerated accepted individual clips")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded unchanged chapter")):
            resumed = self.chapters()
        self.assertEqual(first["chapters"], resumed["chapters"])
        self.assertEqual(self.accepted_clips(), accepted)
        self.assertEqual(self.engine.dialogue_calls, [])

    def test_interruption_reuses_the_complete_accepted_paragraph_on_resume(self):
        self.engine.interrupt_speech_number = 4
        with self.assertRaisesRegex(KeyboardInterrupt, "test interruption"):
            self.chapters()
        grouped = self.grouped_clips()
        self.assertEqual(len(grouped), 1)
        path, accepted = next(iter(grouped.items()))
        self.assertEqual(len(accepted["request"]["chunk"]["paragraph_turns"]), 3)
        wav = path.parent / accepted["audio_file"]
        old_hash = file_hash(wav)
        calls = len(self.verifier.calls)
        self.assertFalse((self.root / "work" / ".render.lock").exists())
        self.engine.interrupt_speech_number = None
        resumed = self.chapters()
        self.assertEqual(resumed["status"], "chapters_complete")
        self.assertEqual(len(self.engine.dialogue_calls), 1)
        self.assertEqual(len(self.engine.speech_calls), 7)  # six successes plus interruption
        self.assertEqual(len(self.verifier.calls) - calls, 3)  # interrupted and final two passages
        self.assertEqual(read_json(path), accepted)
        self.assertEqual(file_hash(wav), old_hash)
        self.assertEqual(len(self.grouped_clips()), 1)

    def test_partially_rendered_legacy_paragraph_keeps_its_boundaries(self):
        self.config["render"]["paragraph_synthesis"] = False
        self.engine.interrupt_speech_number = 4
        with self.assertRaises(KeyboardInterrupt):
            self.chapters()
        before = self.accepted_clips()
        self.assertEqual(len(before), 3)
        # The first short quotation in b000003 is accepted. Keep that entire
        # paragraph as individual turns. b000004 remains split at its voice change.
        self.assertEqual(sum(record["request"]["chunk"]["segment"]["block_id"] == "b000003"
                             for record in before.values()), 1)
        self.config["render"]["paragraph_synthesis"] = True
        self.engine.interrupt_speech_number = None
        self.chapters()
        self.assertEqual(len(self.engine.dialogue_calls), 0)
        self.assertEqual(len(self.engine.speech_calls), 10)  # nine accepted plus the interrupted call
        self.assertTrue(all(read_json(path) == record for path, record in before.items()))
        accepted = self.accepted_clips()
        self.assertEqual(len(accepted), 9)
        self.assertEqual(sum(record["request"]["kind"] == "speech"
                             and record["request"]["chunk"]["segment"]["block_id"] == "b000003"
                             for record in accepted.values()), 3)
        self.assertEqual(len(self.grouped_clips()), 0)


if __name__ == "__main__":
    unittest.main()
