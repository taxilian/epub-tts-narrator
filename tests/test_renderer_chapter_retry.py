"""Standalone chapter jobs retry exhausted speech without recasting voices."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import digest, read_json
from epub_tts_narrator.renderer.retry_budget import SpeechRetryBudget
from epub_tts_narrator.renderer.workflow import (
    RenderBlocked, _render_chapter_chunks, make_render_plan,
)
import test_renderer_paragraph_workflow as paragraph_fixtures
from test_renderer_workflow import FFMPEG, FFPROBE


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for chapter retry tests")
class StandaloneChapterRetryTests(unittest.TestCase):
    def setUp(self):
        paragraph_fixtures.ParagraphWorkflowTests.setUp(self)
        for row in self.bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000004":
                row["voice_id"] = "voice_lena"
                row["role_id"] = "lena_dialogue"
        plan = make_render_plan(self.bundle, self.config)
        target = next(chunk for chunk in plan["chapters"][0]["chunks"]
                      if chunk["segment"]["block_id"] == "b000004")
        self.chapter = dict(plan["chapters"][0], chunks=[deepcopy(target)])
        self.voices = {voice_id: {"voice_id": voice_id, "asset_hash": digest(voice_id)}
                       for voice_id in plan["voice_ids"]}
        self.text = target["spoken_text"]
        self.verifier.wrong_texts.add(self.text)

    def run_chapter(self, *, retry_budget=None):
        return _render_chapter_chunks(
            self.chapter, self.voices, self.engine, self.engine.identity,
            self.verifier, self.config, self.root / "work", notify=self.messages.append,
            total_chapters=len(self.bundle["chapters"]), retry_budget=retry_budget,
        )

    def attempts(self):
        return {path: path.read_bytes()
                for path in (self.root / "work" / "clips").glob("*/attempt-*.json")}

    def test_explicit_chapter_retry_gets_new_bounded_takes_then_reuses_accepted_audio(self):
        for invocation in range(2):
            before = self.attempts()
            with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
                self.run_chapter()
            self.assertEqual(len(self.engine.dialogue_calls), 2 * (invocation + 1))
            self.assertEqual(len(self.verifier.calls), 2 * (invocation + 1))
            for path, saved in before.items():
                self.assertEqual(path.read_bytes(), saved)
        failed = self.attempts()
        self.assertEqual(len(failed), 4)
        self.assertEqual({read_json(path)["attempt"] for path in failed}, {1, 2, 3, 4})
        self.assertEqual(len({call["seed"] for call in self.engine.dialogue_calls}), 4)
        self.assertEqual(self.engine.voice_calls, [])
        self.assertEqual(self.engine.speech_calls, [])

        self.verifier.wrong_texts.clear()
        clips, acceptance = self.run_chapter()
        self.assertEqual(len(self.engine.dialogue_calls), 5)
        self.assertEqual(len(self.verifier.calls), 5)
        self.assertEqual(len({call["seed"] for call in self.engine.dialogue_calls}), 5)
        accepted_path = next((self.root / "work" / "clips").glob("*/accepted.json"))
        accepted = read_json(accepted_path)
        self.assertEqual(accepted["attempt"], 5)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["paragraph_turns"], self.chapter["chunks"][0]["paragraph_turns"])
        self.assertEqual(acceptance[0]["audio_sha256"], accepted["audio_sha256"])
        for path, saved in failed.items():
            self.assertEqual(path.read_bytes(), saved)

        with patch.object(self.engine, "synthesize_dialogue", side_effect=AssertionError("regenerated paragraph")), \
             patch("epub_tts_narrator.renderer.workflow.validate_audio", side_effect=AssertionError("reverified acceptance")):
            self.assertEqual(self.run_chapter(), (clips, acceptance))
        self.assertEqual(read_json(accepted_path), accepted)

    def test_shared_chapter_invocation_cannot_reset_paragraph_allowance(self):
        budget = SpeechRetryBudget(retry_paragraphs=True)
        for _ in range(2):
            with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
                self.run_chapter(retry_budget=budget)
            self.assertEqual(len(self.engine.dialogue_calls), 2)
        self.assertEqual(len(self.attempts()), 2)

    def test_local_book_budget_keeps_exhausted_group_available_for_split_recovery(self):
        for _ in range(2):
            with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
                self.run_chapter(retry_budget=SpeechRetryBudget())
            self.assertEqual(len(self.engine.dialogue_calls), 2)
        self.assertEqual(len(self.attempts()), 2)
