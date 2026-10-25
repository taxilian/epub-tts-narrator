import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import write_json
import test_renderer_clip_repair as fixtures
from test_renderer_workflow import FFMPEG, FFPROBE


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class ClipRepairPublicationTests(unittest.TestCase):
    setUp = fixtures.BroadRepairTests.setUp
    setUpFixture = fixtures.BroadRepairTests.setUpFixture
    render_chunk = fixtures.BroadRepairTests.render_chunk
    accepted_record = fixtures.BroadRepairTests.accepted_record

    @staticmethod
    def fail_acceptance(path, record):
        if path.name == 'accepted.json':
            raise OSError('offline acceptance publication interruption')
        return write_json(path, record)

    def test_completed_ordinary_verification_resumes_publication_without_a_chooser(self):
        self.client.plan = {'parts': [{'text': 'Hummm.', 'delivery': '', 'speech_traits': ''}],
                            'reason': 'A phonetic experiment.'}
        self.engine.heard_override = lambda text, direction, number: 'Hmm.' if number > 3 else 'Wrong instruction'
        with patch('epub_tts_narrator.renderer.clip_repair_execute.write_json', side_effect=self.fail_acceptance):
            with self.assertRaises(OSError):
                self.render_chunk()
        _, accepted = self.render_chunk()
        self.assertTrue(accepted['report']['passed'])
        self.assertEqual(len(self.client.selection_calls), 0)
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.review_calls), 2)

    def test_completed_choice_resumes_publication_without_another_choice_or_tts(self):
        with patch('epub_tts_narrator.renderer.exhaustion_selection.write_json', side_effect=self.fail_acceptance):
            with self.assertRaises(OSError):
                self.render_chunk()
        calls = len(self.engine.calls), len(self.client.selection_calls)
        _, accepted = self.render_chunk()
        self.assertEqual(accepted['report']['text_confirmation_method'], 'llm_exhaustion_selection')
        self.assertEqual((len(self.engine.calls), len(self.client.selection_calls)), calls)
