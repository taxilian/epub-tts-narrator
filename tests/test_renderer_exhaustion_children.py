import unittest

from epub_tts_narrator.renderer.speech_cache import SpeechCache
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.engines import _directed_text
from epub_tts_narrator.renderer.fish_http import FishHTTPError
from epub_tts_narrator.renderer.speech_recovery import recover_speech
from test_renderer_clip_repair import BroadClient
import test_renderer_split_delivery_recovery as fixtures
from test_renderer_workflow import FFMPEG, FFPROBE


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class ExhaustionChildrenTests(unittest.TestCase):
    generate = fixtures.SplitDeliveryRecoveryTests.generate
    plain = fixtures.SplitDeliveryRecoveryTests.plain
    recover = fixtures.SplitDeliveryRecoveryTests.recover

    def setUp(self):
        fixtures.SplitDeliveryRecoveryTests.setUp(self)
        self.client = BroadClient()
        self.client.plan = {'parts': [], 'reason': 'Select the best available full span.'}
        self.verifier = RecurringSpeechVerifier(self.asr, SpeechResolver(self.client))

    def test_child_selection_survives_ordinary_join_proofs_and_cache(self):
        self.asr.final_transcript = fixtures.TEXT
        wav, accepted = self.recover()
        first = accepted['engine_metadata']['recovery']['parts'][0]['acceptance']
        self.assertEqual(first['report']['text_confirmation_method'], 'llm_exhaustion_selection')
        self.assertFalse(first['report']['original_failed_report']['passed'])
        expected = generation_identity(self.request, self.voices)
        self.assertEqual(generation_identity(accepted['request'], self.voices,
                                             engine_metadata=accepted['engine_metadata']), expected)
        cache = SpeechCache(self.work, {self.voice['voice_id']: self.voice})
        cache.remember(expected, accepted)
        self.assertEqual(cache.find(expected, request=self.request), (wav, accepted))
        calls = len(self.calls)
        self.assertEqual(self.recover(), (wav, accepted))
        self.assertEqual(len(self.calls), calls)

    def test_failed_full_join_is_selectable_without_reassembly_or_lexical_reveto(self):
        self.asr.final_transcript = 'The pain is fading. We leave now.'
        wav, accepted = self.recover()
        self.assertEqual(accepted['report']['text_confirmation_method'], 'llm_exhaustion_selection')
        selected = accepted['engine_metadata']['exhaustion_selection']['selected_record']
        self.assertIn('recovery', selected['engine_metadata'])
        self.assertFalse(selected['report']['passed'])
        expected = generation_identity(self.request, self.voices)
        self.assertEqual(generation_identity(accepted['request'], self.voices,
                                             engine_metadata=accepted['engine_metadata']), expected)
        calls = len(self.calls), len(self.client.selection_calls)
        self.assertEqual(self.recover(), (wav, accepted))
        self.assertEqual((len(self.calls), len(self.client.selection_calls)), calls)

    def test_no_audio_child_repair_failure_resumes_the_unfinished_parent_assembly(self):
        self.fail_directed = False
        self.client.plan = {'parts': [{'text': 'We should leave now.', 'delivery': '', 'speech_traits': ''}],
                            'reason': 'Use a complete sentence with a clean ending.'}
        repair_failed = True
        repair_starts = []

        def generate(text, path, seed):
            if text.startswith('We should'):
                raise FishHTTPError("Drama 3 did not finish the text: 'silence'")
            return self.generate(text, path, seed)

        def repair(part, path, seed):
            repair_starts.append(seed)
            if repair_failed:
                raise FishHTTPError('fixture repair request timed out')
            metadata = self.engine._write(part['text'], path, seed)
            directed, controls = _directed_text(part['text'], '')
            return {**metadata, **controls, 'synthesis_text': directed}

        def recover():
            from epub_tts_narrator.renderer.retry_budget import SpeechRetryBudget
            return recover_speech(
                fixtures.TEXT, generate, self.parent, self.request, self.verifier, self.config,
                retry_budget=SpeechRetryBudget(), voices_by_asset=self.voices,
                synthesize_plain=self.plain, synthesize_plan_part=repair, notify=self.messages.append)

        with self.assertRaisesRegex(ValueError, 'No physically usable'):
            recover()
        marker = read_json(self.parent / 'speech-recovery.json')
        directory = self.parent.parent / marker['recovery_request_hash']
        incomplete = read_json(directory / 'attempt-001.json')
        self.assertIsNone(incomplete['audio_file'])
        good_child = next((directory / 'parts').glob('*/accepted.json'))
        good_before = good_child.read_bytes()
        repair_failed = False

        wav, accepted = recover()

        self.assertTrue(wav.is_file())
        self.assertTrue(accepted['report']['passed'])
        self.assertEqual(good_child.read_bytes(), good_before)
        self.assertEqual(len(repair_starts), 2)
        self.assertNotEqual(*repair_starts)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)
