import unittest

from epub_tts_narrator.renderer.speech_cache import SpeechCache
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
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
