from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import read_json, write_json
from legacy_recovery_fixtures import LegacySpeechResolver as SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed, VerificationUnavailable
import test_renderer_plain_delivery_integration as fixtures
from test_renderer_recurring_resolution import PlanClient
from test_renderer_speech_identity import speech_request
from test_renderer_workflow import FFMPEG, FFPROBE


REPLACEMENT = 'thoughtful, sustained hum'
TRAITS = 'light Welsh accent; low, resonant tone'


class DirectionClient(PlanClient):
    def __init__(self):
        super().__init__([{'action': 'stop', 'edits': [], 'reason': 'No supported control repair.'}])
        self.direction_calls = []
        self.fail_review = None
        self.approve = True

    def complete(self, prompt, payload):
        if payload.get('mode') in {'direction_repair', 'direction_review'}:
            self.direction_calls.append(deepcopy(payload))
            if self.fail_review is not None and payload.get('reviewer') == self.fail_review:
                raise TimeoutError('offline review outage')
            if payload['mode'] == 'direction_repair':
                return {'control_related': True, 'replacement_direction': REPLACEMENT,
                        'reason': 'Compact phrasing preserves the humming intention.'}
            return {'equivalent': self.approve, 'nonspoken': True, 'preserves_words': True,
                    'preserves_traits': True, 'reason': 'Same audible intent without quoted speech.'}
        return super().complete(prompt, payload)


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class DirectionRepairTests(unittest.TestCase):
    render_chunk = fixtures.PlainDeliveryIntegrationTests.render_chunk
    accepted_record = fixtures.PlainDeliveryIntegrationTests.accepted_record

    def setUp(self):
        fixtures.PlainDeliveryIntegrationTests.setUp(self)
        self.client = DirectionClient()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.chunk = speech_request(self.voice, 'Hmm.')['chunk']
        self.chunk['speech_traits'] = {self.voice['voice_id']: TRAITS}
        self.chunk['segment'].update(text='Hmmm.', start=0, end=5,
            delivery={'direction': 'thoughtful'}, speech_normalization={'version': 1, 'edits': [
                {'start': 0, 'end': 4, 'original': 'Hmmm', 'replacement': 'Hmm',
                 'direction': 'Maintain the extended hum as a searching, thoughtful sound.',
                 'reason': 'A stretched contemplative hum.'}]})
        self.engine.heard_override = lambda text, direction, number: (
            text if direction == TRAITS + '; ' + REPLACEMENT else 'Maintain the extended hum as a searching thoughtful sound')

    def test_normalized_control_leakage_gets_reviewed_replacement_and_full_verification(self):
        original = deepcopy(self.chunk)
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(self.chunk, original)
        self.assertEqual([call['text'] for call in self.engine.calls], ['Hmm.'] * 4)
        self.assertTrue(all(call['voice'] == self.voice for call in self.engine.calls))
        self.assertEqual(self.engine.calls[-1]['direction'], TRAITS + '; ' + REPLACEMENT)
        self.assertEqual(len(self.client.direction_calls), 4)
        reviews = self.client.direction_calls[1:]
        self.assertEqual([row['reviewer'] for row in reviews], [1, 2, 3])
        self.assertTrue(all('reviews' not in row and 'reason' not in row for row in reviews))
        saved = self.accepted_record(result)
        self.assertEqual(saved['request']['chunk'], original)
        self.assertIn('direction_repair', saved['engine_metadata'])
        counts = len(self.engine.calls), len(self.base.calls), len(self.client.direction_calls)
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual((len(self.engine.calls), len(self.base.calls), len(self.client.direction_calls)), counts)

    def test_prior_stop_reaches_new_capability_without_changing_old_audit(self):
        with patch.object(SpeechResolver, 'direction_repair', return_value=None):
            with self.assertRaises(SpeechVerificationFailed):
                self.render_chunk()
        path = next((self.work / 'clips').glob('*/speech-resolution.json'))
        original = path.read_bytes()
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertIn('direction_repair', self.accepted_record(result)['engine_metadata'])

    def test_review_outage_resumes_only_missing_reviews_without_synthesis(self):
        self.client.fail_review = 2
        with self.assertRaises(VerificationUnavailable):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        path = next((self.work / 'clips').glob('*/direction-repair.json'))
        state = read_json(path)
        self.assertEqual(state['phase'], 'pending')
        self.assertEqual(len(state['audit']['reviews']), 1)
        self.client.fail_review = None
        _, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual([row.get('reviewer') for row in self.client.direction_calls], [None, 1, 2, 2, 3])
        self.assertEqual(len(self.engine.calls), 4)

    def test_rejected_intent_is_not_rerolled_on_restart(self):
        self.client.approve = False
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        counts = len(self.engine.calls), len(self.client.direction_calls), len(self.client.plans)
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual((len(self.engine.calls), len(self.client.direction_calls), len(self.client.plans)), counts)
        self.assertFalse(list((self.work / 'clips').glob('*/accepted.json')))

    def test_failed_repaired_take_remains_rejected_without_new_trial(self):
        self.engine.heard_override = lambda *args: 'An unrelated spoken instruction'
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 4)
        path = next((self.work / 'clips').glob('*/direction-repair.json'))
        self.assertEqual(read_json(path)['phase'], 'failed')
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.direction_calls), 4)
        self.assertFalse(list((self.work / 'clips').glob('*/accepted.json')))

    def test_interrupted_verification_resumes_bound_audio_without_model_or_tts(self):
        from epub_tts_narrator.renderer.workflow import validate_audio

        def interrupt(wav, *args, **kwargs):
            if wav.name == 'attempt-004.wav':
                raise KeyboardInterrupt('offline interruption before recognition')
            return validate_audio(wav, *args, **kwargs)

        with patch('epub_tts_narrator.renderer.workflow.validate_audio', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        self.assertEqual(len(self.engine.calls), 4)
        _, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.direction_calls), 4)

    def test_interrupted_approved_candidate_resumes_without_new_model_decisions(self):
        from epub_tts_narrator.renderer.direction_repair_state import DirectionRepair
        save = DirectionRepair._save

        def interrupt(session):
            save(session)
            if session.state['phase'] == 'approved':
                raise KeyboardInterrupt('offline interruption after approval')

        with patch.object(DirectionRepair, '_save', interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        _, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.client.direction_calls), 4)
        self.assertEqual(len(self.engine.calls), 4)

    def test_repair_cannot_exceed_existing_shared_generation_cap(self):
        self.config['verification']['max_resolution_attempts'] = 3
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        self.assertFalse(list((self.work / 'clips').glob('*/accepted.json')))

    def test_interrupted_unbound_generation_reuses_approval_but_never_accepts_orphan(self):
        synthesize = self.engine.synthesize

        def interrupt(*args, **kwargs):
            metadata = synthesize(*args, **kwargs)
            if len(self.engine.calls) == 4:
                raise KeyboardInterrupt('offline interruption before generation metadata')
            return metadata

        with patch.object(self.engine, 'synthesize', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        orphan = next((self.work / 'clips').glob('*/attempt-004.json'))
        original = orphan.read_bytes()
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(self.accepted_record(result)['attempt'], 5)
        self.assertEqual(orphan.read_bytes(), original)
        self.assertEqual(len(self.engine.calls), 5)
        self.assertEqual(len(self.client.direction_calls), 4)

    def test_accepted_record_remains_reusable_after_provider_change(self):
        result, acceptance = self.render_chunk()
        self.client.identity = dict(self.client.identity, model='another-offline-model')
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.direction_calls), 4)

    def test_unproven_conditioning_cannot_publish_success_after_fresh_recognition(self):
        self.engine.heard_override = lambda *args: 'An unrelated spoken instruction'
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        path = next((self.work / 'clips').glob('*/attempt-004.json'))
        record = read_json(path)
        record['engine_metadata']['direction'] = REPLACEMENT
        write_json(path, record)
        for key in self.registry:
            self.registry[key] = 'Hmm.'
        with self.assertRaises(ValueError):
            self.render_chunk()
        self.assertFalse(list((self.work / 'clips').glob('*/accepted.json')))
        self.assertEqual(len(self.engine.calls), 4)
