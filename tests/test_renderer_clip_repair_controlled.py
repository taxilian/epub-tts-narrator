from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

from epub_tts_narrator.performance_plan import text_sha256, validate_performance
from epub_tts_narrator.renderer import clip_repair_execute
from epub_tts_narrator.renderer.clip_repair_performance import (
    controlled_repair_guidance, project_plan_performance)
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.engines import EngineError, prepare_synthesis
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
import test_renderer_clip_repair as clip_repair_tests
from test_renderer_speech_identity import speech_request
from test_renderer_speech_recovery_integration import pcm
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class ControlledClipRepairTests(unittest.TestCase):
    setUpFixture = clip_repair_tests.BroadRepairTests.setUpFixture
    render_chunk = clip_repair_tests.BroadRepairTests.render_chunk
    accepted_record = clip_repair_tests.BroadRepairTests.accepted_record

    def setUp(self):
        self.setUpFixture()
        self.client = clip_repair_tests.BroadClient()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.engine.heard_override = lambda text, direction, number: 'Forgive me, Lady Polgara'

    def set_controlled_polgara(self):
        text = '"Forgive me, Lady Polgara,"'
        self.chunk = speech_request(self.voice, text)['chunk']
        self.chunk['segment'].update(text=text, start=0, end=len(text),
                                     delivery={'direction': 'croaked', 'evidence': []})
        self.chunk.update(spoken_text=text, spoken_start=0, spoken_end=len(text),
                          source_ranges=[{'start': 0, 'end': len(text)}], chunk_id='controlled-polgara',
                          expressive_direction='croaked', expressive_traits={},
                          expressive_performance={'version': 1, 'text_sha256': text_sha256(text),
                                                  'controls': [{'kind': 'pronunciation', 'start': 18, 'end': 25,
                                                                'text': 'Polgara', 'alphabet': 'cmu_arpabet',
                                                                'phonemes': 'P AH1 L G EH0 R AH0'}]})

        def controlled_synthesize(text, voice, path, *, seed, direction='', performance=None, traits=''):
            prepared, compiled, metadata = prepare_synthesis(
                text, direction, performance=performance, traits=traits)
            self.engine.calls.append({'text': prepared, 'raw_text': text, 'voice': deepcopy(voice),
                                      'direction': direction, 'seed': seed, 'compiled': compiled,
                                      'performance': performance})
            if compiled.endswith(','):
                raise EngineError("Drama 3 did not finish the text: 'length'")
            write_tone(path, frequency=220 + len(self.engine.calls) * 31)
            self.registry[hashlib.sha256(pcm(path)).hexdigest()] = 'Forgive me, Lady Polgara'
            return {'fixture': True, **metadata, 'synthesis_text': compiled}

        self.engine.synthesize = controlled_synthesize

    def test_no_audio_controlled_failure_gets_reviewed_punctuation_repair_with_same_phones(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Remove the trailing comma while preserving every word.'}
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(sorted(row['reviewer'] for row in self.client.review_calls), [1, 2])
        self.assertEqual(len(self.engine.calls), 4)
        repaired = self.engine.calls[-1]
        self.assertEqual(repaired['raw_text'], '"Forgive me, Lady Polgara."')
        self.assertIn('<|phoneme_start|>P AH1 L G EH0 R AH0<|phoneme_end|>', repaired['compiled'])
        record = self.accepted_record(result)
        metadata = record['engine_metadata']['clip_repair']['parts'][0]
        self.assertEqual(metadata['performance']['controls'][0]['phonemes'], 'P AH1 L G EH0 R AH0')
        self.assertIn('<|phoneme_start|>P AH1 L G EH0 R AH0<|phoneme_end|>', metadata['metadata']['synthesis_text'])
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)

    def test_projection_preserves_confirmed_phones_across_valid_split_parts(self):
        self.set_controlled_polgara()
        plan = {'parts': [{'text': '"Forgive me,', 'delivery': 'croaked', 'speech_traits': ''},
                          {'text': ' Lady Polgara."', 'delivery': 'croaked', 'speech_traits': ''}],
                'reason': 'Split the address at its natural pause.'}
        projected = project_plan_performance({'kind': 'speech', 'chunk': self.chunk}, plan)
        self.assertIsNone(projected[0])
        self.assertEqual(projected[1]['controls'][0]['phonemes'], 'P AH1 L G EH0 R AH0')
        self.assertEqual((projected[1]['controls'][0]['start'], projected[1]['controls'][0]['end']), (6, 13))
        self.assertEqual(projected[1]['controls'][0]['text'], 'Polgara')

    def test_controlled_proposal_and_reviews_carry_the_word_exact_contract(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Remove the trailing comma while preserving every word.'}
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.client.repair_calls), 1)
        guidance = self.client.repair_calls[0]['controlled_pronunciation']
        self.assertTrue(guidance['word_exact'])
        self.assertEqual(guidance['protected_spans'], ['Polgara'])
        self.assertIn('punctuation outside protected spans', guidance['allowed_changes'])
        self.assertIn('word spelling', guidance['forbidden_changes'])
        self.assertIn('word order', guidance['forbidden_changes'])
        self.assertEqual(sorted(row['reviewer'] for row in self.client.review_calls), [1, 2])
        for review in self.client.review_calls:
            self.assertEqual(review['controlled_pronunciation'], guidance)
        self.assertIn('<|phoneme_start|>P AH1 L G EH0 R AH0<|phoneme_end|>',
                      self.engine.calls[-1]['compiled'])

    def test_plain_repair_payload_carries_no_controlled_guidance(self):
        self.engine.heard_override = lambda text, direction, number: text if number > 3 else 'Wrong'
        _, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertIsNone(controlled_repair_guidance(
            {'kind': 'speech', 'chunk': self.chunk}))
        self.assertNotIn('controlled_pronunciation', self.client.repair_calls[0])
        self.assertTrue(all('controlled_pronunciation' not in row for row in self.client.review_calls))

    def make_multiword_request(self):
        text = '"We cross Lake Atitlan tomorrow,"'
        start = text.index('Lake Atitlan')
        chunk = speech_request(self.voice, text)['chunk']
        chunk['segment'].update(text=text, start=0, end=len(text),
                                delivery={'direction': '', 'evidence': []})
        record = {'version': 1, 'text_sha256': text_sha256(text),
                  'controls': [{'kind': 'pronunciation', 'start': start, 'end': start + 12,
                                'text': 'Lake Atitlan', 'alphabet': 'cmu_arpabet',
                                'phonemes': 'L EY1 K AE2 T IH0 T L AE1 N'}]}
        chunk.update(spoken_text=text, spoken_start=0, spoken_end=len(text),
                     source_ranges=[{'start': 0, 'end': len(text)}], chunk_id='controlled-atitlan',
                     expressive_direction='', expressive_traits={},
                     expressive_performance=validate_performance(record, text))
        return {'kind': 'speech', 'chunk': chunk}

    def test_multiword_control_survives_punctuation_repair_with_shifted_part_offsets(self):
        request = self.make_multiword_request()
        plan = {'parts': [{'text': '"We cross', 'delivery': '', 'speech_traits': ''},
                          {'text': ' Lake Atitlan tomorrow."', 'delivery': '', 'speech_traits': ''}],
                'reason': 'Split at the pause and replace the trailing comma.'}
        projected = project_plan_performance(request, plan)
        self.assertIsNone(projected[0])
        control = projected[1]['controls'][0]
        self.assertEqual(control['phonemes'], 'L EY1 K AE2 T IH0 T L AE1 N')
        self.assertEqual((control['start'], control['end']), (1, 13))
        self.assertEqual(control['text'], 'Lake Atitlan')
        _, compiled, metadata = prepare_synthesis(' Lake Atitlan tomorrow."', '', performance=projected[1])
        self.assertIn('<|phoneme_start|>L EY1 K AE2 T IH0 T L AE1 N<|phoneme_end|>', compiled)
        self.assertNotIn('Lake', compiled)
        self.assertEqual(metadata['expressive_performance']['plan']['controls'][0]['phonemes'],
                         'L EY1 K AE2 T IH0 T L AE1 N')

    def test_multiword_control_split_across_parts_is_rejected(self):
        request = self.make_multiword_request()
        plan = {'parts': [{'text': '"We cross Lake', 'delivery': '', 'speech_traits': ''},
                          {'text': ' Atitlan tomorrow,"', 'delivery': '', 'speech_traits': ''}],
                'reason': 'Split inside the place name.'}
        with self.assertRaisesRegex(ValueError, 'splits a confirmed pronunciation span'):
            project_plan_performance(request, plan)

    def test_repair_changing_text_inside_a_multiword_control_is_rejected(self):
        request = self.make_multiword_request()
        plan = {'parts': [{'text': '"We cross Lake, Atitlan tomorrow."',
                           'delivery': '', 'speech_traits': ''}],
                'reason': 'Add a pause inside the place name.'}
        with self.assertRaisesRegex(ValueError, 'changes text inside a confirmed pronunciation span'):
            project_plan_performance(request, plan)

    def test_controlled_plan_that_drops_protected_word_is_rejected_without_new_synthesis_or_chooser(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Try a shorter version.'}
        with self.assertRaisesRegex(ValueError, 'No physically usable.*did not finish.*Inspect') as failure:
            self.render_chunk()
        self.assertIn(str(self.work / 'clips'), str(failure.exception))
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(self.client.review_calls, [])
        self.assertEqual(self.client.selection_calls, [])
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertEqual(state['phase'], 'rejected')

    def test_controlled_plan_that_duplicates_protected_word_is_rejected_without_new_synthesis(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Try a repeated address.'}
        with self.assertRaisesRegex(ValueError, 'No physically usable.*did not finish'):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.review_calls), 0)

    def test_controlled_plan_that_moves_protected_word_is_rejected_without_new_synthesis(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Polgara, forgive me, Lady."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Try a fronted address.'}
        with self.assertRaisesRegex(ValueError, 'No physically usable.*did not finish'):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.review_calls), 0)

    def test_controlled_plan_with_unsupported_scoped_control_is_rejected_without_claiming_recovery(self):
        self.set_controlled_polgara()
        record = self.chunk['expressive_performance']
        record['controls'].append({'kind': 'delivery', 'start': 1, 'end': 25, 'text': 'Forgive me, Lady Polgara',
                                   'tag': 'whisper', 'direction': ''})
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Remove the trailing comma while preserving every word.'}
        with self.assertRaisesRegex(ValueError, 'No physically usable.*did not finish'):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.review_calls), 0)
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertEqual(state['phase'], 'rejected')
        self.assertEqual(state['error_code'], 'unsupported_performance_controls')

    def test_controlled_repair_respects_shared_generation_cap(self):
        self.set_controlled_polgara()
        self.config['verification']['max_resolution_attempts'] = 3
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Remove the trailing comma while preserving every word.'}
        with self.assertRaisesRegex(ValueError, 'No physically usable.*did not finish'):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(self.client.repair_calls, [])
        self.assertEqual(self.client.review_calls, [])
        self.assertEqual(self.client.selection_calls, [])

    def test_interrupted_controlled_repair_resumes_joined_audio_without_new_proposal_reviews_or_tts(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Remove the trailing comma while preserving every word.'}
        validate = clip_repair_execute.validate_audio

        def interrupt(wav, *args, **kwargs):
            if (wav.parent / 'clip-repair.json').exists() and wav.name.startswith('attempt-004-'):
                raise KeyboardInterrupt('offline interruption before clip repair verification')
            return validate(wav, *args, **kwargs)

        with patch.object(clip_repair_execute, 'validate_audio', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        self.assertEqual(len(self.engine.calls), 4)
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)
        self.assertTrue(self.accepted_record(result)['report']['passed'])
