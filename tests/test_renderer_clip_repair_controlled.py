from pathlib import Path
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

from epub_tts_narrator.performance_plan import text_sha256, validate_performance
from epub_tts_narrator.renderer import clip_recovery, clip_repair_execute
from epub_tts_narrator.renderer.clip_repair_performance import (
    controlled_repair_guidance, project_plan_performance)
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.engines import EngineError, prepare_synthesis
from epub_tts_narrator.renderer.fish_http import FishHTTPError
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

    def test_approved_repair_retries_model_truncation_without_reasking_or_losing_phones(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Preserve the words and phones with a complete sentence.'}
        synthesize = self.engine.synthesize
        starts = []

        def transient(text, voice, path, *, seed, **options):
            if text == self.client.plan['parts'][0]['text']:
                starts.append(seed)
                if len(starts) == 1:
                    raise FishHTTPError("Fish HTTP 500 (fixture): EngineError: Drama 3 did not finish the text: 'silence'")
            return synthesize(text, voice, path, seed=seed, **options)

        with patch.object(self.engine, 'synthesize', side_effect=transient):
            result, acceptance = self.render_chunk()

        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(set(starts)), 2)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(sorted(row['reviewer'] for row in self.client.review_calls), [1, 2])
        self.assertEqual(self.client.selection_calls, [])
        record = self.accepted_record(result)
        performance = record['engine_metadata']['clip_repair']['parts'][0]['performance']
        self.assertEqual(performance['controls'][0]['phonemes'],
                         self.chunk['expressive_performance']['controls'][0]['phonemes'])

    def assert_failed_repair(self, error, expected_starts, maximum=12):
        self.set_controlled_polgara()
        self.config['verification']['max_resolution_attempts'] = maximum
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Keep the confirmed reading with complete punctuation.'}
        synthesize = self.engine.synthesize
        starts = []

        def fail_repair(text, voice, path, *, seed, **options):
            if text == self.client.plan['parts'][0]['text']:
                starts.append(seed)
                raise error
            return synthesize(text, voice, path, seed=seed, **options)

        with patch.object(self.engine, 'synthesize', side_effect=fail_repair):
            with self.assertRaisesRegex(ValueError, 'No physically usable'):
                self.render_chunk()
        self.assertEqual(len(starts), expected_starts)
        self.assertEqual(len(set(starts)), expected_starts)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)
        return read_json(next((self.work / 'clips').glob('*/clip-repair.json')))

    def test_repair_model_retries_stop_at_the_configured_per_part_limit(self):
        state = self.assert_failed_repair(
            FishHTTPError("Drama 3 did not finish the text: 'silence'"),
            self.config['render']['max_attempts'])
        self.assertEqual(state['error_code'], 'synthesis_or_input_failure')
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)
        resumed = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertEqual(resumed['parts'][0]['starts'], self.config['render']['max_attempts'] + 1)

    def test_repair_retries_do_not_exceed_the_shared_allowance(self):
        state = self.assert_failed_repair(
            FishHTTPError("Drama 3 did not finish the text: 'silence'"), 1, maximum=4)
        self.assertEqual(state['error_code'], 'shared_budget_exhausted')

    def test_repair_transport_timeout_is_not_retried_as_model_truncation(self):
        state = self.assert_failed_repair(FishHTTPError('fixture request timed out'), 1)
        self.assertEqual(state['error_code'], 'synthesis_or_input_failure')
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)

    def test_approved_repair_resumes_when_new_resolution_is_disabled(self):
        self.assert_failed_repair(FishHTTPError('fixture request timed out'), 1)
        run = clip_recovery.run

        def resume_without_new_resolution(*args, **options):
            options['allow_resolution'] = False
            return run(*args, **options)

        with patch.object(clip_recovery, 'run', side_effect=resume_without_new_resolution):
            result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertIn('clip_repair', self.accepted_record(result)['engine_metadata'])
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)

    def test_disabled_resolution_does_not_start_a_new_repair(self):
        self.set_controlled_polgara()
        run = clip_recovery.run

        def without_new_resolution(*args, **options):
            options['allow_resolution'] = False
            return run(*args, **options)

        with patch.object(clip_recovery, 'run', side_effect=without_new_resolution):
            with self.assertRaisesRegex(ValueError, 'No physically usable'):
                self.render_chunk()
        self.assertEqual(self.client.repair_calls, [])
        self.assertEqual(self.client.review_calls, [])

    def test_repair_retries_preserve_a_start_for_each_pending_plan_part(self):
        self.set_controlled_polgara()
        self.config['verification']['max_resolution_attempts'] = 5
        self.client.plan = {'parts': [
            {'text': '"Forgive me,', 'delivery': 'croaked', 'speech_traits': ''},
            {'text': 'Lady Polgara."', 'delivery': 'croaked', 'speech_traits': ''}],
            'reason': 'Keep the address and its confirmed phones in separate complete pieces.'}
        with self.assertRaisesRegex(ValueError, 'No physically usable'):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 4)
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertEqual(state['error_code'], 'shared_budget_exhausted')
        self.assertEqual([part['starts'] for part in state['parts']], [1, 0])

    def test_interrupted_model_retry_resumes_the_approved_repair_with_a_new_seed(self):
        self.set_controlled_polgara()
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked', 'speech_traits': ''}],
                            'reason': 'Keep the reading and complete punctuation.'}
        synthesize = self.engine.synthesize
        starts = []

        def interrupted(text, voice, path, *, seed, **options):
            if text == self.client.plan['parts'][0]['text']:
                starts.append(seed)
                if len(starts) == 1:
                    raise FishHTTPError("Drama 3 did not finish the text: 'silence'")
                raise KeyboardInterrupt('interrupted repair retry')
            return synthesize(text, voice, path, seed=seed, **options)

        with patch.object(self.engine, 'synthesize', side_effect=interrupted):
            with self.assertRaisesRegex(KeyboardInterrupt, 'interrupted repair retry'):
                self.render_chunk()
        self.assertEqual(len(starts), 2)
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)
        self.assertNotIn(self.engine.calls[-1]['seed'], starts)

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

    def test_a_line_s_scoped_delivery_gives_way_to_the_repair_s_own_delivery(self):
        # A whispered span (like a spelling repair's "drawn-out" style span) is delivery a repair re-decides:
        # the reviewed repair runs, the confirmed pronunciation stays exact and the whisper is the parts' own call.
        self.set_controlled_polgara()
        record = self.chunk['expressive_performance']
        record['controls'].append({'kind': 'delivery', 'start': 1, 'end': 25, 'text': 'Forgive me, Lady Polgara',
                                   'tag': 'whisper', 'direction': ''})
        self.client.plan = {'parts': [{'text': '"Forgive me, Lady Polgara."',
                                      'delivery': 'croaked whisper', 'speech_traits': ''}],
                            'reason': 'Remove the trailing comma while preserving every word.'}
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(sorted(row['reviewer'] for row in self.client.review_calls), [1, 2])
        repaired = self.engine.calls[-1]
        self.assertIn('<|phoneme_start|>P AH1 L G EH0 R AH0<|phoneme_end|>', repaired['compiled'])
        self.assertNotIn('<whisper>', repaired['compiled'])
        self.assertEqual(repaired['direction'], 'croaked whisper')

    def test_a_repair_refused_only_for_scoped_delivery_reopens_for_review(self):
        from epub_tts_narrator.renderer.clip_repair_state import ClipRepairState
        self.set_controlled_polgara()
        request = speech_request(self.voice, self.chunk['spoken_text'])
        request['chunk'] = self.chunk
        directory = self.work / 'refused'
        directory.mkdir()
        session = ClipRepairState(directory, request, self.client)
        session.state.update(phase='rejected', error_code='unsupported_performance_controls',
                             proposal_received=True, audit={'proposal': {'parts': []}})
        session.save()
        reopened = ClipRepairState(directory, request, self.client)
        self.assertEqual(reopened.state['phase'], 'pending')
        self.assertNotIn('error_code', reopened.state)
        self.assertEqual(reopened.state['audit'], {'proposal': {'parts': []}}, 'its received proposal is kept')

    def test_a_line_with_only_drawn_out_style_spans_can_be_repaired(self):
        # "Wheeerrrrrre… herrrr…", spoken "Where… her…" with each repaired word drawn out: the runaway takes left
        # no audio and no sentence split exists, so a repair splitting the two words must be allowed to run.
        text = '"Where… her…"'
        request = speech_request(self.voice, text)
        request['chunk']['expressive_performance'] = {'version': 1, 'text_sha256': text_sha256(text), 'controls': [
            {'kind': 'delivery', 'tag': 'style', 'direction': 'drawn-out delivery', 'start': 1, 'end': 6,
             'text': 'Where'},
            {'kind': 'delivery', 'tag': 'style', 'direction': 'drawn-out delivery', 'start': 8, 'end': 11,
             'text': 'her'}]}
        plan = {'parts': [{'text': 'Where…', 'delivery': 'soft moaning whisper, drawn out', 'speech_traits': ''},
                          {'text': 'her…', 'delivery': 'the same faint drawn-out whisper', 'speech_traits': ''}]}
        self.assertEqual(project_plan_performance(request, plan), [None, None])
        self.assertIsNone(controlled_repair_guidance(request), 'its repair request keeps the plain payload')

    def test_a_mid_line_change_opens_with_the_accent_and_earlier_takes_render_again(self):
        # "[cue][kind] Don't reproach yourself. [dry, faintly teasing] Or did Lars mention that?": the span's tag
        # takes over from the opening ones, so it carries the accent cue too. A take recorded before, with the
        # span's tag alone, is not the current input: the line renders again instead of failing.
        from epub_tts_narrator.renderer import engines
        text = 'Forgive me, Lady. Or did Lars mention that?'
        self.engine.heard_override = lambda heard, direction, number: heard
        self.chunk = speech_request(self.voice, text)['chunk']
        voice_id = self.chunk['segment']['voice_id']
        self.chunk.update(chunk_id='mid-line-change', expressive_direction='kind',
                          expressive_traits={voice_id: 'light Irish accent'},
                          expressive_performance={'version': 1, 'text_sha256': text_sha256(text), 'controls': [
                              {'kind': 'delivery', 'tag': 'style', 'direction': 'dry, faintly teasing',
                               'start': 18, 'end': len(text), 'text': text[18:]}]})
        calls = []

        def synthesize(spoken, voice, path, *, seed, direction='', performance=None, traits=''):
            prepared, compiled, metadata = prepare_synthesis(spoken, direction, performance=performance, traits=traits)
            calls.append(compiled)
            from test_renderer_workflow import write_tone
            write_tone(path, frequency=220 + len(calls) * 31)
            self.registry[hashlib.sha256(pcm(path)).hexdigest()] = prepared
            return {**metadata, 'synthesis_text': compiled}

        self.engine.synthesize = synthesize
        original = engines.compile_performance

        def earlier_form(*args, **kwargs):
            result = original(*args, **kwargs)
            return dict(result, text=result['text'].replace('[light Irish accent][dry, faintly teasing]',
                                                            '[dry, faintly teasing]'))

        with patch.object(engines, 'compile_performance', earlier_form):
            _, first = self.render_chunk()
        self.assertTrue(first['report']['passed'])
        self.assertIn('Lady. [dry, faintly teasing] Or did', calls[-1])
        _, again = self.render_chunk()
        self.assertTrue(again['report']['passed'])
        self.assertEqual(len(calls), 2, 'the line rendered again in the current form')
        self.assertIn('Lady. [light Irish accent][dry, faintly teasing] Or did', calls[-1])
        self.assertTrue(calls[-1].startswith('[light Irish accent][kind] Forgive me'))

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



SCOPED = 'Arrgh the pain has returned. Ce’Nedra waited outside.'
FIRST_SENTENCE, SECOND_SENTENCE = 'Arrgh the pain has returned. ', 'Ce’Nedra waited outside.'
READING = 'S EH2 N EH1 D R AH0'


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class ScopedSentenceSplitTests(unittest.TestCase):
    """A silent line with a chosen pronunciation splits, each part carrying its own controls."""

    setUpFixture = clip_repair_tests.BroadRepairTests.setUpFixture
    render_chunk = clip_repair_tests.BroadRepairTests.render_chunk
    accepted_record = clip_repair_tests.BroadRepairTests.accepted_record

    def setUp(self):
        self.setUpFixture()
        self.client = clip_repair_tests.BroadClient()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        text = SCOPED
        start = text.index('Ce’Nedra')
        self.chunk = speech_request(self.voice, text)['chunk']
        self.chunk['segment'].update(text=text, start=0, end=len(text), delivery={'direction': '', 'evidence': []})
        self.chunk.update(spoken_text=text, spoken_start=0, spoken_end=len(text),
                          source_ranges=[{'start': 0, 'end': len(text)}], chunk_id='scoped-split',
                          expressive_direction='', expressive_traits={},
                          expressive_performance={'version': 1, 'text_sha256': text_sha256(text), 'controls': [
                              {'kind': 'pronunciation', 'start': start, 'end': start + 8, 'text': 'Ce’Nedra',
                               'alphabet': 'cmu_arpabet', 'phonemes': READING}]})

        def synthesize(text, voice, path, *, seed, direction='', performance=None, traits=''):
            prepared, compiled, metadata = prepare_synthesis(text, direction, performance=performance, traits=traits)
            self.engine.calls.append({'text': text, 'compiled': compiled, 'performance': deepcopy(performance)})
            if text == SCOPED:
                raise EngineError("Drama 3 did not finish the text: 'silence'")
            write_tone(path, frequency=220 + len(self.engine.calls) * 31)
            self.registry[hashlib.sha256(pcm(path)).hexdigest()] = text.strip()
            return {'fixture': True, **metadata, 'synthesis_text': compiled}

        self.engine.synthesize = synthesize

    def test_silent_whole_takes_split_into_parts_with_their_own_controls(self):
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual([call['text'] for call in self.engine.calls],
                         [SCOPED, SCOPED, FIRST_SENTENCE, SECOND_SENTENCE])
        self.assertEqual(self.engine.calls[2]['performance']['controls'], [])
        self.assertEqual(self.engine.calls[3]['performance']['controls'], [
            {'kind': 'pronunciation', 'start': 0, 'end': 8, 'text': 'Ce’Nedra', 'alphabet': 'cmu_arpabet',
             'phonemes': READING}])
        self.assertIn(f'<|phoneme_start|>{READING}<|phoneme_end|>', self.engine.calls[3]['compiled'])
        self.assertEqual(self.client.repair_calls, [])
        record = self.accepted_record(result)
        self.assertEqual(record['request']['recovery']['strategy'], 'sentence_split')
        # Each part was verified against its own slice of the controls.
        parts = {read_json(path)['request']['synthesis_request']['chunk']['spoken_text']:
                 read_json(path)['request']['synthesis_request']['chunk']['expressive_performance']
                 for path in Path(result['wav_path']).parent.glob('parts/*/accepted.json')}
        self.assertEqual(parts[SECOND_SENTENCE], self.engine.calls[3]['performance'])
        # A rerun reuses the join without new takes.
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual(len(self.engine.calls), 4)

    def test_a_split_that_would_cut_a_pronunciation_is_not_offered(self):
        from epub_tts_narrator.renderer.speech_recovery import scoped_split_parts
        self.assertEqual([part['text'] for part in scoped_split_parts(self.chunk)], [FIRST_SENTENCE, SECOND_SENTENCE])
        control = self.chunk['expressive_performance']['controls'][0]
        control.update(start=20, end=len(SCOPED) - 3, text=SCOPED[20:len(SCOPED) - 3])
        self.assertIsNone(scoped_split_parts(self.chunk))
