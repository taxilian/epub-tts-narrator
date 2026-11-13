from copy import deepcopy
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
import tempfile
import unittest
import wave
from unittest.mock import patch

from epub_tts_narrator.renderer import clip_recovery
from epub_tts_narrator.renderer.clip_repair_state import ClipRepairState
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver, _accept_pairs
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import VerificationUnavailable
import test_renderer_direction_repair as fixtures
from test_renderer_recurring_resolution import PlanClient
from test_renderer_speech_identity import reference, speech_request
from test_renderer_workflow import FFMPEG, FFPROBE


class BroadClient(PlanClient):
    def __init__(self):
        super().__init__([{'action': 'stop', 'edits': [], 'reason': 'Old actions exhausted.'}])
        self.plan = {'parts': [{'text': 'Hum.', 'delivery': 'thoughtful', 'speech_traits': ''},
                               {'text': 'Mmm.', 'delivery': 'drawled', 'speech_traits': 'low tone'}],
                     'reason': 'Try alternate phonetics and separate performances.'}
        self.barrier = Barrier(2, timeout=5)
        self.lock = Lock()
        self.repair_calls, self.review_calls, self.selection_calls = [], [], []
        self.reject = None
        self.outage = False
        self.review_outage = None

    def complete(self, prompt, payload):
        mode = payload.get('mode')
        if mode == 'clip_repair':
            self.repair_calls.append(deepcopy(payload))
            return deepcopy(self.plan)
        if mode == 'clip_repair_review':
            with self.lock:
                self.review_calls.append(deepcopy(payload))
                number = len(self.review_calls)
            if number <= 2:
                self.barrier.wait()
            if payload['reviewer'] == self.review_outage:
                raise TimeoutError('offline reviewer outage')
            return {'approve': payload['reviewer'] != self.reject, 'reason': 'Reasonable clip-local experiment.'}
        if mode == 'exhaustion_selection':
            self.selection_calls.append(deepcopy(payload))
            if self.outage:
                raise TimeoutError('offline chooser outage')
            return {'candidate_id': payload['candidates'][-1]['candidate_id'], 'reason': 'Best available rendition.'}
        return super().complete(prompt, payload)


class TerminalClipRepairResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.voice = reference('character')
        self.request = speech_request(self.voice, 'Hello.')
        self.client = SimpleNamespace(identity={'model': 'offline'})
        self.verifier = SimpleNamespace(
            speech_resolver=SimpleNamespace(clip_recovery_client=self.client))
        self.config = {'verification': {'max_resolution_attempts': 12}}
        self.calls = []

    def ordinary(self, text, generate, directory, request, verifier, config, **options):
        self.calls.append((text, directory, request))
        return 'ordinary-result'

    def save_state(self, phase, parts=()):
        state = ClipRepairState(self.directory, self.request, self.client)
        state.state.update(phase=phase, parts=list(parts))
        state.save()

    def test_terminal_repair_states_allow_ordinary_recovery(self):
        for phase in ('rejected', 'failed', 'evaluated'):
            with self.subTest(phase=phase):
                self.calls.clear()
                self.save_state(phase)
                result = clip_recovery.run(
                    self.ordinary, 'Hello.', lambda *args: None,
                    self.directory, self.request, self.verifier, self.config,
                    generate_plan_part=None)
                self.assertEqual(result, 'ordinary-result')
                self.assertEqual(len(self.calls), 1)


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class BroadRepairTests(unittest.TestCase):
    setUpFixture = fixtures.DirectionRepairTests.setUp
    render_chunk = fixtures.DirectionRepairTests.render_chunk
    accepted_record = fixtures.DirectionRepairTests.accepted_record

    def setUp(self):
        self.setUpFixture()
        self.client = BroadClient()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.engine.heard_override = lambda text, direction, number: text if number > 3 else 'Wrong instruction'

    def test_combined_phonetic_trait_delivery_and_split_plan_has_two_overlapping_approvals(self):
        original = deepcopy(self.chunk)
        result, acceptance = self.render_chunk()
        self.assertEqual(self.chunk, original)
        self.assertEqual([call['text'] for call in self.engine.calls][-2:], ['Hum.', 'Mmm.'])
        self.assertEqual([call['direction'] for call in self.engine.calls][-2:], ['thoughtful', 'low tone; drawled'])
        self.assertTrue(all(call['voice'] == self.voice for call in self.engine.calls))
        self.assertEqual(sorted(row['reviewer'] for row in self.client.review_calls), [1, 2])
        self.assertTrue(all('reviews' not in row for row in self.client.review_calls))
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(acceptance['report']['text_confirmation_method'], 'llm_exhaustion_selection')
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        for part in state['parts']:
            self.assertIn(self.chunk['chunk_id'], part['raw_file'])
            self.assertIn(self.chunk['chunk_id'], part['audio_file'])
        candidate = next(read_json(path) for path in (self.work / 'clips').glob('*/attempt-*.json')
                         if 'clip_repair' in read_json(path).get('engine_metadata', {}))
        self.assertIn(self.chunk['chunk_id'], candidate['audio_file'])
        self.assertTrue(candidate['audio_file'].endswith('-take.wav'))
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual(len(self.engine.calls), 5)

    def test_final_selection_is_decisive_when_old_lexical_gate_rejects(self):
        self.chunk = speech_request(self.voice, 'Do not leave.')['chunk']
        self.client.plan = {'parts': [], 'reason': 'No further useful synthesis plan.'}
        self.engine.heard_override = lambda *args: 'Do leave.'
        self.assertIsNone(_accept_pairs('Do not leave.', 'Do leave.'))
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        record = self.accepted_record(result)
        self.assertFalse(record['report']['original_failed_report']['passed'])
        attempts = list((self.work / 'clips').glob('*/attempt-*.json'))
        before = {path: path.read_bytes() for path in attempts}
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual({path: path.read_bytes() for path in attempts}, before)
        self.assertEqual(len(self.engine.calls), 3)
        self.assertTrue(all(not read_json(path)['report']['passed'] for path in attempts))

    def test_one_negative_review_blocks_plan_but_chooser_still_selects_retained_audio(self):
        self.client.reject = 1
        _, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.client.review_calls), 2)
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.selection_calls), 1)

    def test_parallel_review_outage_checkpoints_completed_peer_and_resumes_only_missing_call(self):
        self.client.review_outage = 2
        with self.assertRaises(VerificationUnavailable):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertIsNotNone(state['audit']['reviews'][0])
        self.assertIsNone(state['audit']['reviews'][1])
        self.client.review_outage = None
        self.render_chunk()
        self.assertEqual([row['reviewer'] for row in self.client.review_calls].count(1), 1)
        self.assertEqual([row['reviewer'] for row in self.client.review_calls].count(2), 2)
        self.assertEqual(len(self.client.repair_calls), 1)

    def test_pending_final_selection_resumes_without_tts_or_failed_record_changes(self):
        self.client.outage = True
        with self.assertRaises(VerificationUnavailable):
            self.render_chunk()
        attempts = list((self.work / 'clips').glob('*/attempt-*.json'))
        before = {path: path.read_bytes() for path in attempts}
        calls = len(self.engine.calls)
        self.client.outage = False
        _, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(len(self.engine.calls), calls)
        self.assertEqual({path: path.read_bytes() for path in attempts}, before)
        self.assertEqual(len(self.client.repair_calls), 1)

    def test_acoustic_advisory_does_not_veto_decisive_selection(self):
        self.config['quality']['min_duration_seconds'] = 2.5
        self.client.plan = {'parts': [], 'reason': 'Use the best existing performance.'}
        _, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertIn('audio_too_short', acceptance['report']['original_failed_report']['reasons'])

    def test_changed_candidate_bytes_block_pending_selection_without_new_synthesis(self):
        self.client.outage = True
        with self.assertRaises(VerificationUnavailable):
            self.render_chunk()
        metadata = next((self.work / 'clips').glob('*/attempt-001.json'))
        wav = metadata.parent / read_json(metadata)['audio_file']
        wav.write_bytes(wav.read_bytes() + b'changed')
        calls = len(self.engine.calls)
        self.client.outage = False
        with self.assertRaisesRegex(ValueError, 'audio changed'):
            self.render_chunk()
        self.assertEqual(len(self.engine.calls), calls)

    def test_existing_stop_enters_broad_repair_without_more_ordinary_failures_or_audit_changes(self):
        from legacy_recovery_fixtures import LegacySpeechResolver
        from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
        legacy = LegacySpeechResolver(self.client)
        self.verifier = RecurringSpeechVerifier(self.base, legacy)
        with patch.object(legacy, 'direction_repair', return_value=None):
            with self.assertRaises(SpeechVerificationFailed):
                self.render_chunk()
        path = next((self.work / 'clips').glob('*/speech-resolution.json'))
        before = path.read_bytes()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.render_chunk()
        self.assertEqual(len(self.engine.calls), 5)
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(path.read_bytes(), before)

    def test_partial_split_plan_resumes_completed_parts_without_new_proposal_or_reviews(self):
        synthesize = self.engine.synthesize

        def interrupt(*args, **kwargs):
            result = synthesize(*args, **kwargs)
            if len(self.engine.calls) == 5:
                raise KeyboardInterrupt('offline partial-plan interruption')
            return result

        with patch.object(self.engine, 'synthesize', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        first_hash = state['parts'][0]['audio_sha256']
        self.render_chunk()
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertEqual(state['parts'][0]['audio_sha256'], first_hash)
        self.assertEqual(len(self.engine.calls), 6)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)

    def test_operationally_failed_repair_resumes_approved_plan_without_new_model_calls(self):
        from epub_tts_narrator.renderer.workflow import RenderBlocked
        synthesize = self.engine.synthesize

        def transport_failure(*args, **kwargs):
            synthesize(*args, **kwargs)
            raise RuntimeError('offline transport failure')

        with patch.object(self.engine, 'synthesize', side_effect=transport_failure):
            with self.assertRaises(RenderBlocked):
                self.render_chunk()
        path = next((self.work / 'clips').glob('*/clip-repair.json'))
        state = read_json(path)
        self.assertEqual(state['phase'], 'failed')
        self.assertEqual(state['error_code'], 'synthesis_or_input_failure')
        calls_before = len(self.engine.calls)
        self.render_chunk()
        state = read_json(path)
        self.assertEqual(state['phase'], 'evaluated')
        self.assertTrue(list((self.work / 'clips').glob('*/accepted.json')))
        self.assertEqual(len(self.engine.calls), calls_before + 2)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)

    def test_plan_cannot_spend_more_than_the_shared_cap(self):
        self.config['verification']['max_resolution_attempts'] = 4
        self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(len(self.client.review_calls), 0)

    def test_physically_silent_audio_is_never_offered_to_the_chooser(self):
        self.client.plan = {'parts': [], 'reason': 'No further useful rendering plan.'}
        synthesize = self.engine.synthesize

        def silent(text, voice, path, **kwargs):
            metadata = synthesize(text, voice, path, **kwargs)
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 44100, 0, 'NONE', ''))
                audio.writeframes(bytes(88200))
            return metadata

        with patch.object(self.engine, 'synthesize', side_effect=silent):
            with self.assertRaisesRegex(ValueError, 'No physically usable'):
                self.render_chunk()
        self.assertEqual(self.client.selection_calls, [])
        self.assertFalse(list((self.work / 'clips').glob('*/accepted.json')))

    def test_rejected_terminal_repair_allows_fresh_ordinary_generation_on_rerun(self):
        from legacy_recovery_fixtures import LegacySpeechResolver
        from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
        self.client.plan = {'parts': [], 'reason': 'No further useful rendering plan.'}
        synthesize = self.engine.synthesize

        def silent(text, voice, path, **kwargs):
            metadata = synthesize(text, voice, path, **kwargs)
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 44100, 0, 'NONE', ''))
                audio.writeframes(bytes(88200))
            return metadata

        self.verifier = RecurringSpeechVerifier(self.base, LegacySpeechResolver(self.client))
        with patch.object(self.engine, 'synthesize', side_effect=silent):
            with self.assertRaises(SpeechVerificationFailed):
                self.render_chunk()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        with patch.object(self.engine, 'synthesize', side_effect=silent):
            with self.assertRaisesRegex(ValueError, 'No physically usable'):
                self.render_chunk()
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertEqual(state['phase'], 'rejected')
        resolution = next((self.work / 'clips').glob('*/speech-resolution.json'))
        self.assertTrue(resolution.is_file())
        saved_before = {path: path.read_bytes()
                        for path in (self.work / 'clips').glob('*/*.json')}
        calls = len(self.engine.calls)
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertGreater(len(self.engine.calls), calls)
        self.assertEqual({path: path.read_bytes() for path in saved_before}, saved_before)

    def test_failed_terminal_repair_without_part_audio_allows_fresh_ordinary_generation_on_rerun(self):
        from epub_tts_narrator.renderer.workflow import RenderBlocked
        self.client.plan = {'parts': [{'text': 'Hum.', 'delivery': 'thoughtful', 'speech_traits': ''}],
                            'reason': 'Try a clip-local repair.'}
        with patch.object(self.engine, 'synthesize', side_effect=RuntimeError('offline part failure')):
            with self.assertRaises(RenderBlocked):
                self.render_chunk()
        state = read_json(next((self.work / 'clips').glob('*/clip-repair.json')))
        self.assertEqual(state['phase'], 'failed')
        self.assertEqual(state['error_code'], 'synthesis_or_input_failure')
        self.assertFalse(list((self.work / 'clips').glob('*/clip-repair-parts/*')))
        self.engine.heard_override = lambda text, direction, number: text
        calls = len(self.engine.calls)
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertGreater(len(self.engine.calls), calls)
        self.assertNotIn('clip_repair', self.accepted_record(result)['engine_metadata'])

    def test_selection_proof_rejects_changed_reference_audio_or_original_report(self):
        from epub_tts_narrator.renderer.exhaustion_selection import validate_record
        result, _ = self.render_chunk()
        record = self.accepted_record(result)
        mutations = [lambda row: row.update(audio_sha256='0' * 64),
                     lambda row: row['request'].update(voice_asset_hash='0' * 64),
                     lambda row: row['report']['original_failed_report'].update(transcript='different'),
                     lambda row: row['engine_metadata']['exhaustion_selection']['audit']['payload']['request'].update(voice_asset_hash='changed'),
                     lambda row: row['engine_metadata']['clip_repair']['parts'][0]['part'].update(speech_traits='changed')]
        for mutate in mutations:
            changed = deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_record(changed)

    def test_broad_repair_uses_ordinary_verification_before_any_final_selection(self):
        self.client.plan = {'parts': [{'text': 'Hummm.', 'delivery': 'drawled', 'speech_traits': ''}],
                            'reason': 'Try a sustained phonetic input.'}
        self.engine.heard_override = lambda text, direction, number: 'Hmm.' if number > 3 else 'Wrong instruction'
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance['report']['passed'])
        self.assertEqual(self.client.selection_calls, [])
        self.assertIn('clip_repair', self.accepted_record(result)['engine_metadata'])
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual(len(self.engine.calls), 4)

    def test_existing_v1_acceptance_remains_usable_by_new_policy_without_new_calls(self):
        from legacy_recovery_fixtures import LegacySpeechResolver
        old_client = fixtures.DirectionClient()
        self.verifier = RecurringSpeechVerifier(self.base, LegacySpeechResolver(old_client))
        self.engine.heard_override = lambda text, direction, number: text if number > 3 else 'Wrong instruction'
        result, acceptance = self.render_chunk()
        path = next((self.work / 'clips').glob('*/direction-repair.json'))
        before = path.read_bytes()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.assertEqual(self.render_chunk(), (result, acceptance))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.client.repair_calls, [])
        self.assertEqual(len(self.engine.calls), 4)


class EarlierTagFormatTests(unittest.TestCase):
    def test_a_part_recorded_in_the_earlier_one_bracket_form_still_proves_the_approved_plan(self):
        # A render stopped on these: the take was valid, only its recorded tag format was older.
        from epub_tts_narrator.renderer.clip_repair_contract import confirm_part, part_direction
        from epub_tts_narrator.renderer.engines import _directed_text
        part = {'text': 'Items with an asterisk', 'speech_traits': 'Western American accent',
                'delivery': 'Flat, matter-of-fact note. Begin on the first word.'}
        direction = part_direction(part)
        directed, control = _directed_text(part['text'], direction)
        self.assertEqual(directed, '[Western American accent][Flat, matter-of-fact note. Begin on the first word.] '
                                   'Items with an asterisk')
        fields = {key: control[key] for key in ('direction', 'direction_applied', 'applied_control')}
        for recorded in (directed, f'[{direction}] Items with an asterisk'):
            confirm_part({}, part, dict(fields, synthesis_text=recorded))
        for recorded in ('[Western American accent] Items with an asterisk', f'[{direction}] Items with a star'):
            with self.subTest(recorded=recorded), self.assertRaises(ValueError):
                confirm_part({}, part, dict(fields, synthesis_text=recorded))

    def test_a_controlled_part_recorded_in_the_earlier_form_still_proves_the_approved_plan(self):
        # Its compiler audit carries that earlier engine text too.
        from epub_tts_narrator.renderer.clip_repair_contract import confirm_part, part_direction
        from epub_tts_narrator.renderer.engines import prepare_synthesis
        from test_performance_plan import pronunciation, record
        text = 'Jas spoke.'
        performance = record(text, pronunciation(text, 0, 3))
        part = {'text': text, 'speech_traits': 'clear articulation, unhurried', 'delivery': 'Say the name slowly.'}
        direction = part_direction(part)
        _, directed, control = prepare_synthesis(text, direction, performance=performance)
        earlier = f'[{direction}] ' + directed.split('] ', 1)[1]
        self.assertTrue(directed.startswith('[clear articulation][unhurried][Say the name slowly.] '))
        fields = {key: control[key] for key in ('direction', 'direction_applied', 'applied_control')}
        request = {'kind': 'paragraph_speech'}
        confirm_part(request, part, dict(fields, synthesis_text=directed,
                                         expressive_performance=control['expressive_performance']),
                     performance=performance)
        confirm_part(request, part, dict(fields, synthesis_text=earlier,
                                         expressive_performance=dict(control['expressive_performance'], text=earlier)),
                     performance=performance)
        with self.assertRaises(ValueError):  # the earlier text with an audit that does not match it
            confirm_part(request, part, dict(fields, synthesis_text=earlier,
                                             expressive_performance=control['expressive_performance']),
                         performance=performance)

    def test_a_chosen_or_recovered_take_recorded_in_the_earlier_form_renders_again(self):
        # Valid records, but not asked in the current form: the line renders again in separate tags.
        from epub_tts_narrator.renderer.speech_identity import recorded_in_earlier_form
        voice = reference()
        request = speech_request(voice, 'Wear a skirt.')
        request['chunk']['speech_traits'] = {voice['voice_id']: 'thick New York accent'}
        request['chunk']['segment']['delivery'] = {'direction': 'dry, mildly reproachful'}
        earlier = '[thick New York accent; dry, mildly reproachful] Wear a skirt.'
        current = '[thick New York accent][dry][mildly reproachful] Wear a skirt.'

        def chosen(text):
            return {'exhaustion_selection': {'selected_record': {
                'request': request, 'engine_metadata': {'synthesis_text': text}}}}

        def recovered(text):
            return {'synthesis_text': 'Wear a skirt.', 'recovery': {'parts': [{'acceptance': {
                'request': request, 'engine_metadata': {'synthesis_text': text}}}]}}

        for wrap in (chosen, recovered):
            with self.subTest(wrap=wrap.__name__):
                self.assertTrue(recorded_in_earlier_form(request, wrap(earlier)))
                self.assertFalse(recorded_in_earlier_form(request, wrap(current)))
        # A person's own pick is kept.
        self.assertFalse(recorded_in_earlier_form(request, {'manual_selection': {}, 'synthesis_text': earlier}))
        # A line whose direction has one part reads the same in both forms.
        request['chunk']['segment']['delivery'] = {}
        self.assertFalse(recorded_in_earlier_form(request, chosen('[thick New York accent] Wear a skirt.')))

    def test_a_repair_part_reports_which_form_it_was_recorded_in(self):
        from epub_tts_narrator.renderer.clip_repair_contract import confirm_part, part_direction
        from epub_tts_narrator.renderer.engines import _directed_text
        part = {'text': 'Hush.', 'speech_traits': 'raspy', 'delivery': 'dry, low'}
        directed, control = _directed_text('Hush.', part_direction(part))
        fields = {key: control[key] for key in ('direction', 'direction_applied', 'applied_control')}
        self.assertFalse(confirm_part({}, part, dict(fields, synthesis_text=directed)))
        self.assertTrue(confirm_part({}, part, dict(fields, synthesis_text='[raspy; dry, low] Hush.')))
