from copy import deepcopy
import shutil
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import file_hash, read_json, write_json
from epub_tts_narrator.renderer.clip_repair_state import ClipRecoveryUnavailable
from epub_tts_narrator.renderer.exhaustion_selection import select_retained, validate_record
import test_renderer_clip_repair as repair_fixtures
import test_renderer_exhaustion_candidates as candidate_fixtures
from test_renderer_workflow import FFMPEG, FFPROBE


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class NormalizationCheckpointTests(unittest.TestCase):
    setUp = repair_fixtures.BroadRepairTests.setUp
    setUpFixture = repair_fixtures.BroadRepairTests.setUpFixture
    render_chunk = repair_fixtures.BroadRepairTests.render_chunk
    accepted_record = repair_fixtures.BroadRepairTests.accepted_record

    def test_published_unbound_normalization_resumes_from_raw_and_retains_orphan(self):
        from epub_tts_narrator.renderer.clip_repair_execute import normalize_clip

        def interrupted(raw, output, **options):
            normalize_clip(raw, output, **options)
            stable = raw.parent / 'part-001.wav'
            if output != stable:
                shutil.copyfile(output, stable)
            raise KeyboardInterrupt('offline interruption after normalized WAV publication')

        with patch('epub_tts_narrator.renderer.clip_repair_execute.normalize_clip', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        state_path = next((self.work / 'clips').glob('*/clip-repair.json'))
        before = read_json(state_path)['parts'][0]
        folder = state_path.parent / 'clip-repair-parts'
        orphan_hash = file_hash(folder / 'part-001.wav')
        self.assertIn('raw_sha256', before)
        self.assertNotIn('audio_sha256', before)
        self.assertEqual(len(self.engine.calls), 4)

        _, accepted = self.render_chunk()

        self.assertTrue(accepted['report']['passed'])
        after = read_json(state_path)['parts'][0]
        self.assertEqual(after['raw_sha256'], before['raw_sha256'])
        self.assertEqual(after['starts'], 1)
        self.assertEqual(len(self.engine.calls), 5)
        self.assertEqual(len(self.client.repair_calls), 1)
        self.assertEqual(len(self.client.review_calls), 2)
        self.assertEqual([file_hash(path) for path in folder.glob('part-001-orphan-*.wav')], [orphan_hash])


class FormatClient:
    identity = {'model': 'offline-choice-format', 'base_url': 'https://fixture.invalid'}

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def complete(self, prompt, payload):
        self.calls.append(deepcopy(payload))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index](payload)


def valid_choice(payload):
    return {'candidate_id': payload['candidates'][0]['candidate_id'], 'reason': 'Best available rendition.'}


class ChooserFormatRecoveryTests(unittest.TestCase):
    setUp = candidate_fixtures.ExhaustionCandidatesTests.setUp
    candidate = candidate_fixtures.ExhaustionCandidatesTests.candidate

    def test_unknown_id_and_malformed_reason_get_bound_format_corrections(self):
        self.candidate(1, transcript='Do leave.')
        failed_path = self.directory / 'attempt-001.json'
        before = failed_path.read_bytes()
        client = FormatClient([
            lambda _: {'candidate_id': 'unknown', 'reason': 'A choice.'},
            lambda payload: {**valid_choice(payload), 'reason': None},
            valid_choice,
        ])

        wav, record = select_retained(self.directory, self.request, client)

        self.assertEqual(len(client.calls), 3)
        state = read_json(self.directory / 'exhaustion-selection.json')
        self.assertEqual(len(state['invalid_responses']), 2)
        for index, audit in enumerate(state['invalid_responses']):
            self.assertEqual(audit['payload'], client.calls[index])
        self.assertEqual(record['engine_metadata']['exhaustion_selection']['audit']['payload'], client.calls[-1])
        self.assertIn('format_correction', client.calls[-1])
        validate_record(record)
        changed = deepcopy(record)
        changed['engine_metadata']['exhaustion_selection']['audit']['payload']['format_correction']['instruction'] = 'changed'
        with self.assertRaises(ValueError):
            validate_record(changed)
        self.assertEqual(failed_path.read_bytes(), before)
        self.assertEqual(select_retained(self.directory, self.request, client), (wav, record))
        self.assertEqual(len(client.calls), 3)

    def test_repeated_format_failure_is_bounded_pending_and_resumable(self):
        self.candidate(1, transcript=None)
        client = FormatClient([lambda _: {'candidate_id': []}])
        with self.assertRaises(ClipRecoveryUnavailable):
            select_retained(self.directory, self.request, client)
        self.assertEqual(len(client.calls), 3)
        state = read_json(self.directory / 'exhaustion-selection.json')
        self.assertFalse(state['response_received'])
        self.assertEqual(len(state['invalid_responses']), 3)
        client.responses = [valid_choice]

        _, record = select_retained(self.directory, self.request, client)

        self.assertTrue(record['report']['passed'])
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(read_json(self.directory / 'exhaustion-selection.json')['invalid_responses'], state['invalid_responses'])

    def test_format_correction_cannot_change_an_identifiable_selected_candidate(self):
        self.candidate(1, transcript='Do leave.')
        self.candidate(2, transcript='Do not leave.')
        client = FormatClient([
            lambda payload: {**valid_choice(payload), 'reason': ''},
            lambda payload: {'candidate_id': payload['candidates'][1]['candidate_id'], 'reason': 'Different choice.'},
            valid_choice,
        ])

        _, record = select_retained(self.directory, self.request, client)

        self.assertEqual(record['attempt'], 1)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(record['report']['original_failed_report']['transcript'], 'Do leave.')

    def test_valid_first_choice_is_not_reconsidered(self):
        self.candidate(1, transcript='Do leave.')
        client = FormatClient([valid_choice])
        first = select_retained(self.directory, self.request, client)
        state_path = self.directory / 'exhaustion-selection.json'
        before = state_path.read_bytes()
        self.assertEqual(select_retained(self.directory, self.request, client), first)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(state_path.read_bytes(), before)
        self.assertNotIn('format_correction', first[1]['engine_metadata']['exhaustion_selection']['audit']['payload'])

    def test_persisted_legacy_invalid_response_is_corrected_without_repeating_it(self):
        self.candidate(1, transcript='Do leave.')
        client = FormatClient([lambda _: {'candidate_id': 'unknown', 'reason': 'A choice.'}, valid_choice])

        def interrupt(path, value):
            write_json(path, value)
            if path.name == 'exhaustion-selection.json' and value['response_received']:
                raise KeyboardInterrupt('offline interruption after response checkpoint')

        with patch('epub_tts_narrator.renderer.exhaustion_selection.write_json', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                select_retained(self.directory, self.request, client)
        state = read_json(self.directory / 'exhaustion-selection.json')
        self.assertTrue(state['response_received'])
        self.assertNotIn('invalid_responses', state)

        _, record = select_retained(self.directory, self.request, client)

        self.assertTrue(record['report']['passed'])
        self.assertEqual(len(client.calls), 2)
        saved = read_json(self.directory / 'exhaustion-selection.json')
        self.assertEqual(saved['invalid_responses'], [state['audit']])

    def test_pending_format_correction_rejects_provider_or_candidate_inventory_changes(self):
        self.candidate(1, transcript='Do leave.')
        client = FormatClient([lambda _: {'candidate_id': []}])
        with self.assertRaises(ClipRecoveryUnavailable):
            select_retained(self.directory, self.request, client)
        state_path = self.directory / 'exhaustion-selection.json'
        before = state_path.read_bytes()
        identity = client.identity
        client.identity = dict(identity, model='different-provider-model')
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            select_retained(self.directory, self.request, client)
        client.identity = identity
        candidate_path = self.directory / 'attempt-001.json'
        candidate = read_json(candidate_path)
        candidate['report']['transcript'] = 'Changed recognition'
        write_json(candidate_path, candidate)
        with self.assertRaisesRegex(ValueError, 'history changed|inputs changed'):
            select_retained(self.directory, self.request, client)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(state_path.read_bytes(), before)
