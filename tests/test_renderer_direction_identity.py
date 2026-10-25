from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.direction_repair import validate_marker
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.speech_traits import performance_direction
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget, SpeechVerificationFailed, verified_take
import test_renderer_direction_repair as fixtures
from test_renderer_speech_identity import speech_request
from test_renderer_workflow import FFMPEG, FFPROBE


@unittest.skipUnless(FFMPEG and FFPROBE, 'FFmpeg and ffprobe required')
class DirectionIdentityTests(unittest.TestCase):
    setUp = fixtures.DirectionRepairTests.setUp
    render_chunk = fixtures.DirectionRepairTests.render_chunk
    accepted_record = fixtures.DirectionRepairTests.accepted_record

    def test_original_source_traits_voice_and_normalization_are_bound(self):
        result, _ = self.render_chunk()
        record = self.accepted_record(result)
        original = record['request']
        metadata = record['engine_metadata']
        assets = {self.voice['asset_hash']: self.voice}
        self.assertEqual(generation_identity(original, assets),
                         generation_identity(original, assets, engine_metadata=metadata))
        mutations = [
            lambda row: row.update(voice_asset_hash=digest('other reference')),
            lambda row: row['chunk'].update(spoken_text='Other words.'),
            lambda row: row['chunk']['speech_traits'].update({self.voice['voice_id']: 'different accent'}),
            lambda row: row['chunk']['segment']['delivery'].update(direction='excited'),
            lambda row: row['chunk']['segment']['speech_normalization']['edits'][0].update(direction='shouted'),
            lambda row: row['chunk']['segment']['speech_normalization']['edits'][0].update(reason='changed'),
            lambda row: row['engine'].update(temperature=0.5),
        ]
        for mutate in mutations:
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                generation_identity(changed, assets, engine_metadata=metadata)

    def test_actual_conditioning_cannot_omit_traits_or_change_words(self):
        result, _ = self.render_chunk()
        record = self.accepted_record(result)
        assets = {self.voice['asset_hash']: self.voice}
        for changes in ({'synthesis_text': 'Other words.'}, {'direction': fixtures.REPLACEMENT},
                        {'direction_applied': False}, {'applied_control': None},
                        {'speech_resolution': {}}, {'delivery_recovery': {}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                generation_identity(record['request'], assets,
                                    engine_metadata={**record['engine_metadata'], **changes})

    def test_missing_or_nonblind_reviews_and_old_authority_cannot_authorize_repair(self):
        result, _ = self.render_chunk()
        record = self.accepted_record(result)
        marker = record['engine_metadata']['direction_repair']
        mutations = [lambda row: row['reviews'].pop(),
                     lambda row: row['reviews'][0]['response'].update(equivalent=False),
                     lambda row: row['reviews'][1]['payload'].update(reviewer=1),
                     lambda row: row['proposal'].update(replacement_direction='[new control]'),
                     lambda row: row['identity'].update(name='plain_delivery'),
                     lambda row: row['payload']['failures'][0].update(transcript='changed evidence')]
        for mutate in mutations:
            changed = deepcopy(marker)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_marker(record['request'], changed)

    def test_child_repair_spends_shared_budget_without_consuming_sibling_reserve(self):
        request = speech_request(self.voice, self.chunk['spoken_text'])
        request.update(chunk=deepcopy(self.chunk), engine=self.engine.identity)
        child = {'kind': 'speech_recovery_part', 'version': 1,
                 'parent_request_hash': digest('parent'), 'recovery_request_hash': digest('recovery'),
                 'part': {'start': 0, 'end': 4, 'text': 'Hmm.'}, 'synthesis_request': request}

        def generate(path, seed):
            return self.engine.synthesize('Hmm.', self.voice, path, seed=seed,
                direction=performance_direction(self.chunk, self.voice['voice_id']))

        def repaired(text, replacement, path, seed):
            return self.engine.synthesize(text, self.voice, path, seed=seed,
                direction=performance_direction(self.chunk, self.voice['voice_id'], scene_direction=replacement))

        self.config['verification']['max_resolution_attempts'] = 4
        budget = SpeechRetryBudget()
        with self.assertRaises(SpeechVerificationFailed):
            verified_take('Hmm.', generate, self.work / 'child', child, self.verifier, self.config,
                          notify=lambda _: None, retry_budget=budget, resolution_episode='shared',
                          episode_reserve=1, generate_direction=repaired)
        self.assertEqual(budget.episode_started('shared'), 3)
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(budget.claim(self.work / 'sibling', digest('sibling'), 4,
                                     episode='shared', episode_maximum=4), 1)

    def test_successful_child_audit_validates_against_its_exact_synthesis_request(self):
        result, _ = self.render_chunk()
        record = self.accepted_record(result)
        request = record['request']
        child = {'kind': 'speech_recovery_part', 'version': 1,
                 'parent_request_hash': digest('parent'), 'recovery_request_hash': digest('recovery'),
                 'part': {'start': 9, 'end': 13, 'text': 'Hmm.'}, 'synthesis_request': request}
        marker = record['engine_metadata']['direction_repair']
        self.assertEqual(validate_marker(child, marker), fixtures.REPLACEMENT)
        child['part']['text'] = 'Other text'
        with self.assertRaises(ValueError):
            validate_marker(child, marker)
