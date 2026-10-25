import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from epub_tts_narrator.voice_profile_state import ProfileCheckpoint, read_state, save_state
from epub_tts_narrator.voice_profile_refresh import validate_profile_response
from test_voice_profile_refresh import RESPONSE


class CheckpointRequests:
    def __init__(self):
        self.client = SimpleNamespace(identity={'model': 'test'})
        self.calls = []

    def run(self, system, payload, validate, *, cancelled=None):
        self.calls.append(copy.deepcopy(payload))
        return validate(copy.deepcopy(RESPONSE))


class VoiceProfileStateTests(unittest.TestCase):
    def test_corrupt_prepared_proposal_is_revalidated_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = CheckpointRequests()
            binding = {'voice_ids': ['v1']}
            payload = {'voice': {'id': 'v1'}}
            checkpoint = ProfileCheckpoint(Path(directory), requests, binding)
            checkpoint.run('system', payload, validate_profile_response)
            state = read_state(directory)
            state['voices']['v1']['proposal']['speechTraits'] = ''
            save_state(directory, state)
            resumed = ProfileCheckpoint(Path(directory), requests, binding)
            with self.assertRaises(ValueError):
                resumed.run('system', payload, validate_profile_response)
            self.assertEqual(len(requests.calls), 1)

    def test_state_symlink_is_rejected_without_changing_its_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'other.json'
            target.write_text('untouched')
            (Path(directory) / '.voice-profile-refresh.json').symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'regular'):
                ProfileCheckpoint(Path(directory), CheckpointRequests(), {'voice_ids': ['v1']})
            self.assertEqual(target.read_text(), 'untouched')

    def test_known_cast_and_parent_inputs_require_new_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = CheckpointRequests()
            binding = {'voice_ids': ['v1']}
            payload = {'voice': {'id': 'v1'}, 'known_cast': [{'description': 'original'}],
                       'prior_series_voice': {'speechTraits': 'original'}}
            checkpoint = ProfileCheckpoint(Path(directory), requests, binding)
            checkpoint.run('system', payload, validate_profile_response)
            checkpoint = ProfileCheckpoint(Path(directory), requests, binding)
            checkpoint.run('system', payload, validate_profile_response)
            self.assertEqual(len(requests.calls), 1)
            payload['known_cast'][0]['description'] = 'changed'
            checkpoint.run('system', payload, validate_profile_response)
            self.assertEqual(len(requests.calls), 2)
            payload['prior_series_voice']['speechTraits'] = 'changed'
            checkpoint.run('system', payload, validate_profile_response)
            self.assertEqual(len(requests.calls), 3)
