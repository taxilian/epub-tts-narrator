"""Offline refresh checkpoints survive transport and publication failures."""

import json
from unittest.mock import patch
import unittest

import test_voice_profile_refresh as fixtures
from epub_tts_narrator.analysis_directory import continuation_directories
from epub_tts_narrator.renderer.bundle import load_bundle


class VoiceProfileResumeTests(unittest.TestCase):
    setUp = fixtures.VoiceProfileRefreshTests.setUp
    invoke_refresh = fixtures.VoiceProfileRefreshTests.invoke_refresh

    def state(self):
        return json.loads((self.paths[0] / '.voice-profile-refresh.json').read_text())

    def test_all_profiles_invalid_before_first_call_and_partial_restart_reuses_proposals(self):
        owner = self

        class InspectingClient(fixtures.Client):
            def complete(self, system, payload):
                if not self.calls:
                    owner.assertEqual({row['status'] for row in owner.state()['voices'].values()}, {'invalid'})
                    with owner.assertRaisesRegex(ValueError, 'refresh-voice-profiles.*book-1.*--config'):
                        load_bundle(owner.paths[2])
                return super().complete(system, payload)

        voices = [voice['id'] for voice in self.documents[0]['cast']['voices']]
        client = InspectingClient(fail_on=voices[1])
        self.assertEqual(self.invoke_refresh(client)[0], 1)
        self.assertEqual(self.state()['voices'][voices[0]]['status'], 'prepared')
        with self.assertRaisesRegex(ValueError, 'refresh-voice-profiles'):
            load_bundle(self.paths[2])
        # Remove request-cache responses: the explicit checkpoint must suffice.
        for path in (self.root / 'cache').glob('*.json'):
            path.unlink()
        resumed = fixtures.Client()
        self.assertEqual(self.invoke_refresh(resumed)[0], 0)
        self.assertEqual(resumed.calls, [voices[1]])
        self.assertFalse((self.paths[0] / '.voice-profile-refresh.json').exists())
        self.assertEqual(continuation_directories(self.paths[2]), list(reversed(self.paths[:2])))

    def test_publication_failure_keeps_prepared_proposals(self):
        from epub_tts_narrator.analysis_revision import _install
        before = fixtures.VoiceProfileRefreshTests.managed(self)
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('disk failure')
            return _install(source, destination)

        with patch('epub_tts_narrator.analysis_revision._install', side_effect=fail_second):
            self.assertEqual(self.invoke_refresh(fixtures.Client())[0], 1)
        self.assertEqual(fixtures.VoiceProfileRefreshTests.managed(self), before)
        self.assertEqual(self.state()['phase'], 'publishing')
        for path in (self.root / 'cache').glob('*.json'):
            path.unlink()
        resumed = fixtures.DeadClient()
        self.assertEqual(self.invoke_refresh(resumed)[0], 0)

    def test_marker_clear_failure_reconciles_publication_without_calls(self):
        from pathlib import Path
        unlink = Path.unlink

        def fail_marker(path, *args, **kwargs):
            if path.name == '.voice-profile-refresh.json':
                raise OSError('marker clear failed')
            return unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', fail_marker):
            self.assertEqual(self.invoke_refresh(fixtures.Client())[0], 1)
        self.assertEqual(self.state()['phase'], 'publishing')
        self.assertEqual(self.invoke_refresh(fixtures.DeadClient())[0], 0)
        self.assertFalse((self.paths[0] / '.voice-profile-refresh.json').exists())

    def test_active_descendant_render_refused_before_calls(self):
        (self.paths[2] / '.audio-render.lock').mkdir()
        client = fixtures.Client()
        self.assertEqual(self.invoke_refresh(client)[0], 1)
        self.assertEqual(client.calls, [])

    def test_active_namespaced_casting_refused_before_invalidation_or_calls(self):
        lock = self.paths[1] / 'voices' / 'series-namespace' / '.lock'
        lock.mkdir(parents=True)
        client = fixtures.Client()

        self.assertEqual(self.invoke_refresh(client)[0], 1)

        self.assertEqual(client.calls, [])
        self.assertFalse((self.paths[0] / '.voice-profile-refresh.json').exists())
        self.assertTrue(lock.is_dir())

    def prepare_failed_publication(self):
        with patch('epub_tts_narrator.analysis_revision._install', side_effect=OSError('disk failure')):
            self.assertEqual(self.invoke_refresh(fixtures.Client())[0], 1)
        for path in (self.root / 'cache').glob('*.json'):
            path.unlink()

    def test_provider_change_discards_prepared_proposals(self):
        self.prepare_failed_publication()
        client = fixtures.Client()
        client.identity = dict(client.identity, model='different-model')
        self.assertEqual(self.invoke_refresh(client)[0], 0)
        self.assertEqual(len(client.calls), len(self.documents[0]['cast']['voices']))

    def test_guidance_change_discards_prepared_proposals(self):
        self.prepare_failed_publication()
        client = fixtures.Client()
        with patch('epub_tts_narrator.voice_profile_refresh.SYSTEM', 'different guidance'):
            self.assertEqual(self.invoke_refresh(client)[0], 0)
        self.assertEqual(len(client.calls), len(self.documents[0]['cast']['voices']))

    def test_pending_conflict_refused_and_lease_files_retained(self):
        self.assertEqual(self.invoke_refresh(fixtures.DeadClient())[0], 1)
        lease = self.paths[0] / '.voice-profile-refresh.lock.lease'
        inode = lease.stat().st_ino
        self.assertFalse((self.paths[0] / '.voice-profile-refresh.lock').exists())
        (self.paths[2] / '.voice-profile-refresh.json').write_text('{}')
        client = fixtures.Client()
        self.assertEqual(self.invoke_refresh(client)[0], 1)
        self.assertEqual(client.calls, [])
        self.assertEqual(lease.stat().st_ino, inode)

    def test_preloaded_render_bundle_cannot_acquire_generation_lock_after_invalidation(self):
        from epub_tts_narrator.renderer.workflow import _lock_book_audio
        bundle = load_bundle(self.paths[2])
        self.assertEqual(self.invoke_refresh(fixtures.DeadClient())[0], 1)
        with self.assertRaisesRegex(ValueError, 'refresh-voice-profiles'):
            _lock_book_audio(bundle)
        self.assertFalse((self.paths[2] / '.audio-render.lock').exists())

    def test_preloaded_casting_session_refuses_generation_after_invalidation(self):
        from copy import deepcopy
        from epub_tts_narrator.renderer.casting import CastingSession
        from epub_tts_narrator.renderer.config import DEFAULTS
        (self.paths[0] / 'voice-aliases.json').unlink()
        session = CastingSession(load_bundle(self.paths[0]), deepcopy(DEFAULTS), notify=lambda _: None)
        self.addCleanup(session.close)
        self.assertEqual(self.invoke_refresh(fixtures.DeadClient())[0], 1)
        with self.assertRaisesRegex(ValueError, 'refresh-voice-profiles.*--config'):
            session.generate(self.documents[0]['cast']['voices'][0]['id'])

    def test_concurrent_refresh_is_refused_without_removing_owner_lease(self):
        from epub_tts_narrator.renderer.run_locks import acquire_directory_lock
        lock = acquire_directory_lock(self.paths[0] / '.voice-profile-refresh.lock')
        try:
            client = fixtures.Client()
            self.assertEqual(self.invoke_refresh(client)[0], 1)
            self.assertEqual(client.calls, [])
            self.assertTrue(lock.is_dir())
        finally:
            lock.rmdir()

    def test_no_change_marker_clear_failure_resumes_without_calls(self):
        from pathlib import Path
        self.assertEqual(self.invoke_refresh(fixtures.Client())[0], 0)
        unlink = Path.unlink

        def fail_marker(path, *args, **kwargs):
            if path.name == '.voice-profile-refresh.json':
                raise OSError('marker clear failed')
            return unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', fail_marker):
            self.assertEqual(self.invoke_refresh(fixtures.Client())[0], 1)
        self.assertEqual(self.invoke_refresh(fixtures.DeadClient())[0], 0)

    def test_parallel_proposals_all_survive_publication_failure(self):
        with patch('epub_tts_narrator.analysis_revision._install', side_effect=OSError('disk failure')):
            self.assertEqual(self.invoke_refresh(fixtures.Client(), argv_extra=('--workers', '5'))[0], 1)
        self.assertEqual({row['status'] for row in self.state()['voices'].values()}, {'prepared'})
        for path in (self.root / 'cache').glob('*.json'):
            path.unlink()
        self.assertEqual(self.invoke_refresh(fixtures.DeadClient(), argv_extra=('--workers', '5'))[0], 0)

    def test_external_cast_modification_fails_closed(self):
        from epub_tts_narrator.analysis_revision import revise_directories
        self.assertEqual(self.invoke_refresh(fixtures.DeadClient())[0], 1)
        revised = fixtures.VoiceProfileRefreshTests.revised(self)
        revise_directories({self.paths[0]: revised}, scope='voice-profiles', progress=lambda _: None)
        client = fixtures.Client()
        code, _, errors = self.invoke_refresh(client)
        self.assertEqual(code, 1)
        self.assertIn('Cast changed', errors)
        self.assertEqual(client.calls, [])

    updates = fixtures.VoiceProfileRefreshTests.updates
