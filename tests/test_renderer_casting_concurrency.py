"""Independent Gemini reference jobs overlap during bounded bulk casting."""

import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
from types import SimpleNamespace
from threading import Event, Lock, get_ident
import unittest

from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.gemini_voice_design import GeminiVoiceDesigner, GeminiVoiceError
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
import test_renderer_gemini_fresh_regeneration as fixtures
import test_renderer_voice_design_integration as voice_fixtures


WAIT_SECONDS = 5


class GeminiOverlapTransport:
    """Offline Gemini service with deterministic admission gates."""

    def __init__(self, fixture, *, blocked=(1,), release_on_overlap=True):
        self.fixture = fixture
        self.blocked = set(blocked)
        self.release_on_overlap = release_on_overlap
        self.first_interaction = Event()
        self.overlap = Event()
        self.release = Event()
        self.first_cleanup = Event()
        self.lock = Lock()
        self.calls = []
        self.designers = []
        self.created_voice_ids = []
        self.interaction_voice_ids = []
        self.deleted_voice_ids = []
        self.hosted_voice_ids = set()
        self.active_interactions = 0
        self.maximum_active_interactions = 0
        self.errors = {}
        self.before_create = None

    def designer(self, _settings):
        designer = GeminiVoiceDesigner("fixture-credential-never-sent")
        designer._post = self.post
        designer._opener = SimpleNamespace(open=self.cleanup_request)
        self.designers.append(designer)
        return designer

    def post(self, endpoint, payload):
        if endpoint == "/voices":
            with self.lock:
                self.calls.append((endpoint, deepcopy(payload)))
                identifier = f"voice_overlap_{len(self.created_voice_ids) + 1}"
                self.created_voice_ids.append(identifier)
                self.hosted_voice_ids.add(identifier)
                creation = len(self.created_voice_ids)
            if self.before_create is not None:
                self.before_create(creation)
            return {"id": identifier}

        self.fixture.assertEqual(endpoint, "/interactions")
        identifier = payload["generation_config"]["speech_config"][0]["voice"]
        with self.lock:
            self.calls.append((endpoint, deepcopy(payload)))
            self.fixture.assertIn(identifier, self.hosted_voice_ids)
            self.interaction_voice_ids.append(identifier)
            interaction = len(self.interaction_voice_ids)
            self.active_interactions += 1
            self.maximum_active_interactions = max(
                self.maximum_active_interactions, self.active_interactions)
            if interaction == 1:
                self.first_interaction.set()
            if self.active_interactions == 2:
                self.overlap.set()
                if self.release_on_overlap:
                    self.release.set()
        try:
            if interaction in self.blocked and not self.release.wait(WAIT_SECONDS):
                raise AssertionError("A gated Gemini interaction was never released")
            if interaction in self.errors:
                raise self.errors[interaction]
            text = payload["input"][0]["content"][0]["text"]
            audio = self.fixture.root / f"remote-overlap-{interaction}.wav"
            self.fixture.audio.write_audio(text, audio, payload["generation_config"]["seed"])
            return {"status": "completed", "steps": [{"type": "model_output", "content": [{
                "type": "audio", "mime_type": "audio/wav",
                "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
            }]}]}
        finally:
            with self.lock:
                self.active_interactions -= 1

    def cleanup_request(self, request, **kwargs):
        if request.get_method() == "GET":
            return io.BytesIO(json.dumps({"voices": []}).encode())
        self.fixture.assertEqual(request.get_method(), "DELETE")
        identifier = request.full_url.rsplit("/", 1)[-1]
        with self.lock:
            self.deleted_voice_ids.append(identifier)
            self.hosted_voice_ids.discard(identifier)
            if len(self.deleted_voice_ids) == 1:
                self.first_cleanup.set()
        return io.BytesIO(b"")


class AffineVerifier(voice_fixtures.TranscriptVerifier):
    def __init__(self, transcripts):
        super().__init__(transcripts)
        self.thread_ids = {get_ident()}
        self.close_thread = None
        self.closed = Event()
        self.active = 0
        self.maximum_active = 0
        self.attempts = 0
        self.guard = Lock()

    def transcribe(self, path):
        with self.guard:
            self.thread_ids.add(get_ident())
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.attempts += 1
            failing = self.attempts == 1
        try:
            if failing:
                raise ValueError("fixture verifier fatal")
            return super().transcribe(path)
        finally:
            with self.guard:
                self.active -= 1

    def close(self):
        self.close_thread = get_ident()
        self.thread_ids.add(self.close_thread)
        self.closed.set()


@unittest.skipUnless(fixtures.FFMPEG, "FFmpeg is required for casting integration")
class CastingConcurrencyTests(unittest.TestCase):
    setUp = fixtures.GeminiFreshRegenerationTests.setUp

    def session(self, transport, *, bundle=None, verifier_factory=None):
        session = CastingSession(
            bundle or self.bundle, self.config,
            engine_factory=lambda _: self.engine,
            verifier_factory=verifier_factory or (lambda *_args, **_kwargs: self.verifier),
            designer_factory=transport.designer, notify=self.notices.append,
        )
        self.addCleanup(session.close)
        return session

    def add_voice(self, bundle, voice_id="voice_extra"):
        bundle["cast"]["voices"].append({
            "id": voice_id, "description": "A distinct test voice.", "traits": [],
            "sample_script": {
                "basis": "casting", "text": self.config["voices"]["reference_text"],
                "reason": "Independent concurrency fixture.",
            },
        })
        return voice_id

    def narration_bundle(self):
        self.voice_id = "voice_mara"
        self.alias_id = "voice_mara_narration"
        return voice_fixtures.VoiceDesignIntegrationTests.narration_bundle(self)

    def test_generate_missing_overlaps_independent_gemini_reference_jobs(self):
        remote = GeminiOverlapTransport(self, blocked=(1,), release_on_overlap=False)
        session = self.session(remote)

        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(session.generate_missing)
            try:
                self.assertTrue(remote.first_interaction.wait(WAIT_SECONDS))
                overlapped = remote.overlap.wait(WAIT_SECONDS)
            finally:
                remote.release.set()
            selected = pending.result(2 * WAIT_SECONDS)

        self.assertEqual({row["voice_id"] for row in selected}, {"voice_mara", "voice_lena"})
        self.assertEqual(len({id(designer) for designer in remote.designers}), 2)
        self.assertTrue(overlapped, "Independent Gemini jobs were admitted serially")

    def test_bounded_admission_dedupes_aliases_in_input_order(self):
        bundle = self.narration_bundle()
        extra = self.add_voice(bundle)
        self.config["casting"]["gemini_concurrency"] = 2
        remote = GeminiOverlapTransport(self, blocked=(1, 2), release_on_overlap=False)
        session = self.session(remote, bundle=bundle)
        session.update_prompt(extra, "An extra voice.", "The extra voice speaks clearly.")
        voice_ids = [self.alias_id, "voice_mara", "voice_lena", extra]

        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(session.ensure_ready_many, voice_ids)
            try:
                self.assertTrue(remote.first_interaction.wait(WAIT_SECONDS))
                self.assertTrue(remote.overlap.wait(WAIT_SECONDS))
                self.assertEqual(len(remote.interaction_voice_ids), 2)
                self.assertFalse(pending.done())
            finally:
                remote.release.set()
            results = pending.result(2 * WAIT_SECONDS)

        self.assertEqual([row["canonical_voice_id"] for row in results],
                         ["voice_mara", "voice_lena", extra])
        self.assertEqual(results[0]["voice_id"], self.alias_id)
        self.assertTrue(all(row["selected"] is not None for row in results))
        self.assertEqual(len(remote.interaction_voice_ids), 3)
        self.assertEqual(remote.maximum_active_interactions, 2)

    def test_ordinary_failure_is_returned_while_sibling_finishes(self):
        remote = GeminiOverlapTransport(self)
        remote.errors[1] = GeminiVoiceError("fixture ordinary Gemini failure")
        session = self.session(remote)

        results = session.ensure_ready_many(["voice_mara", "voice_lena"])

        self.assertEqual(sum(isinstance(row["error"], SpeechVerificationFailed) for row in results), 1)
        self.assertEqual(sum(row["selected"] is not None for row in results), 1)

        remote.errors.clear()
        resumed = session.ensure_ready_many(["voice_mara", "voice_lena"])
        self.assertTrue(all(row["selected"] is not None for row in resumed))
        self.assertEqual(len(remote.interaction_voice_ids), 3)

    def test_generate_missing_raises_ordinary_failure_after_sibling_finishes(self):
        remote = GeminiOverlapTransport(self)
        remote.errors[1] = GeminiVoiceError("fixture ordinary Gemini failure")
        session = self.session(remote)

        with self.assertRaises(SpeechVerificationFailed):
            session.generate_missing()

        self.assertEqual(sum(session.store.load(voice_id) is not None
                             for voice_id in ("voice_mara", "voice_lena")), 1)

    def test_regeneration_fatal_drains_started_work_and_preserves_unadmitted_pending(self):
        extra = self.add_voice(self.bundle)
        self.config["casting"]["gemini_concurrency"] = 2
        remote = GeminiOverlapTransport(self, blocked=(1,), release_on_overlap=False)
        verifiers = []

        def verifier_factory(*_args, **_kwargs):
            verifier = AffineVerifier(self.transcripts)
            verifiers.append(verifier)
            return verifier

        session = self.session(remote, verifier_factory=verifier_factory)
        session.update_prompt(extra, "An extra voice.", "The extra voice speaks clearly.")
        voice_ids = ["voice_mara", "voice_lena", extra]

        def check_invalidation(creation):
            if creation == 1:
                for voice_id in voice_ids:
                    self.assertEqual(session.store.regeneration_status(voice_id)["state"], "pending")

        remote.before_create = check_invalidation
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(session.regenerate_many, voice_ids)
            try:
                self.assertTrue(remote.first_interaction.wait(WAIT_SECONDS))
                self.assertTrue(remote.first_cleanup.wait(WAIT_SECONDS))
                self.assertFalse(pending.done())
                self.assertEqual(len(remote.interaction_voice_ids), 2)
                self.assertTrue((session.store.directory / ".lock").exists())
                self.assertFalse(verifiers[0].closed.is_set())
            finally:
                remote.release.set()
            with self.assertRaisesRegex(ValueError, "fixture verifier fatal"):
                pending.result(2 * WAIT_SECONDS)

        self.assertEqual(len(remote.interaction_voice_ids), 2)
        admitted_states = [session.store.regeneration_status(voice_id)
                           for voice_id in ("voice_mara", "voice_lena")]
        self.assertEqual(sum(state is None for state in admitted_states), 1)
        self.assertEqual(sum(state is not None and state["state"] == "failed"
                             for state in admitted_states), 1)
        self.assertEqual(session.store.regeneration_status(extra)["state"], "pending")
        self.assertFalse((session.store.directory / ".lock").exists())
        self.assertFalse(verifiers[0].closed.is_set())
        self.assertEqual(verifiers[0].maximum_active, 1)
        session.close()
        self.assertTrue(verifiers[0].closed.is_set())
        self.assertEqual(len(verifiers[0].thread_ids), 1)
        self.assertIn(verifiers[0].close_thread, verifiers[0].thread_ids)


if __name__ == "__main__":
    unittest.main()
