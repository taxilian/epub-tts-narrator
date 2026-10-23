"""Recurring recovery configuration and backend composition without inference."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.chapter_verification import create_chapter_verifier
from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from epub_tts_narrator.renderer.verification import (
    RecurringSpeechVerifier, create_speech_resolver, create_verifier,
)


class Resolver:
    def __init__(self, client, *, secrets=()):
        self.client = client
        self.secrets = secrets
        self.identity = {"backend": "fixture-resolver", "provider": client.identity}
        self.close = Mock()


def primary():
    return SimpleNamespace(
        identity={"backend": "fixture-asr"},
        transcribe=Mock(return_value={"text": "hello"}),
        transcribe_reference=Mock(return_value={"text": "reference"}),
        corroborate_reference=Mock(return_value={"text": "secondary"}),
        judge_reference=Mock(return_value={"decision": "equivalent"}),
        review_phonetics=Mock(return_value={"decision": "accepted"}),
        close=Mock(),
    )


class RecurringResolverFactoryTests(unittest.TestCase):
    settings = {"model": "/unused/whisper", "language": "en"}
    provider = {"model": "analysis-model", "base_url": "https://offline.invalid/v1",
                "api_key": "private-analysis-fixture"}

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.resolver_constructor = Mock(side_effect=Resolver)
        self.stack.enter_context(patch.dict("sys.modules", {
            "epub_tts_narrator.renderer.speech_resolver":
                SimpleNamespace(SpeechResolver=self.resolver_constructor),
        }))

    def test_factory_defaults_on_without_creating_network_requests(self):
        provider = {**self.provider, "max_output_tokens": 32000, "max_retries": 8}
        before = deepcopy(provider)
        with patch("epub_tts_narrator.provider.OpenAICompatibleClient") as client:
            client.return_value.identity = {"model": "fixture-model"}
            resolver = create_speech_resolver(self.settings, provider)
        client.assert_called_once_with(**{**provider, "max_retries": 2})
        self.assertEqual(resolver.client.identity["output_budget_retry"]["budgets"], [8192, 32000])
        self.assertEqual(resolver.secrets, (provider["api_key"],))
        client.return_value.complete.assert_not_called()
        self.assertEqual(provider, before)

    def test_smaller_provider_limits_are_preserved(self):
        with patch("epub_tts_narrator.provider.OpenAICompatibleClient") as client:
            client.return_value.identity = {"model": "fixture-model"}
            create_speech_resolver(self.settings, {**self.provider, "max_output_tokens": 1000,
                                                    "max_retries": 1})
        self.assertEqual(client.call_args.kwargs["max_output_tokens"], 1000)
        self.assertEqual(client.call_args.kwargs["max_retries"], 1)

    def test_off_missing_provider_and_missing_model_never_create_client(self):
        cases = [({"recurring_resolution": "off"}, self.provider), ({}, None),
                 ({}, {}), ({}, {"base_url": self.provider["base_url"]})]
        with patch("epub_tts_narrator.provider.OpenAICompatibleClient") as client:
            for settings, provider in cases:
                with self.subTest(settings=settings, provider=provider is not None):
                    self.assertIsNone(create_speech_resolver(settings, provider))
        client.assert_not_called()
        self.resolver_constructor.assert_not_called()

    def test_mlx_attaches_even_when_jev_is_disabled_and_forwards_verification(self):
        base = primary()
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier", return_value=base):
            wrapped = create_verifier(self.settings, {"enabled": False}, self.provider)
        self.assertIsInstance(wrapped, RecurringSpeechVerifier)
        self.assertIs(wrapped.verifier, base)
        self.assertEqual(wrapped.transcribe("audio.wav"), {"text": "hello"})
        self.assertEqual(wrapped.transcribe_reference("audio.wav"), {"text": "reference"})
        self.assertEqual(wrapped.corroborate_reference("audio.wav"), {"text": "secondary"})
        self.assertEqual(wrapped.judge_reference("a", "b"), {"decision": "equivalent"})
        self.assertEqual(wrapped.review_phonetics("a", "b"), {"decision": "accepted"})
        self.assertEqual(wrapped.speech_resolution_identity, wrapped.speech_resolver.identity)
        self.assertEqual(wrapped.identity["speech_resolution"], wrapped.speech_resolution_identity)
        wrapped.close()
        base.close.assert_called_once_with()
        wrapped.speech_resolver.close.assert_called_once_with()

    def test_cuda_backend_attaches_once_from_both_factories(self):
        settings = {**self.settings, "type": "faster_whisper", "device": "cuda",
                    "compute_type": "float16", "cpu_threads": 2}
        for factory in (create_verifier, create_chapter_verifier):
            with self.subTest(factory=factory.__name__):
                self.resolver_constructor.reset_mock()
                base = primary()
                with patch("epub_tts_narrator.renderer.chapter_verification.FasterWhisperVerifier",
                           return_value=base) as constructor:
                    wrapped = factory(settings, {"enabled": False}, self.provider)
                constructor.assert_called_once_with("/unused/whisper", language="en", device="cuda",
                                                    compute_type="float16", cpu_threads=2)
                self.assertIs(wrapped.verifier, base)
                self.resolver_constructor.assert_called_once()

    def test_no_provider_and_explicit_off_preserve_unwrapped_legacy_backends(self):
        for factory, kind, target in (
            (create_verifier, "mlx_whisper", "verification.MlxWhisperVerifier"),
            (create_chapter_verifier, "faster_whisper", "chapter_verification.FasterWhisperVerifier"),
        ):
            for provider, extra in ((None, {}), (self.provider, {"recurring_resolution": "off"})):
                with self.subTest(factory=factory.__name__, provider=provider is not None):
                    base = primary()
                    with patch("epub_tts_narrator.renderer." + target, return_value=base):
                        result = factory({**self.settings, "type": kind, **extra}, None, provider)
                    self.assertIs(result, base)
        self.resolver_constructor.assert_not_called()

    def test_identity_excludes_credentials_but_tracks_model(self):
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier", side_effect=lambda *a, **k: primary()):
            initial = create_verifier(self.settings, None, self.provider)
            rotated = create_verifier(self.settings, None, {**self.provider, "api_key": "rotated-private"})
            changed = create_verifier(self.settings, None, {**self.provider, "model": "different-model"})
        self.assertEqual(initial.identity, rotated.identity)
        self.assertNotEqual(initial.identity, changed.identity)
        self.assertNotIn("private", json.dumps(initial.identity))
        self.assertNotIn("api_key", json.dumps(initial.identity))

    def test_resolver_is_closed_even_when_base_close_fails(self):
        base, resolver = primary(), Resolver(SimpleNamespace(identity={}))
        base.close.side_effect = RuntimeError("close failed")
        wrapped = RecurringSpeechVerifier(base, resolver)
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            wrapped.close()
        resolver.close.assert_called_once_with()


class RecurringResolverConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renderer.toml"
            path.write_text('[engine]\ncheckpoint = "/unused/drama-3"\n' + text, encoding="utf-8")
            return load_config(path, require_verifier=False)

    def test_defaults_and_explicit_file_values(self):
        expected = {"recurring_resolution": "analysis", "recurrence_threshold": 3,
                    "max_resolution_attempts": 12, "max_resolution_rounds": 2}
        for key, value in expected.items():
            self.assertEqual(DEFAULTS["verification"][key], value)
            self.assertEqual(self.load("")["verification"][key], value)
        actual = self.load('[verification]\nrecurring_resolution = "off"\nrecurrence_threshold = 4\n'
                           'max_resolution_attempts = 20\nmax_resolution_rounds = 3\n')["verification"]
        for key, value in {"recurring_resolution": "off", "recurrence_threshold": 4,
                           "max_resolution_attempts": 20, "max_resolution_rounds": 3}.items():
            self.assertEqual(actual[key], value)

    def test_invalid_modes_types_and_limits_fail_configuration(self):
        cases = {
            "recurring_resolution": ('true', '[]', '{}', '"unknown"'),
            "recurrence_threshold": ('true', '1', '11', '3.0', '"3"'),
            "max_resolution_attempts": ('true', '2', '31', '12.0', '"12"'),
            "max_resolution_rounds": ('true', '0', '6', '2.0', '"2"'),
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, "verification." + key):
                        self.load(f"[verification]\n{key} = {value}\n")

    def test_attempt_budget_cannot_prevent_recurrence_threshold(self):
        with self.assertRaisesRegex(ValueError, "at least recurrence_threshold"):
            self.load("[verification]\nrecurrence_threshold = 10\nmax_resolution_attempts = 9\n")


if __name__ == "__main__":
    unittest.main()
