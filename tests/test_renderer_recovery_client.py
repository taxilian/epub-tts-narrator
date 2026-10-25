"""Exercise bounded recovery-model retries through the real provider parser."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier, Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator.provider import (
    OpenAICompatibleClient, OutputTruncatedError, ProviderError,
)
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.recovery_client import RecoveryModelClient
from legacy_recovery_fixtures import LegacySpeechResolver as SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import VerificationUnavailable
import test_renderer_recurring_resolution as recurring_fixture


def settings(**overrides):
    return {
        "model": "fixture-reasoning-model",
        "base_url": "https://fixture.invalid/v1",
        "api_key": "fixture-secret-never-in-identity",
        "max_output_tokens": 65536,
        "reasoning_effort": "high",
        **overrides,
    }


def response(value, *, reason="stop", content=None):
    return json.dumps({"choices": [{
        "finish_reason": reason,
        "message": {"role": "assistant", "content": content if content is not None else json.dumps(value)},
    }]}).encode()


class RecoveryModelClientTests(unittest.TestCase):
    def test_planner_retries_same_decision_with_larger_budgets(self):
        client = RecoveryModelClient(settings(token_limit_field="max_tokens"))
        payload = {"expected_text": "B-but Mother", "failures": [{"transcript": "But Mother"}]}
        original = deepcopy(payload)
        bodies = []
        plan = {"action": "retry", "edits": [], "reason": "Try a clearer delivery."}

        def request(_client, encoded, _headers):
            bodies.append(json.loads(encoded))
            return response(plan, reason="length" if len(bodies) < 3 else "stop")

        before = client.identity
        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request):
            self.assertEqual(client.complete("Choose one recovery decision.", payload), plan)
        self.assertEqual([body.pop("max_tokens") for body in bodies], [8192, 32768, 65536])
        self.assertEqual(bodies, [bodies[0]] * 3)
        self.assertEqual(json.loads(bodies[0]["messages"][1]["content"]), original)
        self.assertEqual(payload, original)
        self.assertEqual(client.identity, before)
        self.assertEqual(before["max_output_tokens"], 65536)
        self.assertEqual(before["output_budget_retry"]["budgets"], [8192, 32768, 65536])
        self.assertNotIn("fixture-secret", json.dumps(before))

    def test_reviewer_retries_only_its_own_identical_vote(self):
        client = RecoveryModelClient(settings())
        payload = {"pairs": [{"id": "pair-1", "expected": "Derrins", "heard": "Darin's"}]}
        vote = {"pairs": [{"id": "pair-1", "plausible": True, "reason": "Matching pronunciation."}]}
        bodies = []

        def request(_client, encoded, _headers):
            bodies.append(json.loads(encoded))
            return response(vote, reason="length" if len(bodies) == 1 else "stop")

        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request):
            self.assertEqual(client.complete("Review these sounds.", payload), vote)
        self.assertEqual([body.pop("max_completion_tokens") for body in bodies], [8192, 32768])
        self.assertEqual(bodies[0], bodies[1])

    def test_exhaustion_is_bounded_and_does_not_accept_valid_but_truncated_json(self):
        client = RecoveryModelClient(settings())
        with patch.object(OpenAICompatibleClient, "_request", autospec=True,
                          return_value=response({"action": "accept", "edits": []}, reason="length")) as request:
            with self.assertRaisesRegex(OutputTruncatedError, "after 3 request.*65536 tokens"):
                client.complete("Choose.", {})
        self.assertEqual(request.call_count, 3)

    def test_partial_json_is_discarded_instead_of_completed_or_salvaged(self):
        client = RecoveryModelClient(settings())
        final = {"action": "stop", "edits": [], "reason": "Missing source words."}
        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=[
            response(None, reason="length", content='{"action":"accept","edits":['),
            response(final),
        ]) as request:
            self.assertEqual(client.complete("Choose.", {"expected_text": "Do not go."}), final)
        self.assertEqual(request.call_count, 2)

    def test_smaller_user_limits_are_never_exceeded_or_repeated(self):
        for ceiling, expected in ((2048, [2048]), (8192, [8192]),
                                  (16000, [8192, 16000]), (32768, [8192, 32768]),
                                  (None, [8192, 32768])):
            with self.subTest(ceiling=ceiling):
                client = RecoveryModelClient(settings(max_output_tokens=ceiling))
                with patch.object(OpenAICompatibleClient, "_request", autospec=True,
                                  return_value=response({}, reason="length")) as request:
                    with self.assertRaises(OutputTruncatedError):
                        client.complete("Choose.", {})
                self.assertEqual([json.loads(call.args[1])["max_completion_tokens"]
                                  for call in request.call_args_list], expected)

    def test_nontruncation_errors_are_not_retried_as_output_budget_failures(self):
        cases = (
            ProviderError("Authentication failed", status_code=401),
            ProviderError("The API request failed or timed out.", retryable=True),
            response(None, content='{"action":'),
            response({}, reason="content_filter"),
            response({}, reason="tool_calls"),
        )
        for failure in cases:
            with self.subTest(failure=type(failure).__name__ if isinstance(failure, Exception) else failure):
                kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with patch.object(OpenAICompatibleClient, "_request", autospec=True, **kwargs) as request:
                    with self.assertRaises(ProviderError) as caught:
                        RecoveryModelClient(settings()).complete("Choose.", {})
                self.assertNotIsInstance(caught.exception, OutputTruncatedError)
                self.assertEqual(request.call_count, 1)

    def test_invalid_output_ceiling_is_rejected_before_any_request(self):
        for ceiling in (0, -1, True, 1.5, "65536"):
            with self.subTest(ceiling=ceiling):
                with patch.object(OpenAICompatibleClient, "_request") as request:
                    with self.assertRaises(ProviderError):
                        RecoveryModelClient(settings(max_output_tokens=ceiling))
                request.assert_not_called()

    def test_concurrent_calls_have_independent_retry_budgets_and_stable_identity(self):
        client = RecoveryModelClient(settings())
        identity = client.identity
        rendezvous = Barrier(2)
        lock = Lock()
        bodies = {"planner": [], "reviewer": []}

        def request(_client, encoded, _headers):
            body = json.loads(encoded)
            label = json.loads(body["messages"][1]["content"])["decision"]
            with lock:
                bodies[label].append(body)
                number = len(bodies[label])
            self.assertEqual(client.identity, identity)
            if number == 1:
                rendezvous.wait(timeout=10)
                return response({}, reason="length")
            return response({"decision": label})

        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(client.complete, "Decide.", {"decision": label})
                           for label in bodies]
                self.assertEqual([future.result(timeout=15) for future in futures],
                                 [{"decision": "planner"}, {"decision": "reviewer"}])
        for calls in bodies.values():
            self.assertEqual([body.pop("max_completion_tokens") for body in calls], [8192, 32768])
            self.assertEqual(calls[0], calls[1])
        self.assertEqual(client.identity, identity)
        identity["output_budget_retry"]["budgets"].append(999999)
        self.assertEqual(client.identity["output_budget_retry"]["budgets"], [8192, 32768, 65536])


@unittest.skipUnless(recurring_fixture.FFMPEG and recurring_fixture.FFPROBE,
                     "FFmpeg and ffprobe required for recovery retry/cache integration")
class RecoveryModelRetryIntegrationTests(unittest.TestCase):
    def setUp(self):
        recurring_fixture.RecurringResolutionIntegrationTests.setUp(self)
        self.client = RecoveryModelClient(settings())
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))

    render_chunk = recurring_fixture.RecurringResolutionIntegrationTests.render_chunk

    def test_planner_and_reviewer_truncation_retry_without_more_synthesis_and_cache_reuses(self):
        planner, reviews = [], []

        def request(_client, encoded, _headers):
            body = json.loads(encoded)
            payload = json.loads(body["messages"][1]["content"])
            if "pairs" not in payload:
                planner.append(body)
                return response({"action": "accept", "edits": [], "reason": "Plausible sound spelling."},
                                reason="length" if len(planner) < 3 else "stop")
            reviews.append(body)
            return response({"pairs": [{"id": pair["id"], "plausible": True,
                                        "reason": "Plausible pronunciation spelling."}
                                       for pair in payload["pairs"]]},
                            reason="length" if len(reviews) == 1 else "stop")

        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request) as api:
            result, accepted = self.render_chunk()
            self.assertEqual(len(self.engine.calls), 3)
            self.assertTrue(accepted["report"]["passed"])
            saved = read_json(Path(result["wav_path"]).parent / "accepted.json")
            self.assertTrue(saved["report"]["passed"])
            self.assertTrue(list((self.work / "speech-cache").glob("*.json")))
            counts = (len(self.engine.calls), len(self.base.calls), api.call_count)
            self.assertEqual(self.render_chunk(), (result, accepted))
            self.assertEqual((len(self.engine.calls), len(self.base.calls), api.call_count), counts)
        self.assertEqual([body["max_completion_tokens"] for body in planner], [8192, 32768, 65536])
        self.assertEqual([body["max_completion_tokens"] for body in reviews], [8192, 32768, 8192, 8192])

    def test_exhausted_recovery_budget_preserves_audio_and_resumes_without_synthesis(self):
        with patch.object(OpenAICompatibleClient, "_request", autospec=True,
                          return_value=response({}, reason="length")) as api:
            with self.assertRaises(VerificationUnavailable):
                self.render_chunk()
        self.assertEqual(api.call_count, 3)
        self.assertEqual(len(self.engine.calls), 3)
        candidates = list((self.work / "clips").glob("*/attempt-003.wav"))
        self.assertEqual(len(candidates), 1)
        original = candidates[0].read_bytes()

        def request(_client, encoded, _headers):
            payload = json.loads(json.loads(encoded)["messages"][1]["content"])
            if "pairs" not in payload:
                return response({"action": "accept", "edits": [], "reason": "Plausible sound spelling."})
            return response({"pairs": [{"id": pair["id"], "plausible": True,
                                        "reason": "Plausible pronunciation spelling."}
                                       for pair in payload["pairs"]]})

        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request):
            result, accepted = self.render_chunk()
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(Path(result["wav_path"]), candidates[0])
        self.assertEqual(candidates[0].read_bytes(), original)
        self.assertTrue(accepted["report"]["passed"])


if __name__ == "__main__":
    unittest.main()
