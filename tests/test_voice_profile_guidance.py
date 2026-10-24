"""Acoustic design guidance reaches every path that can propose a new voice.

These tests verify routing and identity protection, not model obedience or the
perceptual suitability of generated audio. Those require separate live evaluation.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.identity import audit_identities
from epub_tts_narrator.pipeline import analyze, check_cast, make_units, seed_cast
from epub_tts_narrator.prompts import VOICE_PROFILE_GUIDANCE, VOICE_PROFILE_RECONCILIATION
from test_contracts import valid_document
from test_identity import Requests, case, fixture, split_patch
from test_pipeline import FakeClient


class CapturedIdentityRequests(Requests):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = []

    def run(self, system, payload, validate, *, cancelled=None):
        with self.lock:
            self.prompts.append((system, deepcopy(payload)))
        return super().run(system, payload, validate, cancelled=cancelled)


class VoiceProfileGuidanceTests(unittest.TestCase):
    def test_discovery_and_every_reconciliation_route_share_acoustic_contract(self):
        # Bounded discovery and whole-book reconciliation construct different
        # request suffixes. Later cumulative rounds must keep the contract too.
        for whole_book in (False, True):
            with self.subTest(whole_book=whole_book), tempfile.TemporaryDirectory() as root:
                client = FakeClient()
                document = analyze(
                    deepcopy(valid_document()["source"]), client, Path(root) / "cache",
                    identity_consensus=False, whole_book_discovery=whole_book,
                    cast_reconciliation_rounds=2,
                )
                modes = [payload.get("mode") for _, payload in client.calls]
                self.assertEqual(modes.count("discover"), 1)
                self.assertEqual(modes.count("reconcile"), 2)
                self.assertEqual(modes.count("voice_ensemble"), 1)
                for system, payload in client.calls:
                    if payload.get("mode") in {"discover", "reconcile", "voice_ensemble"}:
                        self.assertEqual(system.count(VOICE_PROFILE_GUIDANCE), 1)
                        self.assertEqual(system.count(VOICE_PROFILE_RECONCILIATION), 1)
                    else:
                        # Acoustic rewriting is not an attribution task.
                        self.assertNotIn(VOICE_PROFILE_GUIDANCE, system)
                        self.assertEqual(payload["final_cast"], document["cast"])

    def test_new_split_voice_gets_contract_without_recasting_existing_person(self):
        source, cast = fixture(("Mara",))
        original = deepcopy(cast)
        requests = CapturedIdentityRequests([case("split", ("c1",))])
        updated, issues, provenance = audit_identities(
            source, cast, requests, make_units(source), whole_book=True,
        )
        preparation = [(system, payload) for system, payload in requests.prompts
                       if payload["mode"] == "identity_prepare_split"]
        self.assertEqual(len(preparation), 1)
        system, payload = preparation[0]
        self.assertEqual(system.count(VOICE_PROFILE_GUIDANCE), 1)
        self.assertNotIn(VOICE_PROFILE_RECONCILIATION, system)
        self.assertIn("Never change an existing voice record", system)
        self.assertEqual(issues, [])
        self.assertEqual(provenance["cases"][0]["status"], "applied")
        self.assertEqual(cast, original)
        for voice in original["voices"]:
            self.assertEqual(next(row for row in updated["voices"] if row["id"] == voice["id"]), voice)
        new_id = payload["available_ids"]["voices"][0]
        self.assertTrue(any(row["id"] == new_id for row in updated["voices"]))
        self.assertEqual(len([p for _, p in requests.prompts if p["mode"] == "identity_vote"]), 3)
        check_cast(source, updated)

    def test_acoustic_contract_does_not_allow_split_to_replace_existing_voice(self):
        source, cast = fixture(("Mara",))

        def recast_existing(payload):
            response = split_patch(payload)
            response["voices"].append(dict(
                deepcopy(cast["voices"][1]),
                description="A clearer acoustic design that was not authorized by the split.",
            ))
            return response

        requests = CapturedIdentityRequests([case("split", ("c1",))], split=recast_existing)
        updated, issues, provenance = audit_identities(source, cast, requests, make_units(source))
        self.assertEqual(updated, cast)
        self.assertEqual(provenance["cases"][0]["status"], "invalid_proposal")
        self.assertTrue(issues)
        self.assertFalse(any(p["mode"] == "identity_vote" for _, p in requests.prompts))

    def test_seed_is_usable_casting_without_invented_source_traits(self):
        cast = seed_cast()
        check_cast(valid_document()["source"], cast)
        narrator = cast["voices"][0]
        self.assertEqual({trait["name"] for trait in narrator["traits"]},
                         {"resonance", "articulation", "pace"})
        self.assertTrue(all(trait["basis"] == "casting" and not trait["evidence"]
                            for trait in narrator["traits"]))
        self.assertFalse(any(word in narrator["description"].casefold()
                              for word in ("male", "female", "young", "elderly", "accent", "baritone")))
        self.assertEqual(narrator["speechTraits"],
                         "neutral General American accent; steady conversational pace")
        self.assertIsNone(cast["roles"][0]["character_id"])


if __name__ == "__main__":
    unittest.main()
