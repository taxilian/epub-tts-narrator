from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.recurring_resolution import RecurringResolution
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver, validate_resolution
import epub_tts_narrator.renderer.speech_resolver as policies
import epub_tts_narrator.renderer.verification_prompts as prompts
import test_renderer_speech_resolver as fixtures


SOURCE = "Everyone knows this is—ack!"
SPAN = "this is—ack"


def edit(replacement="this is. Ack"):
    start = SOURCE.index(SPAN)
    return {"start": start, "end": start + len(SPAN), "source": SPAN, "replacement": replacement}


class ProsodyRewriteTests(unittest.TestCase):
    def test_word_identical_boundary_repair_requires_three_reviews_and_preserves_source(self):
        client = fixtures.Client(action="rewrite", edits=[edit()])
        resolver = SpeechResolver(client)
        failures = [fixtures.failure("Everyone knows this is.")]
        audit = resolver.resolve(SOURCE, failures, can_split=False)
        proposal = validate_resolution(audit, SOURCE, failures, resolver.identity, can_split=False)
        self.assertIsNotNone(proposal, audit.get("validation_error"))
        self.assertEqual(proposal["tts_text"], "Everyone knows this is. Ack!")
        self.assertEqual(audit["payload"]["requested_text"], SOURCE)
        self.assertEqual(len(audit["reviews"]), 3)

    def test_no_word_changes_markup_or_spelling_noops_can_use_prosodic_exception(self):
        for replacement in ("this is. Yes", "this. Ack", "this is. Ack again",
                            "this is. [Ack]", "this is. <Ack>", "this is.\x00Ack",
                            "this is—ACK"):
            with self.subTest(replacement=replacement):
                client = fixtures.Client(action="rewrite", edits=[edit(replacement)])
                audit = SpeechResolver(client).resolve(SOURCE, [fixtures.failure("Everyone knows this is.")],
                                                       can_split=False)
                self.assertEqual(audit["error_code"], "invalid_proposal")
                self.assertEqual(len(client.calls), 1)

    def test_prosody_cannot_skip_sound_rejection_or_repeat_a_failed_input(self):
        client = fixtures.Client(action="rewrite", edits=[edit()], votes=[[False], [True], [False]])
        resolver = SpeechResolver(client)
        audit = resolver.resolve(SOURCE, [fixtures.failure("Everyone knows this is.")], can_split=False)
        self.assertEqual(audit["error_code"], "sound_review_rejected")
        self.assertEqual(len(audit["reviews"]), 3)
        failed = fixtures.failure("Everyone knows this is.", tts_text="Everyone knows this is. Ack!")
        audit = SpeechResolver(fixtures.Client(action="rewrite", edits=[edit()])).resolve(
            SOURCE, [failed], can_split=False)
        self.assertEqual(audit["error_code"], "invalid_proposal")

    def test_previous_policy_cannot_authorize_the_new_punctuation_repair(self):
        self.assertEqual(digest(prompts._PRE_PROSODY_PROMPT), prompts._PRE_PROSODY_PROMPT_SHA256)
        client = fixtures.Client(action="rewrite", edits=[edit()])
        resolver = SpeechResolver(client)
        failures = [fixtures.failure("Everyone knows this is.")]
        audit = resolver.resolve(SOURCE, failures, can_split=False)
        self.assertIsNotNone(validate_resolution(audit, SOURCE, failures, resolver.identity, can_split=False))
        old = deepcopy(audit)
        old["identity"] = resolver.previous_identity
        old["prompt"] = prompts._PRE_PROSODY_PROMPT
        self.assertIsNone(validate_resolution(old, SOURCE, failures, resolver.previous_identity,
                                              can_split=False))

    def test_policy_upgrade_keeps_prior_plain_delivery_and_reviewed_spelling(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request = {"kind": "speech", "text": SOURCE}
        old_prompt = getattr(prompts, "_PRE_PROSODY_PROMPT", prompts.PROMPT)
        failures = [fixtures.failure("Everyone knows this is.", attempt=3)]
        with patch.object(policies, "PROMPT", old_prompt), patch.object(fixtures, "PROMPT", old_prompt):
            plain_resolver = SpeechResolver(fixtures.Client(action="plain_delivery"))
            session = RecurringResolution(root, request, SOURCE, plain_resolver, {},
                                          can_rewrite=True, can_split=False, can_plain_delivery=True)
            plain = plain_resolver.resolve(SOURCE, failures, can_split=False, can_plain_delivery=True)
            session.state["history"].append({"attempt": 3, "failures": failures, "audit": plain,
                                               "previous_decisions": []})
            spelling = {"start": SOURCE.index("ack"), "end": SOURCE.index("ack") + 3,
                        "source": "ack", "replacement": "Ak"}
            previous = session._feedback(session.state["history"])
            later = [*failures, fixtures.failure("Everyone knows this is.", attempt=4)]
            rewrite = SpeechResolver(fixtures.Client(action="rewrite", edits=[spelling])).resolve(
                SOURCE, later, can_split=False, can_plain_delivery=True, previous_decisions=previous)
            session.state["history"].append({"attempt": 4, "failures": later, "audit": rewrite,
                                               "previous_decisions": previous})
            session._save()
        original = (root / "speech-resolution.json").read_bytes()
        current = RecurringResolution(root, request, SOURCE, SpeechResolver(fixtures.Client()), {},
                                      can_rewrite=True, can_split=False, can_plain_delivery=True)
        self.assertEqual(len(current.state["history"]), 2)
        self.assertEqual(current.active_delivery()["action"], "plain_delivery")
        self.assertEqual(current.active_rewrite()["tts_text"], SOURCE.replace("ack", "Ak"))
        self.assertEqual((root / "speech-resolution.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
