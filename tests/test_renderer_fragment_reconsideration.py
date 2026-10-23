from copy import deepcopy
import unittest
import wave

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.recurring_resolution import RecurringResolution, confirm_rewritten_report
from epub_tts_narrator.renderer.speech_coverage import speech_coverage, stretched_word_equivalent
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver, validate_resolution
from epub_tts_narrator.renderer.verification_prompts import PROMPT, SOUND_PROMPT
import epub_tts_narrator.renderer.speech_resolver as policies
import test_renderer_recurring_resolution as fixtures


SOURCE = "Carry the ssssssh—the ssssss—carry it home!"
TARGET = "Carry the sh—the s—carry it home!"


def rewrite():
    return {"action": "rewrite", "edits": [
        {"start": start, "end": start + len(word), "source": word, "replacement": replacement}
        for word, replacement, start in (("ssssssh", "sh", SOURCE.index("ssssssh")),
                                         ("ssssss", "s", SOURCE.rindex("ssssss")))],
        "reason": "Shorten written consonant runs without losing a word or distinct letter."}


def feedback():
    return [{"attempt": 17, "proposal": rewrite(), "decision": "blocked",
             "error_code": "invalid_proposal", "validation_error": "The previous guard rejected a short target.",
             "sound_reviews": []}]


def failures():
    return [{"attempt": 18, "audio_sha256": "a" * 64, "transcript": "", "tts_text": SOURCE,
             "reasons": ["excessive_silence"]}]


class SoundClient:
    identity = {"model": "offline-fragment-reconsideration", "base_url": "https://fixture.invalid"}

    def __init__(self, *, approve=True, planner=False):
        self.approve, self.planner = approve, planner
        self.calls = []
        self.unavailable = False
        self.invalid_once = False

    def complete(self, prompt, payload):
        self.calls.append((prompt, deepcopy(payload)))
        if prompt == PROMPT and self.planner:
            return rewrite()
        if prompt != SOUND_PROMPT:
            raise AssertionError("A previously proposed candidate must not call the planner again")
        if self.unavailable:
            raise TimeoutError("Fixture sound reviewer unavailable")
        if self.invalid_once:
            self.invalid_once = False
            return {"pairs": []}
        return {"pairs": [{"id": pair["id"], "plausible": self.approve,
                            "reason": "The complete consonant fragment survives."} for pair in payload["pairs"]]}


class FragmentReconsiderationTests(unittest.TestCase):
    def test_two_span_fragment_rewrite_reaches_all_sound_reviews(self):
        client = SoundClient(planner=True)
        resolver = SpeechResolver(client)
        audit = resolver.resolve(SOURCE, failures(), can_split=False)
        proposal = validate_resolution(audit, SOURCE, failures(), resolver.identity, can_split=False)
        self.assertIsNotNone(proposal, audit.get("validation_error"))
        self.assertEqual(proposal["tts_text"], TARGET)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(audit["reviews"]), 3)

    def test_fragment_eligibility_does_not_relax_audio_acceptance(self):
        self.assertFalse(stretched_word_equivalent("ssssss", "s"))
        self.assertFalse(speech_coverage("ssssss", "s")["passed"])
        for source, target in (("SSSSSS", "s"), ("aaaaaa", "a"), ("yyy", "y"),
                               ("ssshhh", "s"), ("ssssss", "S"), ("ss", "s")):
            with self.subTest(source=source, target=target):
                self.assertFalse(policies._stretched_fragment(source, target))
        self.assertTrue(policies._stretched_fragment("ssssss", "s"))
        self.assertTrue(policies._stretched_fragment("Mmmm", "m"))

    def test_reconsideration_reviews_saved_candidate_without_planner_and_binds_origin(self):
        client = SoundClient()
        resolver = SpeechResolver(client)
        audit = resolver.reconsider(SOURCE, failures(), can_split=False, previous_decisions=feedback())
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(audit["reconsidered"], {"attempt": 17, "error_code": "invalid_proposal"})
        proposal = validate_resolution(audit, SOURCE, failures(), resolver.identity, can_split=False,
                                       previous_decisions=feedback())
        self.assertEqual(proposal["tts_text"], TARGET)
        bad = deepcopy(audit)
        bad["reconsidered"]["attempt"] = 7
        self.assertIsNone(validate_resolution(bad, SOURCE, failures(), resolver.identity, can_split=False))
        bad = deepcopy(audit)
        bad["proposal"]["edits"][1]["replacement"] = "z"
        self.assertIsNone(validate_resolution(bad, SOURCE, failures(), resolver.identity, can_split=False))
        original = {"passed": False, "expected_text": SOURCE, "transcript": TARGET,
                    "reasons": ["excessive_silence"]}
        marker = {"original_text": SOURCE, "tts_text": TARGET, "audit": audit, "failures": failures()}
        self.assertEqual(confirm_rewritten_report(original, marker), original)

    def test_failed_or_already_tested_candidate_is_not_reconsidered_repeatedly(self):
        client = SoundClient(approve=False)
        resolver = SpeechResolver(client)
        audit = resolver.reconsider(SOURCE, failures(), can_split=False, previous_decisions=feedback())
        self.assertEqual(audit["error_code"], "sound_review_rejected")
        previous = feedback() + [{"attempt": 18, "proposal": audit["proposal"], "decision": "blocked",
                                  "error_code": "sound_review_rejected", "validation_error": None,
                                  "sound_reviews": [{"reviewer": row["reviewer"], "response": row["response"]}
                                                    for row in audit["reviews"]]}]
        self.assertIsNone(resolver.reconsider(SOURCE, failures(), can_split=False, previous_decisions=previous))
        self.assertEqual(len(client.calls), 3)
        self.assertIsNone(resolver.reconsider(SOURCE, [{**failures()[0], "tts_text": TARGET}],
                                             can_split=False, previous_decisions=feedback()))
        self.assertIsNone(resolver.reconsider(SOURCE, failures(), can_rewrite=False,
                                             can_split=False, previous_decisions=feedback()))
        self.assertEqual(len(client.calls), 3)

    def test_newest_eligible_candidate_is_bound_and_tested_candidate_is_skipped(self):
        previous = feedback()
        older = deepcopy(previous[0])
        older["attempt"] = 7
        older["proposal"]["edits"][0]["replacement"] = "ssh"
        previous.insert(0, older)
        resolver = SpeechResolver(SoundClient())
        audit = resolver.reconsider(SOURCE, failures(), can_split=False, previous_decisions=previous)
        self.assertEqual(audit["reconsidered"]["attempt"], 17)
        bad = deepcopy(audit)
        bad["proposal"] = older["proposal"]
        bad["reconsidered"]["attempt"] = 7
        self.assertIsNone(validate_resolution(bad, SOURCE, failures(), resolver.identity, can_split=False))
        fallback = resolver.reconsider(SOURCE, [{**failures()[0], "tts_text": TARGET}],
                                       can_split=False, previous_decisions=previous)
        self.assertEqual(fallback["reconsidered"]["attempt"], 7)


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe required")
class FragmentResumeTests(unittest.TestCase):
    def setUp(self):
        fixtures.RecurringResolutionIntegrationTests.setUp(self)
        self.text = SOURCE
        self.request = {"kind": "speech", "text": SOURCE, "engine": self.engine.identity}
        self.engine.heard = lambda text, number: text
        self.client = SoundClient()
        self.verifier = fixtures.RecurringSpeechVerifier(self.base, SpeechResolver(self.client))

    def saved_stop(self):
        directory = self.work / "takes"
        directory.mkdir()
        wav = directory / "attempt-018.wav"
        with wave.open(str(wav), "wb") as audio:
            audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(88200))
        record = {"attempt": 18, "request": self.request, "request_hash": digest(self.request),
                  "audio_file": wav.name, "audio_sha256": file_hash(wav),
                  "report": {"passed": False, "expected_text": SOURCE, "transcript": "",
                             "reasons": ["silent_audio", "excessive_silence"]}}
        write_json(directory / "attempt-018.json", record)
        resolver = self.verifier.speech_resolver
        session = RecurringResolution(directory, self.request, SOURCE, resolver, {}, can_rewrite=True, can_split=False)
        evidence = [session._failure(record)]
        blocked = {"identity": resolver.identity, "prompt": PROMPT,
                   "payload": policies._payload(SOURCE, evidence, True, False, acoustic_rewrite=True),
                   "proposal": rewrite(), "reviews": [], "decision": "blocked", "error_code": "invalid_proposal",
                   "validation_error": "The old guard rejected a single-letter target."}
        row = {"attempt": 17, "audio_sha256": file_hash(wav), "failures": evidence, "audit": blocked}
        session.state["history"] = [row]
        previous = session._feedback([row])
        stop = {"identity": resolver.identity, "prompt": PROMPT,
                "payload": policies._payload(SOURCE, evidence, True, False,
                                              previous_decisions=previous, acoustic_rewrite=True),
                "proposal": {"action": "stop", "edits": [], "reason": "No previously permitted repair helped."},
                "reviews": [], "decision": "resolved"}
        session.state["history"].append({"attempt": 18, "audio_sha256": file_hash(wav),
                                          "failures": evidence, "audit": stop, "previous_decisions": previous})
        session.state["handled"] = {"takes": 1}
        session._save()
        return directory

    def test_resume_reviews_newly_valid_candidate_before_buying_more_failed_takes(self):
        directory = self.saved_stop()
        original = file_hash(directory / "attempt-018.wav")
        wav, accepted = fixtures.RecurringResolutionIntegrationTests.take(self)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.engine.calls[0]["text"], TARGET)
        self.assertEqual(wav.name, "attempt-019.wav")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["expected_text"], SOURCE)
        self.assertEqual(len(self.client.calls), 3)
        self.assertEqual(file_hash(directory / "attempt-018.wav"), original)
        replay, _ = fixtures.RecurringResolutionIntegrationTests.take(self)
        self.assertEqual(replay, wav)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(self.client.calls), 3)

    def test_review_outage_resumes_same_pending_candidate_without_more_audio_or_history_growth(self):
        directory = self.saved_stop()
        path = directory / "speech-resolution.json"
        original = read_json(path)["history"]
        self.client.unavailable = True
        for _ in range(2):
            with self.assertRaises(fixtures.VerificationUnavailable):
                fixtures.RecurringResolutionIntegrationTests.take(self)
            saved = read_json(path)
            self.assertEqual(len(saved["history"]), len(original) + 1)
            self.assertEqual(saved["history"][:len(original)], original)
            self.assertEqual(self.engine.calls, [])
        self.client.unavailable = False
        _, accepted = fixtures.RecurringResolutionIntegrationTests.take(self)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(read_json(path)["history"]), len(original) + 1)

    def test_malformed_sound_review_is_pending_not_a_permanent_rejection(self):
        directory = self.saved_stop()
        self.client.invalid_once = True
        with self.assertRaises(fixtures.VerificationUnavailable):
            fixtures.RecurringResolutionIntegrationTests.take(self)
        self.assertEqual(self.engine.calls, [])
        saved = read_json(directory / "speech-resolution.json")
        self.assertEqual(saved["history"][-1]["audit"]["error_code"], "invalid_review")
        _, accepted = fixtures.RecurringResolutionIntegrationTests.take(self)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.engine.calls), 1)


if __name__ == "__main__":
    unittest.main()
