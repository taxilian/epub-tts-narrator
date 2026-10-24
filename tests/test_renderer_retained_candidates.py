from pathlib import Path
import unittest

from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.recurring_resolution import RecurringResolution
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import RenderBlocked, SpeechVerificationFailed, verified_take
import test_renderer_recurring_resolution as fixtures


SOURCE = "Everyone heard a short ack before the door closed."
HEARD = SOURCE.replace("ack", "ugh")
MISSING = SOURCE.replace("ack ", "")


class RejectingClient(fixtures.PlanClient):
    def complete(self, prompt, payload):
        response = super().complete(prompt, payload)
        if "pairs" in payload:
            for pair in response["pairs"]:
                pair["plausible"] = False
        return response


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe required")
class RetainedCandidateTests(unittest.TestCase):
    def setUp(self):
        fixtures.RecurringResolutionIntegrationTests.setUp(self)
        self.text = SOURCE
        self.request["text"] = SOURCE
        self.engine.heard = lambda text, number: HEARD if number == 1 else MISSING
        self.config["render"]["max_attempts"] = 2
        def generate(path, seed):
            return self.engine.synthesize(SOURCE, self.voice, path, seed=seed)
        with self.assertRaises(SpeechVerificationFailed):
            verified_take(SOURCE, generate, self.work / "takes", self.request, self.base,
                          self.config, notify=self.messages.append)
        self.assertEqual(len(self.engine.calls), 2)
        self.config["verification"]["recurrence_threshold"] = 1
        stopped = fixtures.PlanClient([{"action": "stop", "edits": [], "reason": "No useful new take remains."}])
        session = RecurringResolution(self.work / "takes", self.request, SOURCE, SpeechResolver(stopped),
                                      self.config["verification"], can_rewrite=True, can_split=False)
        result = session.consider(session.records[2], notify=self.messages.append)
        self.assertEqual(result["action"], "stop")

    def resume(self):
        return fixtures.RecurringResolutionIntegrationTests.take(self)

    def test_complete_older_candidate_gets_fresh_asr_and_jury_without_new_generation(self):
        first = self.work / "takes" / "attempt-001.json"
        saved = first.read_bytes()
        before = len(self.base.calls)

        wav, accepted = self.resume()

        self.assertEqual(wav.name, "attempt-001.wav")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["transcript"], HEARD)
        self.assertEqual(accepted["report"]["text_confirmation_method"], "recurring_resolution")
        self.assertGreater(len(self.base.calls), before)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(len(self.client.reviews), 3)
        self.assertEqual(first.read_bytes(), saved)
        self.assertEqual(self.resume(), (wav, accepted))
        self.assertEqual(len(self.engine.calls), 2)

    def test_saved_transcript_cannot_supply_a_word_missing_from_fresh_recognition(self):
        import hashlib
        from test_renderer_speech_recovery_integration import pcm
        audio = self.work / "takes" / "attempt-001.wav"
        self.registry[hashlib.sha256(pcm(audio)).hexdigest()] = MISSING
        self.client.actions = [{"action": "stop", "edits": [], "reason": "No complete candidate remains."}]
        self.config["verification"]["recurrence_threshold"] = 1
        with self.assertRaises(SpeechVerificationFailed):
            self.resume()
        self.assertFalse((self.work / "takes" / "accepted.json").exists())
        self.assertEqual(self.client.reviews, [])

    def test_modified_retained_waveform_is_not_recognized_or_reviewed(self):
        audio = self.work / "takes" / "attempt-001.wav"
        audio.write_bytes(b"changed waveform")
        before = len(self.base.calls)
        with self.assertRaises(RenderBlocked):
            self.resume()
        self.assertEqual(len(self.base.calls), before)
        self.assertEqual(self.client.plans, [])

    def test_negative_sound_votes_never_publish_retained_candidate(self):
        client = RejectingClient()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(client))
        with self.assertRaises(SpeechVerificationFailed):
            self.resume()
        self.assertFalse((self.work / "takes" / "accepted.json").exists())
        self.assertEqual(len(client.reviews), 3)
        reviews = list((self.work / "takes").glob("retained-review-*.json"))
        self.assertTrue(reviews)
        self.assertEqual(read_json(reviews[0])["audit"]["error_code"], "sound_review_rejected")


if __name__ == "__main__":
    unittest.main()
