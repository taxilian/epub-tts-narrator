"""A retry is a real retry: a passage that failed always gets new takes, whatever was saved before.

From a person's side, retrying a failed chapter (``r`` in the run shell, the
failure dialog's retry, a book retry, or running the render again) means
"try that again". Verified takes are reused, a take a person chose is used,
and the failed passage gets a fresh allowance of new takes. A retry must never
end by re-reading the same saved failures and reporting them again: whatever
saved recovery state the failure left (a rejected split, a best-effort choice
that found nothing usable, a repair, a planner's stop), the retry makes at
least one new take of the passage before it can fail again.
"""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import clip_recovery
from epub_tts_narrator.renderer.exhaustion_selection import NoUsableTake
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.take_feedback import record_verdict
from epub_tts_narrator.renderer.take_selection import list_takes
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import RenderBlocked, SpeechVerificationFailed
from test_renderer_clip_repair import BroadClient
from test_renderer_failed_assembly_resume import TEXT, FailedAssemblyResumeTests
from test_renderer_workflow import FFMPEG, FFPROBE


@contextmanager
def refusing_chooser():
    """The best-effort chooser finds nothing it may publish (every take misread a chosen name)."""
    def refuse(*args, **kwargs):
        raise NoUsableTake("Every usable retained take misread a name whose pronunciation you chose")
    with patch.object(clip_recovery, "select_or_filter", side_effect=refuse):
        yield


def every_take_marked_bad(work):
    """A person listened to every saved take of every passage and marked each one bad."""
    for directory in (work / "clips").glob("*"):
        if not (directory / "attempt-001.json").exists():
            continue
        for row in list_takes(work / "clips", directory.name):
            if not row["eligible"]:
                continue
            record_verdict(work, directory.name, row, "bad")


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class RetryMakesNewTakesTests(FailedAssemblyResumeTests):
    # Only the retry scenarios below run here; the inherited tests run in their own module.
    test_saved_rejected_join_can_resume_with_a_new_verified_plain_whole_take = None
    test_failed_join_and_fresh_whole_recovery_share_one_invocation_budget = None
    test_saved_explicit_split_decision_cannot_replay_the_already_failed_strategy = None
    test_unavailable_join_verification_retains_parts_without_fresh_whole_synthesis = None
    test_child_rejection_is_not_mistaken_for_an_exhausted_join = None

    def modern(self):
        """A resolver with the best-effort chooser and clip repair, as configured renders have."""
        self.client = BroadClient()
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))

    def fail_once(self):
        """Render until the passage fails (``RenderBlocked`` or ``SpeechVerificationFailed``)."""
        with self.assertRaises((RenderBlocked, SpeechVerificationFailed)):
            self.render_chunk()
        return len(self.engine.calls)

    def assert_retry_makes_new_takes(self, made):
        try:
            self.render_chunk()
        except (RenderBlocked, SpeechVerificationFailed):
            pass
        self.assertGreater(len(self.engine.calls), made, "The retry made no new take: " + " | ".join(self.messages[-6:]))

    def test_after_a_rejected_split_with_nothing_publishable(self):
        # Whole takes, then a sentence split whose joined take was rejected.
        made = self.fail_once()
        every_take_marked_bad(self.work)
        self.modern()
        self.assert_retry_makes_new_takes(made)

    def test_after_a_rejected_split_without_a_resolver(self):
        made = self.fail_once()
        self.assert_retry_makes_new_takes(made)

    def test_after_a_rejected_split_with_nothing_publishable_without_a_resolver(self):
        made = self.fail_once()
        every_take_marked_bad(self.work)
        self.assert_retry_makes_new_takes(made)

    def test_after_the_chooser_found_nothing_publishable(self):
        # Every retained take misread a name a person chose, so the best-effort
        # chooser may publish none of them (as with a misread "Martje").
        self.modern()
        with refusing_chooser():
            made = self.fail_once()
            self.assert_retry_makes_new_takes(made)

    def test_a_second_retry_after_the_chooser_found_nothing_again(self):
        self.modern()
        with refusing_chooser():
            self.fail_once()
            with self.assertRaises((RenderBlocked, SpeechVerificationFailed)):
                self.render_chunk()
            self.assert_retry_makes_new_takes(len(self.engine.calls))

    def test_a_whole_take_that_never_split(self):
        self.engine.heard_override = lambda given, direction, number: given.replace("Arrgh", "Oh")
        self.base.fail_assembly = False
        made = self.fail_once()
        self.assert_retry_makes_new_takes(made)


if __name__ == "__main__":
    unittest.main()
