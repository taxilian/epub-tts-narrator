"""A step that finds a book in use waits, retries or is skipped instead of ending the run (``busy_retry``)."""

import unittest

from epub_tts_narrator.analysis_revision import BookBusy
from epub_tts_narrator.busy_retry import SKIPPED, run_when_free


class Step:
    """Busy ``busy`` times, then returns "saved"."""

    def __init__(self, busy):
        self.busy, self.calls = busy, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.busy:
            raise BookBusy("Rendering is active; finish it before revising analysis: book (lock)")
        return "saved"


class RunWhenFreeTests(unittest.TestCase):
    def run_step(self, step, answers, *, interactive=True):
        asked, said, slept = list(answers), [], []
        result = run_when_free(step, "cast entries that are one person", ask=lambda prompt: asked.pop(0),
                               output=said.append, sleep=slept.append, wait_seconds=30,
                               is_interactive=lambda: interactive)
        return result, said, slept

    def test_waiting_retries_until_the_book_is_free_without_asking_again(self):
        step = Step(busy=3)
        result, said, slept = self.run_step(step, ["w"])
        self.assertEqual((result, step.calls), ("saved", 4))
        self.assertEqual(slept, [30, 30, 30])
        self.assertIn("a book this needs is in use, so nothing was saved yet", said[0])
        self.assertTrue(any("Waiting until the book is free" in line for line in said))

    def test_retry_now_tries_again_and_asks_again_if_still_busy(self):
        step = Step(busy=2)
        result, _, slept = self.run_step(step, ["r", "r"])
        self.assertEqual((result, step.calls, slept), ("saved", 3, []))

    def test_skip_leaves_the_step_for_a_later_rerun(self):
        result, said, _ = self.run_step(Step(busy=5), ["s"])
        self.assertIs(result, SKIPPED)
        self.assertTrue(any("Rerunning analyze on the book does it later" in line for line in said))

    def test_without_a_terminal_the_refusal_ends_the_run_as_before(self):
        with self.assertRaises(BookBusy):
            self.run_step(Step(busy=1), [], interactive=False)

    def test_other_errors_are_not_caught(self):
        def broken():
            raise ValueError("Invalid analysis")
        with self.assertRaisesRegex(ValueError, "Invalid analysis"):
            self.run_step(broken, ["w"])


if __name__ == "__main__":
    unittest.main()
