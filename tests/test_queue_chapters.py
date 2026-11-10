"""Queue-wide chapter capacity and book priority, using real coordinators."""

import threading
import unittest

from epub_tts_narrator.renderer.queue_chapters import QueueChapterCoordinators
from epub_tts_narrator.renderer.render_chapter_flow import ChapterRunAdmission, run_units


class QueueChapterTests(unittest.TestCase):
    def test_earlier_books_refill_the_global_limit_before_later_books_start(self):
        pool = QueueChapterCoordinators(2)
        self.addCleanup(pool.close)
        entered = [threading.Event() for _ in range(4)]
        release = [threading.Event() for _ in range(4)]
        later_submitted = threading.Event()
        later_started = threading.Event()
        lock = threading.Lock()
        active = 0
        peak = 0
        results = []

        def early(index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            entered[index].set()
            if not release[index].wait(5):
                raise RuntimeError("test failed to release an earlier chapter")
            with lock:
                active -= 1
            return index

        def later(index):
            later_started.set()
            return index

        def book(priority, units, on_submitted=None):
            results.append(run_units(units, 2, ChapterRunAdmission(), coordinator_pool=pool,
                                     priority=priority, on_submitted=on_submitted))

        first = threading.Thread(target=book, args=(0, [early] * 4))
        second = threading.Thread(target=book, args=(1, [later], later_submitted.set))
        first.start()
        try:
            self.assertTrue(entered[0].wait(5))
            self.assertTrue(entered[1].wait(5))
            second.start()
            self.assertTrue(later_submitted.wait(5))
            self.assertFalse(later_started.is_set())
            release[0].set()
            self.assertTrue(entered[2].wait(5))
            self.assertFalse(later_started.is_set(), "earlier pending chapters must refill first")
            release[1].set()
            self.assertTrue(entered[3].wait(5))
            self.assertFalse(later_started.is_set())
            release[2].set()
            self.assertTrue(later_started.wait(5), "later book must fill the freed tail slot")
        finally:
            for event in release:
                event.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(peak, 2)
        self.assertIn([0, 1, 2, 3], results)
        self.assertIn([0], results)

    def test_failed_book_does_not_close_another_books_chapter_admission(self):
        pool = QueueChapterCoordinators(1)
        self.addCleanup(pool.close)

        def fail(index):
            raise ValueError("chapter failure")

        with self.assertRaisesRegex(ValueError, "chapter failure"):
            run_units([fail, fail], 1, ChapterRunAdmission(), coordinator_pool=pool, priority=0)
        result = run_units([lambda index: "completed"], 1, ChapterRunAdmission(),
                           coordinator_pool=pool, priority=1)
        self.assertEqual(result, ["completed"])
