"""BacklogBudget monotonic peak gauges: truth through reuse, release, and races."""

from threading import Lock, Thread
import unittest

from epub_tts_narrator.renderer.render_backlog import BacklogBudget


PEAK_KEYS = {"peak_pending_count", "peak_materialized_bytes"}


class BacklogPeakTests(unittest.TestCase):
    def budget(self, **kwargs):
        budget = BacklogBudget(**kwargs)
        self.addCleanup(budget.close)
        return budget

    def test_reuse_and_release_keep_peaks_while_current_counts_return_to_zero(self):
        budget = self.budget(max_pending=4, max_materialized_bytes=1000)
        with budget.reserve(new_generation=True, initial_bytes=0) as ticket:
            ticket.materialize(400)
            first = budget.snapshot()
            self.assertEqual(first["pending_count"], 1)
            self.assertEqual(first["peak_pending_count"], 1)
            self.assertEqual(first["materialized_bytes"], 400)
            self.assertEqual(first["peak_materialized_bytes"], 400)
        with budget.reserve(new_generation=False, initial_bytes=0) as second:
            self.assertEqual(second.accounted_bytes, 0)
            reused = budget.snapshot()
            # The current byte count is zero but the peak must stay above it.
            self.assertEqual(reused["materialized_bytes"], 0)
            self.assertEqual(reused["peak_materialized_bytes"], 400)
            self.assertEqual(reused["peak_pending_count"], 1)
        drained = budget.snapshot()
        self.assertEqual(drained["pending_count"], 0)
        self.assertEqual(drained["peak_pending_count"], 1)
        self.assertEqual(drained["peak_materialized_bytes"], 400)

    def test_peaks_track_real_overshoot_above_the_soft_high_water(self):
        budget = self.budget(max_pending=2, max_materialized_bytes=500)
        with budget.reserve(initial_bytes=0) as first:
            with budget.reserve(initial_bytes=0) as second:
                # Already-admitted attempts may land and overshoot the soft
                # byte mark; the peak records what actually materialized.
                first.materialize(4000)
                second.materialize(6000)
                gauges = budget.snapshot()
                self.assertLessEqual(gauges["peak_pending_count"], 2)
                self.assertEqual(gauges["peak_materialized_bytes"], 10000)
                self.assertGreater(gauges["peak_materialized_bytes"], 500)

    def test_concurrent_reservations_keep_peaks_monotonic_and_consistent(self):
        budget = self.budget(max_pending=4, max_materialized_bytes=100000)
        errors = []

        def worker(index):
            try:
                for round_index in range(50):
                    with budget.reserve(initial_bytes=index + round_index) as ticket:
                        ticket.materialize((index + round_index) * 7)
            except BaseException as error:  # pragma: no cover - failure reporter
                errors.append(error)

        threads = [Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        gauges = budget.snapshot()
        self.assertEqual(gauges["pending_count"], 0)
        self.assertEqual(gauges["materialized_bytes"], 0)
        self.assertGreater(gauges["peak_pending_count"], 0)
        self.assertLessEqual(gauges["peak_pending_count"], 4)
        self.assertGreater(gauges["peak_materialized_bytes"], 0)

    def test_snapshot_stays_a_bounded_numeric_contract(self):
        budget = self.budget(max_pending=3, max_materialized_bytes=900)
        with budget.reserve(initial_bytes=10):
            gauges = budget.snapshot()
        self.assertEqual(set(PEAK_KEYS) - set(gauges), set())
        for key, value in gauges.items():
            self.assertIsInstance(value, (int, bool), key)

    def test_reservation_context_materialize_updates_the_byte_peak(self):
        budget = self.budget(max_pending=2, max_materialized_bytes=1000)
        with budget.reserve(initial_bytes=5) as ticket:
            ticket.update_bytes(60)
            joined = budget.snapshot()
            self.assertEqual(joined["peak_materialized_bytes"], 60)
            ticket.update_bytes(30)  # absolute re-record after a trim
            trimmed = budget.snapshot()
            self.assertEqual(trimmed["materialized_bytes"], 30)
            self.assertEqual(trimmed["peak_materialized_bytes"], 60)


if __name__ == "__main__":
    unittest.main()
