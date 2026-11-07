"""Continuous batching rules for Drama 3 decoding, with a scripted fake backend."""

from concurrent.futures import CancelledError
import threading
import unittest

from epub_tts_narrator.renderer.drama3_batch_scheduler import Drama3BatchScheduler, SchedulerClosed

EOS = 0


class ScriptedBackend:
    """Each request's prompt is its script: the frame the model emits each step."""

    def __init__(self, rows):
        self.rows, self.eos_id = rows, EOS
        self.live, self.prefilled, self.released, self.steps, self.log = {}, [], [], [], []
        self.fail_step = None
        self.fail_release = None
        self.gate = None

    def prefill(self, slot, request):
        if request.prompt == "bad prompt":
            raise ValueError("prompt does not fit")
        assert slot not in self.live, "a slot was reused before release"
        self.live[slot] = [request, 0]
        self.prefilled.append((slot, request.seed))
        self.log.append(("prefill", slot, request.seed))

    def step(self, slots):
        if self.gate is not None:
            self.gate.wait(5)
        self.steps.append(list(slots))
        if self.fail_step is not None:
            error, self.fail_step = self.fail_step, None
            raise error
        out = []
        for slot in slots:
            entry = self.live[slot]
            frame = entry[0].prompt[entry[1]]
            entry[1] += 1
            out.append(([7], False) if frame == "nan" else (frame, True))
        return out

    def release(self, slot):
        if self.fail_release is not None:
            raise self.fail_release
        self.live.pop(slot, None)  # like the real backend: releasing a never-filled slot is harmless
        self.released.append(slot)
        self.log.append(("release", slot))


def frames(*mains):
    return [[main, main % 7, main % 5] for main in mains]


class SchedulerTests(unittest.TestCase):
    def scheduler(self, rows):
        backend = ScriptedBackend(rows)
        scheduler = Drama3BatchScheduler(backend)
        self.addCleanup(scheduler.close)
        return backend, scheduler

    def submit(self, scheduler, script, *, max_new=50, seed=0):
        return scheduler.submit(script, max_new, temperature=0.7, top_k=30, top_p=0.7, seed=seed)

    def test_each_request_gets_its_own_frames_and_freed_rows_are_reused(self):
        backend = ScriptedBackend(2)
        backend.gate = threading.Event()  # hold steps until all three are queued
        scheduler = Drama3BatchScheduler(backend)
        self.addCleanup(scheduler.close)
        scripts = [frames(101, 102, 103) + frames(EOS), frames(201) + frames(EOS),
                   frames(301, 302, 303, 304) + frames(EOS)]
        futures = [self.submit(scheduler, script, seed=index) for index, script in enumerate(scripts)]
        backend.gate.set()
        results = [future.result(timeout=5) for future in futures]
        self.assertEqual(results, [(script[:-1], "eos") for script in scripts])
        # Two rows for three requests: the third waited for a freed slot.
        self.assertTrue(all(len(slots) <= 2 for slots in backend.steps))
        self.assertEqual(sorted(seed for _slot, seed in backend.prefilled), [0, 1, 2])
        third = backend.log.index(next(entry for entry in backend.log if entry[:1] == ("prefill",) and entry[2] == 2))
        slot = backend.log[third][1]
        self.assertIn(("release", slot), backend.log[:third])
        self.assertEqual({entry[1] for entry in backend.log}, {0, 1})
        self.assertEqual(scheduler.peak_active, 2)
        self.assertEqual(scheduler.row_steps, sum(len(script) for script in scripts))

    def test_end_of_speech_is_never_returned_and_length_limit_reports_length(self):
        _backend, scheduler = self.scheduler(1)
        self.assertEqual(self.submit(scheduler, frames(EOS)).result(timeout=5), ([], "eos"))
        self.assertEqual(self.submit(scheduler, frames(5, 6, 7, 8), max_new=2).result(timeout=5),
                         (frames(5, 6), "length"))

    def test_a_watch_ends_only_its_own_request_as_silence(self):
        class StopAfter:
            def __init__(self, count):
                self.count, self.seen = count, []

            def observe(self, codes):
                self.seen.append(list(codes))
                return len(self.seen) >= self.count

        _backend, scheduler = self.scheduler(2)
        watch = StopAfter(2)
        silent = scheduler.submit(frames(5, 6, 7, 8) + frames(EOS), 50, temperature=0.7, top_k=30,
                                  top_p=0.7, seed=0, watch=watch)
        other = self.submit(scheduler, frames(9, 9, 9) + frames(EOS))
        self.assertEqual(silent.result(timeout=5), (frames(5, 6), "silence"))
        # The watch sees the codes of each kept frame, never the main token.
        self.assertEqual(watch.seen, [row[1:] for row in frames(5, 6)])
        self.assertEqual(other.result(timeout=5), (frames(9, 9, 9), "eos"))

    def test_a_failing_watch_fails_only_its_own_request(self):
        class Broken:
            def observe(self, codes):
                raise RuntimeError("codec unavailable")

        _backend, scheduler = self.scheduler(2)
        broken = scheduler.submit(frames(5, 6) + frames(EOS), 50, temperature=0.7, top_k=30,
                                  top_p=0.7, seed=0, watch=Broken())
        healthy = self.submit(scheduler, frames(9, 9) + frames(EOS))
        with self.assertRaisesRegex(RuntimeError, "codec unavailable"):
            broken.result(timeout=5)
        self.assertEqual(healthy.result(timeout=5), (frames(9, 9), "eos"))
        self.assertEqual(self.submit(scheduler, frames(3) + frames(EOS)).result(timeout=5), (frames(3), "eos"))

    def test_a_watch_without_observe_is_rejected(self):
        _backend, scheduler = self.scheduler(1)
        with self.assertRaisesRegex(ValueError, "observe"):
            scheduler.submit(frames(EOS), 5, temperature=0.7, top_k=30, top_p=0.7, seed=0, watch=object())

    def test_non_finite_logits_fail_only_that_request(self):
        _backend, scheduler = self.scheduler(2)
        broken = self.submit(scheduler, frames(5) + ["nan"])
        healthy = self.submit(scheduler, frames(9, 9, 9) + frames(EOS))
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            broken.result(timeout=5)
        self.assertEqual(healthy.result(timeout=5), (frames(9, 9, 9), "eos"))

    def test_a_prompt_that_does_not_fit_fails_alone(self):
        backend, scheduler = self.scheduler(2)
        bad = self.submit(scheduler, "bad prompt")
        good = self.submit(scheduler, frames(4) + frames(EOS))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            bad.result(timeout=5)
        self.assertEqual(good.result(timeout=5), (frames(4), "eos"))
        self.assertEqual(backend.live, {})

    def test_a_failed_step_fails_its_rows_and_the_scheduler_keeps_serving(self):
        backend = ScriptedBackend(2)
        backend.fail_step = RuntimeError("CUDA error")
        scheduler = Drama3BatchScheduler(backend)
        self.addCleanup(scheduler.close)
        first = self.submit(scheduler, frames(3, 3) + frames(EOS))
        with self.assertRaisesRegex(RuntimeError, "CUDA error"):
            first.result(timeout=5)
        later = self.submit(scheduler, frames(8) + frames(EOS))
        self.assertEqual(later.result(timeout=5), (frames(8), "eos"))

    def test_an_internal_failure_fails_every_waiting_request_and_refuses_new_work(self):
        backend = ScriptedBackend(1)
        backend.gate = threading.Event()
        scheduler = Drama3BatchScheduler(backend)
        self.addCleanup(scheduler.close)
        backend.fail_release = RuntimeError("slot bookkeeping broke")
        running = self.submit(scheduler, frames(EOS))
        waiting = self.submit(scheduler, frames(2) + frames(EOS))
        backend.gate.set()
        # The finished request still gets its result; nothing is stranded.
        self.assertEqual(running.result(timeout=5), ([], "eos"))
        with self.assertRaises(SchedulerClosed):
            waiting.result(timeout=5)
        with self.assertRaises(SchedulerClosed):
            self.submit(scheduler, frames(EOS))

    def test_requests_being_admitted_are_not_stranded_by_an_internal_failure(self):
        backend = ScriptedBackend(2)
        backend.gate = threading.Event()
        backend.fail_release = RuntimeError("slot bookkeeping broke")
        scheduler = Drama3BatchScheduler(backend)
        self.addCleanup(scheduler.close)
        bad = self.submit(scheduler, "bad prompt")      # its release raises during admission
        neighbour = self.submit(scheduler, frames(EOS))  # admitted in the same round
        backend.gate.set()
        with self.assertRaisesRegex(ValueError, "does not fit"):
            bad.result(timeout=5)
        with self.assertRaises(SchedulerClosed):
            neighbour.result(timeout=5)

    def test_close_fails_queued_requests_and_refuses_new_ones(self):
        backend = ScriptedBackend(1)
        backend.gate = threading.Event()
        scheduler = Drama3BatchScheduler(backend)
        running = self.submit(scheduler, frames(1, 1, 1) + frames(EOS))
        queued = self.submit(scheduler, frames(EOS))
        closer = threading.Thread(target=scheduler.close)
        closer.start()
        backend.gate.set()
        closer.join(5)
        for future in (running, queued):
            with self.assertRaises(SchedulerClosed):
                future.result(timeout=5)
        with self.assertRaises(SchedulerClosed):
            self.submit(scheduler, frames(EOS))

    def test_a_cancelled_request_is_never_prefilled(self):
        backend = ScriptedBackend(1)
        backend.gate = threading.Event()
        scheduler = Drama3BatchScheduler(backend)
        self.addCleanup(scheduler.close)
        first = self.submit(scheduler, frames(1) + frames(EOS), seed=1)
        cancelled = self.submit(scheduler, frames(EOS), seed=2)
        self.assertTrue(cancelled.cancel())
        backend.gate.set()
        first.result(timeout=5)
        with self.assertRaises(CancelledError):
            cancelled.result(timeout=5)
        self.assertEqual([seed for _slot, seed in backend.prefilled], [1])

    def test_invalid_requests_and_backends_are_rejected(self):
        _backend, scheduler = self.scheduler(1)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_new_tokens"):
                scheduler.submit([], value, temperature=0.7, top_k=30, top_p=0.7, seed=0)
        with self.assertRaisesRegex(ValueError, "at least one row"):
            Drama3BatchScheduler(ScriptedBackend(0))


if __name__ == "__main__":
    unittest.main()
