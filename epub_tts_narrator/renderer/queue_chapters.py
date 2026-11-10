"""One bounded, book-ordered chapter coordinator pool for a render queue."""

from __future__ import annotations

from concurrent.futures import Future
import heapq
from itertools import count
from threading import Condition, Thread


class QueueChapterCoordinators:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Queue chapter capacity must be a positive integer")
        self.capacity = capacity
        self._condition = Condition()
        self._pending = []
        self._sequence = count()
        self._closed = False
        self._threads = [Thread(target=self._work, name=f"queue-chapter-{index + 1}")
                         for index in range(capacity)]
        for thread in self._threads:
            thread.start()

    def run(self, units, admission, *, priority, on_submitted=None):
        futures = []
        with self._condition:
            if self._closed:
                raise RuntimeError("Queue chapter coordinators are closed")
            for index, unit in enumerate(units):
                future = Future()
                futures.append(future)
                heapq.heappush(self._pending, (priority, index, next(self._sequence),
                                              unit, admission, future))
            if on_submitted is not None:
                on_submitted()
            self._condition.notify_all()
        results = [None] * len(futures)
        failure = None
        for index, future in enumerate(futures):
            try:
                results[index] = future.result()
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
        return results

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    return
                _, index, _, unit, admission, future = heapq.heappop(self._pending)
            try:
                admission.enter()
                result = unit(index)
            except BaseException as error:
                admission.fail(error)
                future.set_exception(error)
            else:
                future.set_result(result)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join()
