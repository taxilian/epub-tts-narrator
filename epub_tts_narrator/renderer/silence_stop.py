"""End a Drama 3 take once it has gone silent instead of finishing.

Drama 3 sometimes stops speaking mid-line and keeps sampling silent frames for
tens of seconds without ending the take (measured on a finished book: 2.2% of
takes, 5.2% of all generated audio). Such a take is always rejected, so its
remaining frames are wasted decoder time. The watch decodes each take's recent
frames as they arrive and reports when its audio has been silent, by
verification's own measure (25 ms windows below -60 dBFS RMS), for
``STOP_SECONDS`` plus any pauses the text itself requests. The engine then
raises the usual unfinished-take error, which the render records as a rejected
attempt and retries at once.

Verification already rejects any take with more than 2.5 seconds of internal
silence that is not a requested pause, so a take that resumes after the watch
would have stopped it is never one that could have been accepted. Takes that
end on their own are bit-identical with and without the watch: it only reads
frames, never the sampler, which is why it is not part of any engine or cache
identity. The rare take that ends in more than 3 seconds of trailing silence
is retried instead of accepted.
"""

import re

STOP_SECONDS = 3.0
SILENCE_DBFS = -60.0
WINDOW_SECONDS = 0.025
# Frames decoded before the new ones so the codec has left context; a window
# decoded cold differs from the full decode at its start.
CONTEXT_FRAMES = 4
# New frames between checks: about two seconds while the take is speaking,
# half a second once its audio has gone quiet. Each check decodes only the
# newest half second (SILENT_CHECK_FRAMES) unless that is silent. The first
# version decoded every frame once a second and cost 2.5% of decode time on an
# RTX PRO 6000, against the 5.8% the stop saves.
CHECK_FRAMES = 44
SILENT_CHECK_FRAMES = 11

_PAUSE = re.compile(r"\[pause for ([0-9]+(?:\.[0-9]+)?) seconds\]")


def stop_seconds(text: str) -> float:
    """Silence that ends a take: the base allowance plus requested pauses."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return STOP_SECONDS + sum(float(value) for value in _PAUSE.findall(text))


class SilenceWatch:
    """Per-take silence tracker over generated codec frames.

    ``decode(frames)`` turns a list of code frames (each the N codebook values,
    without the main token) into mono float audio scaled to [-1, 1] at
    ``sample_rate``. ``observe(codes)`` takes one generated frame and returns
    True once the take's audio has been continuously silent for the take's
    ``stop_seconds``.
    """

    def __init__(self, text, decode, *, sample_rate, check_frames=CHECK_FRAMES,
                 silent_check_frames=SILENT_CHECK_FRAMES, context_frames=CONTEXT_FRAMES):
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        for name, value in (("check_frames", check_frames), ("silent_check_frames", silent_check_frames)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(context_frames) is not int or context_frames < 0:
            raise ValueError("context_frames must be a non-negative integer")
        self.limit = stop_seconds(text)
        self._decode = decode
        self._window = max(1, round(sample_rate * WINDOW_SECONDS))
        self._sample_rate = sample_rate
        self._threshold = (10 ** (SILENCE_DBFS / 20)) ** 2
        self._check = check_frames
        self._silent_check = silent_check_frames
        self._context = context_frames
        self._frames = []
        self._checked = 0
        self._silent_samples = 0
        self.decodes = 0
        self.stopped = False

    @property
    def silent_seconds(self):
        return self._silent_samples / self._sample_rate

    def observe(self, codes) -> bool:
        """Record one frame's codes as given (a list, or a tensor the caller never overwrites)."""
        if self.stopped:
            return True
        self._frames.append(codes)
        interval = self._silent_check if self._silent_samples else self._check
        if len(self._frames) - self._checked < interval:
            return False
        self._advance()
        self.stopped = self.silent_seconds >= self.limit
        return self.stopped

    def _advance(self):
        # Decode only the newest frames. Voiced audio there ends any silent
        # run, which is all a speaking take ever needs; only a silent tail
        # makes the frames back to the previous check worth decoding.
        end = len(self._frames)
        tail = max(self._checked, end - self._silent_check)
        quiet, trailing, span = self._measure(tail, end)
        if not quiet:
            self._silent_samples = trailing
        else:
            before = self._silent_samples
            if tail > self._checked:
                gap_quiet, gap_trailing, gap_span = self._measure(self._checked, tail)
                before = before + gap_span if gap_quiet else gap_trailing
            self._silent_samples = before + span
        self._checked = end

    def _measure(self, first, end):
        """(all quiet, trailing quiet samples, measured samples) for frames[first:end]."""
        import numpy as np
        start = max(0, first - self._context)
        audio = np.asarray(self._decode(self._frames[start:end]), dtype=np.float64).reshape(-1)
        self.decodes += 1
        per_frame = len(audio) / (end - start)
        audio = audio[int(round((first - start) * per_frame)):]
        whole = len(audio) // self._window * self._window
        if not whole:
            return True, 0, 0
        quiet = (audio[:whole].reshape(-1, self._window) ** 2).mean(axis=1) < self._threshold
        voiced = np.flatnonzero(~quiet)
        trailing = (len(quiet) - 1 - voiced[-1]) if len(voiced) else len(quiet)
        return not len(voiced), int(trailing) * self._window, whole
