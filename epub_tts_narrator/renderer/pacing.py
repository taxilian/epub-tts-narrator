"""Conservative PCM edge analysis for assembly; accepted audio stays immutable."""

from array import array
import re
import sys
import wave


def audio_edges(path, *, trim=True, leading_pause_ms=0, trailing_pause_ms=0):
    """Locate quiet outer padding with 10ms peak windows and protective guards.

    Peak detection at -60 dBFS (or -40 dB relative to a very quiet clip) keeps
    breaths and soft consonants. No interior samples are removed. Uncertain or
    wholly silent audio is returned intact; this is not a speech quality gate.
    """
    if any(type(value) is not int or not 0 <= value <= 5000 for value in (leading_pause_ms, trailing_pause_ms)):
        raise ValueError("Intentional outer pauses must be integer milliseconds from zero through 5000")
    with wave.open(str(path), "rb") as source:
        rate, frames = source.getframerate(), source.getnframes()
        window = max(1, round(rate / 100))
        peaks, read = [], 0
        while data := source.readframes(window):
            if len(data) % 2:
                raise ValueError(f"Truncated PCM frame: {path}")
            values = array("h")
            values.frombytes(data)
            if sys.byteorder != "little":
                values.byteswap()
            peaks.append(max(map(abs, values), default=0))
            read += len(values)
        if read != frames:
            raise ValueError(f"Truncated PCM audio: {path}")
    threshold = min(32768 * 10 ** (-60 / 20), max(peaks, default=0) * .01)
    active = [i for i, peak in enumerate(peaks) if peak > threshold]
    if not active:
        return dict(start=0, end=frames, leading=0, trailing=0, frames=frames)
    first, last = active[0] * window, min(frames, (active[-1] + 1) * window)
    start = max(0, first - max(round(rate * .04), round(rate * leading_pause_ms / 1000))) if trim else 0
    end = min(frames, last + max(round(rate * .06), round(rate * trailing_pause_ms / 1000))) if trim else frames
    return dict(start=start, end=end, leading=first - start,
                trailing=end - last, frames=frames)


def punctuation_pause(text, sentence_frames, clause_frames):
    """Closing quote marks do not hide the punctuation that controls a seam."""
    tail = text.rstrip().rstrip('\"\'”’»)]}').rstrip()
    if re.search(r"[.!?…]$", tail):
        return sentence_frames
    if re.search(r"[,;:—–]$", tail):
        return clause_frames
    return 0
