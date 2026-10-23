"""Run-local speech generation limits, separate from retained take numbering."""

from __future__ import annotations

from pathlib import Path
import re
import stat
from threading import Lock


_GENERATION_FILE = re.compile(
    r"attempt-([0-9]{3,})(?:\.(?:json|wav)|"
    r"-(?:raw|recovery-raw|regenerated-[0-9a-f]{16})\.wav)"
)


def generation_attempts(directory):
    """Inventory actual take files without assuming consecutive attempt IDs.

    Orphan audio still reserves its ID so a later invocation cannot overwrite
    it. Unknown and noncanonical names remain untouched and are not interpreted
    as renderer-owned attempts. Recognized names must be regular files.
    """
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Speech attempt directory must not be a symlink")
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("Speech attempt path must be a directory")
    attempts = set()
    for path in directory.iterdir():
        match = _GENERATION_FILE.fullmatch(path.name)
        if match is None:
            continue
        number = int(match[1])
        if number < 1 or match[1] != f"{number:03d}":
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Speech attempt must be a regular, nonsymlink file: {path}")
        attempts.add(number)
    return sorted(attempts)


class SpeechRetryBudget:
    """Share one bounded allowance across a render invocation and its replans.

    Claims count actual generation starts, including a provider error, rather
    than saved candidates inspected or reverified. Callers hold their per-take
    lock while choosing filenames; this lock protects only the run allowance.
    """

    def __init__(self, *, retry_paragraphs=False):
        # Local book rendering splits an exhausted paragraph into its source
        # passages. Standalone chapter jobs can instead opt into fresh paragraph
        # attempts because they do not own that book-level recovery plan.
        if type(retry_paragraphs) is not bool:
            raise ValueError("retry_paragraphs must be a boolean")
        self.retry_paragraphs = retry_paragraphs
        self._counts = {}
        self._episodes = {}
        self._lock = Lock()

    @staticmethod
    def _key(directory, request_hash):
        return Path(directory).resolve(), request_hash

    def claim(self, directory, request_hash, maximum, *, episode=None, episode_maximum=None,
              episode_reserve=0):
        """Return this run's next ordinal, or None if its allowance is spent."""
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Speech generation allowance must be a positive integer")
        if episode is not None and (not isinstance(episode, str) or
                type(episode_maximum) is not int or episode_maximum < 1):
            raise ValueError("Speech resolution requires a bounded parent allowance")
        if type(episode_reserve) is not int or episode_reserve < 0:
            raise ValueError("Speech recovery reserve must be a nonnegative integer")
        key = self._key(directory, request_hash)
        with self._lock:
            started = self._counts.get(key, 0)
            if started >= maximum:
                return None
            if episode is not None:
                total = self._episodes.get(episode, 0)
                if total >= episode_maximum - episode_reserve:
                    return None
                self._episodes[episode] = total + 1
            started += 1
            self._counts[key] = started
            return started

    def started(self, directory, request_hash):
        """Return the number of generation starts claimed in this invocation."""
        key = self._key(directory, request_hash)
        with self._lock:
            return self._counts.get(key, 0)

    def episode_started(self, episode):
        """Return total starts across a whole take and all of its recovery parts."""
        with self._lock:
            return self._episodes.get(episode, 0)


def whole_strategy_allowance(maximum, fallback_parts):
    """Keep roughly half the bounded episode for shorter generation requests.

    At least one whole take is tried. Where the total limit permits, every
    fallback part retains one opportunity; no new allowance is created.
    """
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Speech generation allowance must be a positive integer")
    if type(fallback_parts) is not int or fallback_parts < 0:
        raise ValueError("Speech fallback count must be a nonnegative integer")
    if not fallback_parts:
        return maximum
    return max(1, min(maximum // 2, maximum - fallback_parts))
