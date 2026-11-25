"""Coordinate pending profile refreshes with book generation entry points."""

from contextlib import contextmanager
from pathlib import Path
import shlex

from .voice_profile_state import FILENAME, LOCKNAME, read_state


def require_valid_profiles(directory):
    from .analysis_directory import _continuation_entry, _saved_manifest, continuation_directories
    directory = Path(directory).resolve()
    current, chain = directory, []
    # Find the durable owner before transient chain locks or publication checks.
    while current not in chain:
        chain.append(current)
        _require_ready(current, marker_only=True)
        link = _continuation_entry(_saved_manifest(current))
        if link is None:
            break
        current = (current / link["path"]).resolve()
    else:
        raise ValueError("Cycle in voice profile refresh continuation chain")
    for path in chain:
        _require_ready(path)
    continuation_directories(directory)


def _require_ready(directory, *, marker_only=False):
    for name in ((FILENAME,) if marker_only else (FILENAME, LOCKNAME)):
        path = directory / name
        if path.exists() or path.is_symlink():
            command = f"epub-tts-narrator refresh-voice-profiles {shlex.quote(str(directory))}"
            try:
                state = read_state(directory)
            except (OSError, ValueError):
                state = None
            if state and isinstance(state.get("resume_command"), str):
                command = state["resume_command"]
            raise ValueError(f"Voice profiles are invalid while refresh is pending: {directory}. "
                             f"Resume: {command}")


@contextmanager
def refresh_locks(directory):
    from .analysis_directory import continuation_directories
    from .analysis_revision import related_directories, _render_active
    from .renderer.run_locks import acquire_directory_lock, directory_lock_status
    directory = Path(directory).resolve()
    paths = sorted(set(related_directories([directory]) + continuation_directories(directory)))
    locks = []
    try:
        for path in paths:
            locks.append(acquire_directory_lock(path / LOCKNAME))
        for path in paths:
            if path != directory and ((path / FILENAME).exists() or (path / FILENAME).is_symlink()):
                _require_ready(path)
            # Every lock is probed through its lease: a bare existence check would keep
            # blocking on a lock whose process is provably gone (a casting session that
            # was killed leaves its voice-store lock behind).
            sentinels = [path / "voices" / ".lock", path / ".analysis-update.lock"]
            sentinels.extend((path / "voices").glob("*/.lock"))
            held = [f"{marker}: {reason}" for marker in sentinels
                    if (marker.exists() or marker.is_symlink()) and (reason := directory_lock_status(marker))]
            if reason := _render_active(path):
                held.insert(0, reason)
            if held:
                raise ValueError(f"Rendering, casting, or analysis revision is active in the refresh chain: {path} "
                                 f"({'; '.join(held)}). Let it finish, then rerun.")
        yield
    finally:
        for lock in reversed(locks):
            lock.rmdir()
