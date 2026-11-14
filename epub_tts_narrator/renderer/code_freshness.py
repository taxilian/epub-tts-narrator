"""Whether a long-running process still runs the code in this checkout.

A local ``serve`` worker outlives the commands that start it, so a change to
the renderer does not reach it until it restarts. The server reports which of
its loaded modules changed on disk after it started (``/health``
``code_changed``), and clients replace an idle server whose code is older.
"""

from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # epub_tts_narrator/


def changed_since(started, *, modules=None):
    """Package source files of loaded modules modified after ``started`` (epoch seconds), sorted.

    Paths are relative to the repository root, e.g.
    ``epub_tts_narrator/performance_plan.py``. A file that cannot be read
    counts as changed: the running copy can no longer be the one on disk.
    """
    changed = set()
    for module in list((modules if modules is not None else sys.modules).values()):
        name = getattr(module, "__file__", None)
        if not name or not name.endswith(".py"):
            continue
        path = Path(name).resolve()
        if PACKAGE_ROOT not in path.parents:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = float("inf")
        if modified > started:
            changed.add(path.relative_to(PACKAGE_ROOT.parent).as_posix())
    return sorted(changed)
