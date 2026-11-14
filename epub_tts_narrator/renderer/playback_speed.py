"""One listening speed, 1x or 2x, shared by every place that plays audio to a person.

Every ``audition.AudioPlayer`` reads the process-wide ``SHARED`` speed unless
it was given its own, so the run shell's takes, ``review-ui``, casting,
``pronounce`` and the pronunciation queue all play at the speed last chosen
in any of them. ``>`` switches it everywhere (``KEY``). The speed changes
tempo only, never pitch: ``afplay`` plays with Apple's spectral time-pitch
rate (``-r 2 -q 1``), and any other configured player gets a temporary
FFmpeg ``atempo`` copy; the saved audio is never changed.
"""

from __future__ import annotations

RATES = (1.0, 2.0)
KEY = ">"


class PlaybackSpeed:
    """A listening rate that switches between 1x and 2x."""

    def __init__(self, rate=1.0):
        if rate not in RATES:
            raise ValueError("Playback speed must be 1x or 2x")
        self.rate = float(rate)

    def toggle(self):
        self.rate = RATES[(RATES.index(self.rate) + 1) % len(RATES)]
        return self.rate

    @property
    def label(self):
        return f"{self.rate:g}x"


SHARED = PlaybackSpeed()


def speed_of(player=None):
    """The speed ``player`` plays at: its own when it has one, else the shared speed."""
    speed = getattr(player, "speed", None)
    return speed if isinstance(speed, PlaybackSpeed) else SHARED


def label(player=None):
    return speed_of(player).label


def toggle(player=None):
    """Switch between 1x and 2x; a player that is playing restarts its audio at the new speed.

    Returns the new label. Restarting is cheap (the same file from its
    start), so a person hears the change at once instead of at the next play.
    """
    speed = speed_of(player)
    speed.toggle()
    restart = getattr(player, "restart", None)
    if callable(restart):
        restart()
    return speed.label


def help_text(player=None):
    """The short help entry every screen and line menu shows: the key and the current speed."""
    return f"{KEY} speed ({label(player)})"


def toggled_notice(player=None):
    return f"Playback speed is now {label(player)} ({KEY} switches between 1x and 2x; pitch is unchanged)."


def line_command(command, player=None):
    """A line menu's ``>`` command: switch the speed and return the notice, else None."""
    if command.strip() != KEY:
        return None
    toggle(player)
    return toggled_notice(player)
