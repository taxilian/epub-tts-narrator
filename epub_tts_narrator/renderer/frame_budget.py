"""Per-take Drama 3 frame limit from the length of the text it speaks.

Drama 3 sometimes stops speaking without ending the take: it keeps sampling
near-silent frames, or rarely keeps babbling, until the configured frame limit
(2048 frames, about 95 seconds). A one-word line can then hold a decoder for a
minute and a half before it is rejected and retried. The budget ends such a
take once it is far longer than any accepted reading of its text, so the retry
starts sooner. A take that ends on its own within the budget is unchanged,
which is why the budget is not part of any engine or cache identity.

Measured on a finished 12,734-passage book, the slowest accepted take used 0.48
seconds per spoken character and none exceeded the budget below; the budget
only shortens attempts that verification rejects (as an unfinished take).
"""

import math
import re

# Drama 3's codec: 44.1 kHz audio, one frame per 2048 samples.
FRAMES_PER_SECOND = 44100 / 2048
BASE_SECONDS = 10.0
SECONDS_PER_UNIT = 0.5
# A digit is read as a word or more ("1923", "4th"), never as one letter.
DIGIT_UNITS = 3

# Speaker tags and bracketed performance directions are not read aloud.
_UNSPOKEN = re.compile(r"<\|[^|>]*\|>|\[[^\]]*\]")


def spoken_units(text: str) -> int:
    """Letters plus weighted digits outside speaker tags and directions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    spoken = _UNSPOKEN.sub(" ", text)
    return sum(DIGIT_UNITS if character.isdigit() else 1
               for character in spoken if character.isalnum())


def frame_budget(text: str, limit: int) -> int:
    """At most ``limit`` frames; fewer for text too short to need them."""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    seconds = BASE_SECONDS + SECONDS_PER_UNIT * spoken_units(text)
    return min(limit, math.ceil(seconds * FRAMES_PER_SECOND))
