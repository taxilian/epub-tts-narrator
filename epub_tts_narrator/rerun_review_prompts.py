"""Instructions for reviewing a saved book when ``analyze BOOK`` is rerun (``cast_review``, ``direction_review``)."""

from .narrator_prompts import _DATA, _DELIVERY

CAST_REVIEW_VERSION = "1"

NOMINATE = f"""You review the cast of an audiobook for entries that may be one person.
{_DATA}

"cast": one line per entry: id|name|aliases|sex|description|first line it speaks.
"notes" (optional): a listener's own knowledge about some entries: id|later names|gender|origin|notes.
Return {{"r":[[id,id]]}} with every pair of entries that may be the same person under different names: a description or epithet used before the name is revealed ("the hooded figure", "the stranger"), a title or rank, a nickname, a disguise, a first name and a full name, or a later name a listener's note gives. List a pair only when the cast or notes give a reason; return {{"r":[]}} when none. Each pair is checked against the book's text afterwards."""

DIRECTION_REVIEW_VERSION = "1"

DIRECTION_CHECK = f"""You review an audiobook director's line directions. Who speaks is already decided.
{_DATA}

Input: "before"/"after" = neighbouring text for context only, "text" = numbered paragraphs (N|text) where each line is prefixed by its id and speaker, like ‹q17 Mara›; ‹q18 Mara, thought› is a thought and ‹q19 Mara, written› written text read aloud. "d" = each line's current direction ("" = none). "cues" = for some lines, the narration beside the line that says how its speaker says it: a direction must agree with it.
A direction is {_DELIVERY}
Return {{"q":[[qid,delivery]]}} with a corrected delivery only for lines whose current direction is wrong: it contradicts the scene or a cue, invents a feeling, volume, pace or sound the text does not support, overacts an ordinary line, or misses what the speech tag or the scene plainly shows. Leave out every line whose direction is acceptable; most are. Return {{"q":[]}} when none is wrong."""
