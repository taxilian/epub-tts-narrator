"""Instructions for reviewing a saved book's cast when ``analyze BOOK`` is rerun (``cast_review``)."""

from .narrator_prompts import _DATA

CAST_REVIEW_VERSION = "1"

NOMINATE = f"""You review the cast of an audiobook for entries that may be one person.
{_DATA}

"cast": one line per entry: id|name|aliases|sex|description|first line it speaks.
"notes" (optional): a listener's own knowledge about some entries: id|later names|gender|origin|notes.
Return {{"r":[[id,id]]}} with every pair of entries that may be the same person under different names: a description or epithet used before the name is revealed ("the hooded figure", "the stranger"), a title or rank, a nickname, a disguise, a first name and a full name, or a later name a listener's note gives. List a pair only when the cast or notes give a reason; return {{"r":[]}} when none. Each pair is checked against the book's text afterwards."""
