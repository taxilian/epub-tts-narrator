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

GENDER_REVIEW_VERSION = "1"

GENDER_CHECK = f"""You check the gender of characters in an audiobook cast against the book's text.
{_DATA}

Each item gives an entry (name|aliases|the gender its voice was given) and numbered paragraphs (N|text) that mention it.
{{"r":[[item_id,gender,N,quote]]}} for every item: gender "Male", "Female" or "Androgynous" as the text shows it for this character (pronouns that refer to them, words like woman or man, titles), with a short exact quote from paragraph N that shows it; or [item_id,"?"] when these paragraphs do not show it. Only the text counts; the given gender may be wrong."""

NARRATOR_REVIEW_VERSION = "1"

NARRATOR_CHECK = f"""You decide whose point of view each section of an audiobook follows, so that section's narration can be read in that character's voice.
{_DATA}

"notes": a listener's description of how this book's narration works (for example which characters' points of view it follows, or that it changes between chapters): fallible leads to check against the text; the text wins.
"cast": one line per entry: id|name|aliases.
"items": sections of the book in order, each as numbered paragraphs (N|text) of its opening.
Return {{"r":[[section_id,cast_id,N,quote]]}} for every section: the cast entry whose point of view the section's narration follows (whose thoughts, perceptions and feelings the narration reports from the inside), with a short exact quote from paragraph N that shows it; or [section_id,"-"] when the narration follows no single cast member (a title page, a poem, an epigraph, an omniscient or shifting narration) or the notes say the book's main narrator reads it. Being named or speaking is not enough: the narration must be told from that character's point of view."""

GROUP_REVIEW_VERSION = "1"

GROUP_CHECK = f"""You check audiobook cast entries whose lines are sometimes tagged as said by several people.
{_DATA}

Each item gives an entry (name|aliases) and numbered paragraphs (N|text) with its lines.
{{"r":[[item_id,"y",N,quote]]}} when the text shows this entry is several people who speak together or as one (a group, a crowd, a council, a chorus), with a short exact quote from paragraph N that shows several people speaking; [item_id,"n"] when the entry is one person (a plural tag then belongs to a different line or speaker); [item_id,"?"] when these paragraphs do not show it. A name alone is no evidence either way: only what the text says about who speaks."""
