"""Instructions for reviewing a saved book when ``analyze BOOK`` is rerun (``cast_review``, ``direction_review``)."""

from .narrator_prompts import _DATA, _DELIVERY

CAST_REVIEW_VERSION = "1"

NOMINATE = f"""You review the cast of an audiobook for entries that may be one person.
{_DATA}

"cast": one line per entry: id|name|aliases|sex|description|first line it speaks.
"notes" (optional): a listener's own knowledge about some entries: id|later names|gender|origin|notes.
Return {{"r":[[id,id]]}} with every pair of entries that may be the same person under different names: a description or epithet used before the name is revealed ("the hooded figure", "the stranger"), a title or rank, a nickname, a disguise, a first name and a full name, or a later name a listener's note gives. List a pair only when the cast or notes give a reason; return {{"r":[]}} when none. Each pair is checked against the book's text afterwards."""

SAME_PERSON_VERSION = "1"

# The rerun cast review's check of each pair; a "y" merges two entries' lines into one voice.
SAME_PERSON_CHECK = f"""You check whether two entries in an audiobook cast are one person, so that every line of both is read in one voice.
{_DATA}

Each item gives entries "a" and "b" (id|name|aliases|sex|description), numbered paragraphs (N|text), and "a_lines"/"b_lines": the paragraphs whose spoken lines are now given to each entry.
{{"r":[[item_id,"y",N,quote],[item_id,"p"],[item_id,"n"],[item_id,"?"]]}} for every item:
"y" only when the text shows a and b are the same individual and every line in a_lines and b_lines is theirs: the narration or a speaker calls one by the other's name, or the description is used of the named character in the same scene. Give a short exact quote from paragraph N that shows it.
"p" when some lines of one entry are the other's but others belong to someone else (an entry that mixes several people).
"n" when the text treats them as two people: both present at once, one speaks to or about the other, different places, roles or relationships, or one belongs to the other (X's servant, X's wife, X's people).
"?" when these paragraphs do not show it.
A resemblance is not evidence: sharing a people, nation, family, title, rank, occupation, a similar description or a form of address ("your Grace", "Sir Knight", "general") does not make two entries one person, and two unnamed people of the same kind (two guards, two teamsters) in different scenes are different people unless the text shows the same individual returns. Aliases in the entries may be wrong; the text wins."""

# Used instead of NOMINATE only when a person gave notes with ``analyze BOOK --instructions``.
NOMINATE_WITH_INSTRUCTIONS = f"""You review the cast of an audiobook for entries that may be one person.
{_DATA}

"cast": one line per entry: id|name|aliases|sex|description|first line it speaks.
"notes" (optional): a listener's own knowledge about some entries: id|later names|gender|origin|notes.
"instructions": a listener's notes about this book's cast; fallible leads, the text is checked afterwards.
Return {{"r":[[id,id]],"same":[[id,id]],"apart":[[id,id]]}}. "r": every pair of entries that may be the same person under different names: a description or epithet used before the name is revealed ("the hooded figure", "the stranger"), a title or rank, a nickname, a disguise, a first name and a full name, or a later name a listener's note gives; list a pair only when the cast or notes give a reason. "same": pairs the instructions say are one person. "apart": pairs the instructions say are different people. Omit a key with no pairs. Each pair is checked against the book's text afterwards."""

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

NARRATOR_NOTES_VERSION = "1"

# Asked once per new set of notes: most notes are about the cast, and a book with one narrator keeps it.
NARRATOR_NOTES_CHECK = f"""You read a listener's notes about an audiobook and decide whether they ask for its narration to be read by its point-of-view characters.
{_DATA}

"notes": the listener's notes.
Return {{"r":"y"}} when the notes ask for the narration (the text outside dialogue) to be read in the voice of the character whose point of view it follows, or say which characters' points of view the narration follows in which chapters, sections or scenes; {{"r":"n"}} when the notes are about something else (who is who in the cast, which entries are one person, who says a line, names, genders, accents, pronunciations, directions) or say the narration keeps one narrator; {{"r":"?"}} when you cannot tell."""

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

JOINT_REVIEW_VERSION = "1"

JOINT_CHECK = f"""You check audiobook lines whose speech tag says several people speak but which were given to one character.
{_DATA}

Each item gives the "line", its "tag", "speakers" (number|name: the characters who speak in this scene, the line's current speaker first) and the scene's numbered paragraphs (N|text), the line's own paragraph last.
{{"r":[[item_id,[numbers],N,quote]]}} when the text shows that two or more of these speakers say these exact words together (all at once, as one, in chorus), with their speaker numbers and a short exact quote from paragraph N that shows them speaking together; [item_id,"n"] when the line is one speaker's, or is said by people who are not among these speakers; [item_id,"?"] when the paragraphs do not show who says it. Name only speakers the text shows taking part: "the voices all shouted" means the voices heard in this scene, not every character present."""


SPLIT_REVIEW_VERSION = "1"

SPLIT_NOMINATE = f"""You read a listener's notes about an audiobook's cast, looking for entries that wrongly hold two characters' lines.
{_DATA}

"cast": one line per entry: id|name|aliases|description|first line it speaks.
"instructions": the listener's notes about this book; fallible leads, the text is checked afterwards.
Return {{"r":[[id,"who"]]}}: each entry the instructions say gives some of one character's lines to another (for example that a voice heard in one scene is not the creature of an earlier scene), with "who": a few words naming that other character as the notes describe them. Return {{"r":[]}} when the notes say nothing of the kind; entries the notes say are one person are not split."""

SPLIT_CHECK = f"""You check an audiobook cast entry that may hold the lines of two different characters.
{_DATA}

"instructions": a listener's notes about this book; fallible leads to check against the text.
The item gives the "entry" (name|aliases|description), "who": the other character the notes describe, and numbered paragraphs (N|text) holding the entry's spoken lines, in book order.
{{"r":[[item_id,name,description,[N...],N,quote]]}} when the text shows that the entry's lines in paragraphs [N...] are spoken by that other character: name is the other character as the book calls them (a short descriptive name when the book gives none, such as "Third Voice"), description is one sentence on who they are and how they sound, starting with Male, Female or Androgynous, [N...] lists every paragraph whose entry lines are theirs (never all of them), and quote is a short exact text from paragraph N showing they are someone other than the entry. [item_id,"n"] when every line is the entry's; [item_id,"?"] when the paragraphs do not show it."""

NAME_REVIEW_VERSION = "1"

# Which entries are listed under a description although the book gives their personal name.
NAME_NOMINATE = f"""You review the cast of an audiobook for entries listed under a description, title, role or epithet ("Eosian Beacon", "the old guard", "Third Voice") although the book gives that character a personal name ("Aurora").
{_DATA}

"cast": one line per entry: id|name|aliases|note.
Return {{"r":[[id,name]]}}: for each entry whose listed name is not its personal name while its aliases or note give one, that personal name exactly as the book spells it. Not a title, rank, nickname-by-role, species, place or another character's name. Return {{"r":[]}} when none. Each name is checked against the book's text afterwards."""

NAME_CHECK = f"""You check whether a character in an audiobook cast is given a personal name by the book.
{_DATA}

Each item gives an entry (its listed name|aliases|note), the proposed personal "name", and numbered paragraphs (N|text): some with the entry's own lines or listed name, some with the proposed name.
{{"r":[[item_id,"y",N,quote],[item_id,"n"],[item_id,"?"]]}} for every item:
"y" only when the text shows the proposed name is this same character's own name: the character introduces itself by it ("my name is Aurora"), others address or call this character by it, or the narration names the described character with it. Give a short exact quote from paragraph N, containing the name, that shows it.
"n" when the name belongs to someone or something else, or is a title, rank, role, species or place.
"?" when these paragraphs do not show it."""

AUTHOR_REVIEW_VERSION = "1"

# Asked only when the EPUB names no author (or only a placeholder such as "Unknown"): ``author_review``.
AUTHOR_FIND = f"""You find who wrote a book, from the book's own front and back matter.
{_DATA}

Input: "title" = the book's title; "text" = numbered paragraphs (N|text) from its start and end: title page, copyright page, dedication, "also by" lists, about the author.
Return {{"a":[[name,N,quote]]}} with each author of this book that the text names, each with a short exact quote from paragraph N that contains the name and shows it is this book's author: a byline ("by NAME"), a copyright notice ("Copyright © 2002 by NAME"), "Other books by NAME", "About the author" followed by the name. Give the name as the book prints it, in normal capitalization. An editor, translator, illustrator, cover artist, publisher or a co-author of only other books listed is not an author of this book.
Return {{"a":[]}} when these paragraphs do not show who wrote this book. Never guess from the title, the story, or your own knowledge of books."""
