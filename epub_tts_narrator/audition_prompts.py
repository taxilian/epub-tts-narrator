"""Character audition script request instruction and its prompt version."""

from .prompts import COMMON


PROMPT_VERSION = 2
SYSTEM = COMMON + """
Create an original character-specific audition script, not audiobook content.
The available book excerpts are too short or unsuitable for a reusable reference.
Write 35–50 words (hard bounds 25–60 words and 650 characters) that this character
could naturally say, using the supplied personality, setting, speech style and
short examples. For a narrator, write a short passage in their narrative style.
Use several complete sentences with varied natural phrasing and normal delivery.
Write plain spoken prose. Do not wrap the script or individual statements in
quotation marks; preserve quotation marks only when the character quotes wording
within their speech, and preserve normal apostrophes.
Do not recite the voice description, explain the task, introduce the speaker by
name, or include acting directions, headings, sound effects, brackets or markup.
Do not imitate shouting, repeated-letter hissing, sobbing or other extreme delivery.
This invented audition will never enter the audiobook or be claimed as a quotation.
Return exactly {"text":"audition speech only","reason":"brief casting rationale"}.
"""


EXTRA_PROMPT_VERSION = 1
EXTRA_SYSTEM = COMMON + """
Write further original reference scripts for one character's voice recording.
The voice is recorded speaking several short scripts that are joined into one
reference recording, and the book has too few clean excerpts by this character
for all of them. Write exactly the requested number of scripts.
Each script is 35–50 words (hard bounds 25–60 words and 650 characters) that this
character could naturally say, in the same personality, setting and speech style
as the supplied existing scripts and short examples. For a narrator, write short
passages in their narrative style. Each script must differ from the existing
scripts and from the others in content and sentence shapes; vary statements and
questions with natural rhythm, keeping normal delivery.
Write plain spoken prose. Do not wrap the script or individual statements in
quotation marks; preserve quotation marks only when the character quotes wording
within their speech, and preserve normal apostrophes.
Do not recite the voice description, explain the task, introduce the speaker by
name, or include acting directions, headings, sound effects, brackets or markup.
Do not imitate shouting, repeated-letter hissing, sobbing or other extreme delivery.
These invented scripts will never enter the audiobook or be claimed as quotations.
Return exactly {"scripts":[{"text":"speech only","reason":"brief casting rationale"}]}.
"""
