"""Versioned compact attribution prompts. Prompt bytes are cache identity."""

from .prompts import COMMON, PERFORMANCE_POLICY_GUIDANCE, WHISPER_DELIVERY_POLICY


COMPACT_ATTRIBUTION_PROTOCOL = "compact-attribution-v1"

# Frozen for cache-only replay of responses validated under the original wire.
ATTRIBUTE_COMPACT_V1 = COMMON + PERFORMANCE_POLICY_GUIDANCE + """
Assign each target unit's exact text to speaking roles in the final cast. The
final_cast is compact: character/role/voice IDs, names, aliases, and identity
or scope descriptions only. Context units and prior_segments are context only:
do not emit them as target text. Never echo source text in the response.
Return {"units":[{"unit_id":"existing target ID", "segments":[{
 "start_anchor":"exact unique substring that begins this segment; omit for the
   first segment",
 "kind":"narration|dialogue|thought|letter|heading|other",
 "role_id":"existing role ID or null", "basis":"explicit|context|unresolved",
 "reason":"brief attribution explanation", "evidence":[],
 "candidates":["existing role ID"], "review_required":false,
 "delivery":{"direction":"", "evidence":[]}
}]}]}.
Segment boundaries are reconstructed locally: the first segment starts at the
unit start (omit start_anchor there), every later segment's start_anchor is an
exact substring that occurs exactly once in that unit and strictly after the
previous segment's start, and each segment extends to the next anchor or the
unit end. If repeated text cannot identify a boundary, use
"start_anchor":{"left":"exact text immediately before the boundary",
"right":"exact text beginning at the boundary"}. Their concatenation must
occur exactly once; the boundary is after left. This also identifies a repeated
phrase at the unit end without counting offsets. Copy anchors verbatim,
preserving spelling, case, punctuation, and whitespace. Never include a "text"
field, never omit or reorder any source
characters, and never split inside a word.
For the SAME words explicitly spoken together by two or more known characters,
add "simultaneous_role_ids":["existing dialogue role ID","another dialogue role ID"]
to that segment, set role_id=null, basis=explicit or context, candidates=[], and
review_required=false. Include exact evidence for joint delivery and every
performer's identity. List distinct dialogue roles with distinct cast voices;
never use a narrator role. Do not create a new character or voice for the group.
This array names actual simultaneous performers, NEVER alternative guesses.
For example, '"Run," Mara and Lena said together.' has one dialogue segment
with both existing dialogue roles, followed by a separate narration segment.
Omit simultaneous_role_ids for single speakers and genuinely uncertain voices.
The delivery object above is the valid default. A nonempty delivery.direction
MUST have a nonempty evidence array with exact quotations supporting that local
direction. If there is no such support, keep direction="" and evidence=[].
Check this requirement for every segment, not only the first spoken line.
Emit each target unit exactly once, in input order.
Split within a paragraph wherever the speaker changes. For example, a quotation,
the words 'Mara said', and the continuation of the quotation require three spans.
Narration such as 'she whispered' is voiced by the narrator, not the character.
When the same character speaks before and after an intervening narration tag,
keep THREE spans: character speech, kind=narration with the applicable narrator
role, then character speech. Continuing dialogue does not include 'she said',
'he added', or actions outside the quotation. This role/kind boundary still
applies when the narrator and the speaking character share the same voice.
Attribution confidence concerns who is speaking, not the completeness of their
voice profile. Unspecified vocal age, gender, accent, and ordinary casting choices
do not make a clearly named speech tag uncertain or require attribution review.
Inspect a paragraph's opening quotation before assigning its narrative frame.
For example, in '“Go,” Mara said.', Mara's quoted words and the narrator's speech
tag are separate spans. A surrounding narrative frame does not turn that direct
speech into narration; narrative actions between utterances remain narration.
Use book-wide cast identity, chapter scope, local context, and actual speech
attribution, not mechanical turn alternation. A first-person character's narration
and dialogue may use separate roles that share the same voice.
A silent reaction, glance, gesture, or narrative mention of another character
does not transfer the speaking turn. Track the current speaker through these
interruptions and through continuing multi-paragraph quotations. Check what
first-person claims refer to in the surrounding scene; do not select whoever was
most recently named. Switch speakers only when the actual words or context
support a change, and flag a genuinely unclear transition for review.
Use the supplied performance_policy for thoughts, letters, nested quotation,
and nonstory content. Identify the actual thinker or sender before applying that
policy. Directly rendered written or mental messages belong to their sender;
the medium and the surrounding first-person narration do not make the narrator
the sender. Use the sender's role as directed by the applicable policy and keep
narrative descriptions of sending or receiving separate from the message itself.
Distinguish an explicitly rendered direct thought from retrospective narrative
reflection. When the thought policy calls for the thinking character's dialogue
role, use it even if that character is also the narrator. Typography alone does
not establish thought, speech, or identity.
A quotation inside a character's speech is normally still performed by that
character, following the nested-quotation policy.
Unclear speaker/identity/narrator/policy choices:
set role_id=null, basis=unresolved, review_required=true, and list candidates.
For an ambiguous SOLO voice, leave role_id=null with unresolved basis and review
required; do not assign a chorus/group role as a substitute for identifying an
individual. Use a chorus role only when source evidence establishes collective
speech by that established anonymous group; known people speaking in unison use
simultaneous_role_ids with their existing voices. Never silently send
unidentified dialogue to the narrator.
Contextual attribution
may be assigned, but set review_required=true when evidence is weak or conflicting.
Do not invent a role absent from the final cast: leave it unresolved for review.
Use local delivery only when supported by the passage, with exact evidence;
otherwise direction is empty. Do not bake emotion into the persistent voice.
At render time, delivery.direction follows the voice's speechTraits. Describe
only what changes in this passage; do not restate or contradict the voice's
accent or habitual speech. Leave the character's age out of delivery: the voice
already carries it, and age words in a direction can make the engine refuse the
line.
Optionally add a sparse "proposals" object to a segment, only when its source
text clearly supports it. Proposals are suggestions: an independent reviewer
may remove any of them, and unsupported proposals are discarded.
"proposals":{"edits":[...],"controls":[...]} (either array may be omitted).
Each edit normalizes a stylized source spelling for speech without changing
meaning: {"original":"exact substring of this segment",
 "replacement":"plain spoken spelling","direction":"required plain delivery
 note for the normalized style, at most 240 characters",
 "reason":"brief required justification",
 "left":"optional immediate exact context before original","right":"optional
 immediate exact context after original"}. Only stretched letters, drawn-out
 words, and dialect respellings qualify; never fix grammar, punctuation,
 capitalization, or ordinary spelling, never delete words, and never restate
 the original. "left"/"right" must make original+left+right unique in the
 segment when the original alone is ambiguous.
Each control adds a local performance annotation, located like an edit:
 {"kind":"delivery","tag":"whisper","original":...,"reason":...,
  "direction":"optional plain direction; required nonempty for tag=style",
  "left":...,"right":...,"emotion":"optional short source-supported emotion"}
 {"kind":"pause","duration_ms":400,"position":"before","original":...,
  "reason":...,"left":...,"right":...}
 {"kind":"vocalization","sound":"laugh","position":"after","original":...,
  "reason":...,"left":...,"right":...}
delivery tags: style, whisper, emphasis, soft, fast, slow, stress, forceful.
Use tag=style and a brief natural-language direction for local emotions or a
combination such as angry whispering. Optional emotion on a control combines
with that control's delivery; optional segment emotion combines with its
delivery.direction and requires the same source evidence. Use one primary
emotion per sentence when supported; ordinary speech needs no emotion tag.
Standalone vocalizations must belong to this segment's actual performer;
do not make the narrator laugh merely because narration says a character laughed,
and do not double laughter already written as spoken words.
pause duration_ms is an integer from 100 through 5000; position is "before" or
"after" the located text. vocalization sounds: sigh, laugh, chuckle, breath,
gasp, sob, cough, throat_clear. All direction, replacement, and reason text is
plain words only: never square or angle brackets, never markup, and never
phonemes, pronunciation keys, or dictionary guidance. Keep proposals sparse;
omit the whole object when the passage needs none.
"""

ATTRIBUTE_COMPACT = ATTRIBUTE_COMPACT_V1 + WHISPER_DELIVERY_POLICY + """
COMPACT ATTRIBUTION WIRE CLARIFICATION v2
The prohibition on echoing source text means never copy a complete source unit
or segment and never include a segment "text" field. It does not prohibit the
exact source substrings required for start_anchor or evidence.quote.
Every segment.evidence and segment.delivery.evidence value is an array whose
entries are canonical evidence objects:
{"block_id":"existing source block ID","quote":"short exact nonempty substring
of that source block"}. Never use a bare quotation, string, tuple, list, or any
other non-object as an evidence entry. Use evidence=[] when no evidence is
supplied. Every nonempty delivery.direction requires a nonempty
delivery.evidence array containing at least one canonical evidence object that
supports that direction; do not use a bare quote there either.
Return EVERY target unit exactly once in input order. Every unit must have a
nonempty ordered segments array covering its complete source unit through the
anchor rules above, including the final segment and trailing source characters.
Never drop a target unit, return only corrected units, or omit source coverage.
These no-text, canonical-evidence, and complete-coverage rules apply equally to
initial responses and complete replacement responses after validation errors.
"""
