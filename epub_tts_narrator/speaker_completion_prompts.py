"""Speaker completion jury request instructions and reviewer lenses."""

from .prompts import COMMON, PERFORMANCE_POLICY_GUIDANCE


REVIEWERS = (
    ("speaker_completion_1", "Trace explicit speech tags and the speaker of continuing quotations."),
    ("speaker_completion_2", "Distinguish the actual speaker from mentioned people, listeners, and narrative actions."),
    ("speaker_completion_3", "Check alternative identities, narrative boundaries, and solo versus genuinely simultaneous delivery."),
)

SYSTEM = COMMON + PERFORMANCE_POLICY_GUIDANCE + """
Mandatory independent speaker completion, version 2.
Every target block contains text without a selected performer. Independently
assign the COMPLETE target block using the source and existing final_cast. No peer
answers are supplied. Existing assignments, character descriptions, and candidates
are fallible context, subject to the protected assignments described below.
An unassigned passage is not necessarily dialogue or thought: reconsider its kind
from the source before selecting a performer. Ordinary first-person narration,
reported actions, and narrative responses belong to the narrative frame even
when near dialogue. Examine exact wording, surrounding actions, quotation marks,
and available inline_spans (such as emphasis) together; typography is evidence,
not a conclusive rule on its own. Trace who actually communicates words rather
than choosing the nearest name or alternating speakers. A listener, addressee,
or person mentioned is not thereby the speaker.
An existing unassigned segment may incorrectly combine several speakers and
narration. Split it wherever the source changes performer or kind; old unassigned
boundaries are not constraints. Separate speech tags/actions from communicated
words and distinguish successive turns from genuinely simultaneous speech.
Use only existing role IDs. Never invent a character, role, voice, or source fact.
Preserve all original local delivery directions, including empty directions.
Preserve every original segment whose basis is user, including its performer,
kind, text boundaries, and delivery. You cannot override a user's assignment.
When retry_unresolved is true, ALL original segments with assigned performers
are protected, not only user assignments. Preserve their performer, kind, source
boundaries, and delivery; reconsider only the remaining unassigned source spans.
On these retries, unassigned original records intentionally omit previous guessed
kinds, candidate speakers, and explanations. Their state is unassigned and their
role_id is null; only exact source spans and existing delivery are supplied.
Determine their kind and performer independently from the source. Missing prior
labels are not evidence that somebody speaks: ordinary narration is a possible
answer. Assigned original records retain their full attribution for protection.
Only select multiple performers when exact source evidence establishes that these
identified characters say the SAME words together. Represent that as
simultaneous_role_ids with at least two distinct existing dialogue roles and
role_id=null, basis=explicit or context, candidates=[], review_required=false.
Alternative possible speakers are candidates, NEVER simultaneous performers.
A plural attribution covering several separate quotations does not establish
that each quotation was spoken in unison. Adjacent calls may be sequential with
unknown order; leave their performers unresolved unless evidence establishes
which person speaks which call, or actually establishes simultaneous same words.
Do not invent a representative narrator or group performer for an unknown crowd.
If the cast lacks the actual speaker, or the visible source is still insufficient,
leave role_id=null, basis=unresolved, review_required=true with plausible known
roles in candidates. Do not guess merely to complete the audiobook.
Return exactly {"block_id":"target block ID", "reason":"brief source-grounded
explanation", "evidence":[{"block_id":"visible ID","quote":"exact source quote"}],
"segments":[{"text":"exact contiguous source substring", "kind":"narration|dialogue|thought|letter|heading|other",
"role_id":"existing role ID or null", "basis":"explicit|context|unresolved",
"reason":"brief attribution explanation", "evidence":[], "candidates":[],
"review_required":false, "delivery":{"direction":"","evidence":[]}}]}.
The optional simultaneous_role_ids field is allowed ONLY on a simultaneous piece.
Return exactly one complete block, not just its unassigned span. Segment texts
must concatenate VERBATIM to the entire target text, including whitespace,
punctuation, brackets, and spelling. Each resolved spoken segment needs exact
source evidence for its performer; the top-level evidence must include a quote
from the target block. Evidence may use only visible target/context source units.
Nonempty delivery directions require exact supporting evidence. Return no other
properties, no user basis, no review of other blocks, and no private reasoning.
"""

FINAL_CONTEXT_GUIDANCE = """
Final independent source-context review. Additional context may include earlier
or later chapters containing exact source evidence for several candidate
identities. These are retrieval leads, not an answer or a change to the target.
Evaluate every plausible identity against source evidence, including identities
of internal, telepathic, or otherwise indirectly named speakers. Trace plural
speech tags across interrupted quotations and subsequent actions. The same
first-person words can be spoken in unison. A plural speech attribution can
continue across a narrative interruption unless source evidence establishes an
actual speaker change; singular grammar alone does not establish that change.
Do not infer unison from uncertainty, and leave genuine ambiguity unresolved.
"""
