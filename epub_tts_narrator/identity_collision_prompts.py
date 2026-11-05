"""Adaptive identity-collision jury instructions for batched blind reviews."""

from .prompts import COMMON


VOTE_BATCH = COMMON + """
Independently verify a batch of concrete identity-merge hypotheses. You are one
of three blind reviewers. You cannot see other reviews. Your reviewer_id/lens
changes emphasis, but EVERY reviewer must check EVERY case in this batch. The
cases array lists independent merge hypotheses between two cast records each.

Return EXACTLY {"votes":[{"case_id":"the supplied case_id",
 "verdict":"approve|reject|uncertain",
 "reason":"brief source-bound reason",
 "evidence":[{"block_id":"ID","quote":"exact visible text from THAT case's source_units"}]}]}.
Return exactly one vote for EVERY supplied case_id, in any order. Omitting a
supplied case, duplicating a case_id, or inventing an unknown case_id is
invalid, and any malformed entry invalidates the whole batch response.

Each case supplies: character_ids (the two records), that case's source_units,
an original_cast pair view with its cast_scope disclosure, the proposed merge
patch, and a proposed_cast pair view with its proposed_cast_scope disclosure.
proposed_cast_scope.full_cast_fingerprint identifies the complete proposed cast
that was not supplied; judge only the supplied records and never assume
anything about omitted records. source_scope=pair_evidence_and_chapter_neighbors
means the request shows the pair's cited evidence blocks and their same-chapter
neighbors, not the complete book.

For each case, approve only when its supplied source establishes that the two
records are the SAME person beyond a shared or similar name. Names or spelling
resemblance alone NEVER prove identity. Look for coexisting people,
incompatible relationships, and distinct actions that refute the merge. Reject
if the source contradicts the merge; use uncertain when the supplied context
cannot decide. Missing context is not proof of separation, and silence about a
pair is not evidence either way. Every verdict needs exact visible source
evidence from that case's own source_units; an approval additionally needs
narrative evidence beyond names or aliases alone. Judge each case separately:
one weak or malformed case must not change another case's verdict. Schema
validity and the proposer's confidence prove nothing.
"""
