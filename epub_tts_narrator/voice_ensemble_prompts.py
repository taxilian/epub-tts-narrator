"""Whole-cast voice design request: the ensemble goal first, then the contract."""

from .prompts import COMMON, VOICE_PROFILE_GUIDANCE, VOICE_PROFILE_RECONCILIATION


ENSEMBLE_SYSTEM = COMMON + VOICE_PROFILE_GUIDANCE + VOICE_PROFILE_RECONCILIATION + """
GOAL: design the complete voice cast as one coordinated ensemble, now that the
cast is final. Discovery, identity consensus, and any series reconciliation
have already resolved characters, roles, ownership, and inherited profiles.
The payload supplies that fully resolved cast: every character, role, and voice
profile, with source evidence, plus deterministic contrast_findings captured
before this pass, naming pairs of distinct voices whose recognized structured
dimensions are identical or differ in fewer than two dimensions. Those findings
are advisory text analysis of declared profile records, never measurements of
synthesized audio; they do not prove confusability and never require recasting
when source evidence genuinely supports similar voices.
Compare every profile against the whole cast. Return complete voice upserts
only where a coordinated change helps, exactly as
{"voices":[complete voice records], "reasons":{"voice ID":"brief reason"}},
omitting unchanged voices; an empty voices array is a valid answer.
Obey these protections without exception:
- Reference only existing voice IDs. Never invent, remove, rename, merge, or
  reorder voices, and never change character or role ownership or identities.
- Never upsert a voice carrying a series_origin of kind series_casting_reuse.
  Established inherited profiles, including their descriptions, traits, and
  reference identity, are reference-only context for matching new work.
- Every upsert must provide exactly description, traits, and speechTraits;
  stored profile metadata beyond those three fields keeps its current values.
- Preserve every existing basis=text trait exactly, including its evidence,
  and never introduce a new basis=text trait: text support exists only where
  the established profile already carries it. This pass coordinates casting;
  it never discovers source evidence.
- You may coordinate casting-basis traits (no evidence entries), descriptions,
  and speechTraits of local voices, wholly replacing the previous values.
- Apply cue precedence: explicit vocal evidence, deliberate user choices, and
  established series profiles first; narrative characterization second; only
  when none of those settles an attribute may names, ethnicity, gender, or
  descriptions inform a tentative creative casting guess with basis=casting,
  evidence=[], and the tentative basis explained in the reason metadata.
- Resolve contrast findings by strengthening genuinely different audible
  casting choices consistent with evidence. Never invent traits, contradict
  source-supported similarity, or add stereotypes or caricatures.
- Keep every engine-facing direction self-contained, adult-performer framed,
  and free of analysis commentary, quotations, and markup.
"""
