# Analysis: how epub-tts-narrator reads a book

Analysis produces a validated, multi-voice script bundle whose source text and offsets remain traceable to the EPUB. It is independent of the TTS engine and contains no engine-specific voice names, SSML, audio parameters, or audio files. The pipeline is extract, discover candidates, resolve identities, check source coverage, design voices, attribute, review, complete, and auditions. The resulting directory can be rendered directly, or a legacy JSON analysis can be exported separately.

## Analysis stages

1. Extract canonical text from XHTML resources in EPUB spine order, preserving chapter membership and source references.
2. Discover evidence-backed characters and narrator observations in independently analyzed source regions. Start with the whole source and subdivide only when capacity requires it, keeping the original source units and offsets. Repeated names in different regions remain separate candidates, not automatically established identities.
3. Resolve split and incorrectly combined identities, ambiguous names, and continuity with prior books through the independent identity audit. Require three independent unanimous approvals before applying an identity change or confirming continuity.
4. Independently check source regions against the proposed cast for overlooked speakers. Process exact-source-anchored findings and recheck affected regions; repeated findings, cycles, or no progress remain explicit unresolved items instead of causing unlimited global passes. New identity proposals receive the same independent review before voice design.
5. Design the voice profiles as one coordinated ensemble over the final resolved cast, before any attribution. See the ensemble section below.
6. Attribute every exact text span with the resulting cast and nearby context. Compact source anchors identify speaker boundaries without echoing the source text. Sparse spoken-spelling edits and scoped delivery/emotion/vocalization proposals can accompany attribution; only those proposals receive a focused independent review. Dialogue and delivery may change inside paragraphs and sentences.
7. Optionally review the completed script for attribution mistakes. Apply a proposed correction only after three unanimous block-level reviews.
8. Find every remaining unassigned spoken passage and request three independent source-based assignments. Apply only unanimous resolved decisions, and retain genuine uncertainty.
9. Prepare an audition script for each voice from clearly attributed speech, joining statements when useful, or ask the model for character-specific casting text.
10. After automatic completion, present remaining choices in a terminal, then validate and save the analysis, cumulative series cast, and renderer bundle. Notes are advisory. Missing performers are not.

A character, narrative or dialogue role, voice profile, and local delivery are separate records. Distinct characters receive distinct profiles. A narrator role may reference a character and share its profile. Profiles separate source-supported traits from creative casting choices; casting proposals are not claims of source evidence. Fresh profiles describe a few compatible audible traits, while temporary emotion, urgency, whispering, shouting, and changes of pace belong to segment delivery. Returning series characters retain their profiles and stable voice IDs through the continuation chain.

Voice design also requests `speechTraits`: one or two short, mood-independent phrases that the renderer applies to every passage for that voice. State the accent explicitly and specifically, such as Glaswegian Scottish rather than Scottish. Where the source establishes no specific accent, choose one as a labeled production decision, considering the supplied cast; neutral General American is an option, not a universal default or factual inference. Describe audible behavior such as clipped consonants, a persistent lisp, or slow habitual phrasing; omit personality and emotions. This field is distinct from the reference-voice creation description and passage-specific delivery. Existing analyses without it remain compatible; they are not silently assigned a new accent. A continuation attempting to add the field only to an inherited profile fails with the defining book and voice ID until that original definition is updated. Existing traits may improve in later books and are carried into subsequent series context without changing earlier snapshots or recreating reference recordings.

Fresh analysis prepares auditions of 25 to 60 words, normally 35 to 50, with a 650-character maximum. Clearly attributed ordinary speech is preferred. Separate statements may be joined with exact source ranges retained. Resolved simultaneous dialogue can serve as a solo audition for each participant. Ambiguous attribution, extreme delivery, repeated cries, and engine control syntax are excluded. If usable speech is insufficient, the model writes original character-specific casting text. That text is marked as a casting choice, never as a quotation or audiobook script, and is saved in `provenance.voice_samples`. Failed auditions remain pending rather than receiving generic text; rerunning analysis or opening casting retries them.

New attribution requests carry a compact cast containing every character, role, and voice ID, identity descriptions, and aliases, but not repeated voice-creation profiles or accumulated cast quotations. The canonical cast and whole-cast ensemble pass remain complete.

Compact responses reference every target unit in order. The first segment begins at the unit start; subsequent boundaries use unique exact source anchors. The application reconstructs exact segment text and offsets locally. Ambiguous anchors, omitted units, reordered boundaries, and unknown IDs fail validation rather than being guessed. Legacy exact-text responses and their validated caches remain readable without changing the global analysis prompt version. Unknown speakers remain explicitly unassigned, never relabeled as the narrator.

Attribution may attach sparse, source-bound spoken edits for stylized spelling, preserving colloquial grammar, negation, repetitions, names, and intended pauses. It must not copy or paraphrase entire passages as a substitute for source alignment. Bounded repeated-letter edits receive mechanical checks; other edits and new scoped controls receive one independent remove-only review of the proposals and necessary local context, batched rather than replaying whole attribution batches. Rejection or uncertainty leaves the affected wording unchanged. A failed request remains resumable.

Delivery can vary within a single segment. Scoped whispering, emotions, and laughing-while-speaking are ranges; standalone vocalizations and pauses are positioned events. Instructions and short explanations are metadata, never extra spoken words or raw engine tags. A locally supported cue must not become a permanent voice trait or leak into the narrator's text. Source-to-spoken mappings project approved controls through edits and later segmentation changes; unsplittable edits and dependent controls are withheld rather than assigned invented offsets.

When identified characters say the same words together, a segment has a `simultaneous_speakers` list containing each existing role and voice. Its single `role_id` and `voice_id` are null, but the performance is resolved. Multiple candidates do not create a mixed performance. The renderer synthesizes and verifies one recording per performer, then mixes them at the same start time. Current support is unison of the same words, not overlapping speech with different words or staggered entrances.

An empty delivery object such as `{}`, `{"direction":""}`, or `{"evidence":[]}` is completed locally as `{"direction":"","evidence":[]}`. Only omitted empty defaults are supplied. Missing objects, invalid types, unknown partial fields, and nonempty directions without evidence still fail. The same rule applies to proposed semantic corrections before review.

One narrow recovery restores copied repeated-letter counts from the authoritative EPUB. The complete unit must have the same run-character sequence. Runs must be alphabetic, counts must be 2 to 32, at least one count must be 3 or more, the difference must be at most 8, and at most 16 runs may change per unit. Segment boundaries may not split a changed run. Restoration preserves canonical source and speaker assignments, records `source_restoration` metadata, and flags affected segments. It is not fuzzy spelling repair.

Fresh passage tagging uses the shared adaptive runner. Request-budget preflight, explicit provider context-capacity errors, and output truncation can subdivide a batch at whole target-unit boundaries, without a fixed depth or leaf ceiling. Each initial tree is bounded by its original source-unit count. Children retain source context and are revalidated together for exact coverage and order; unrelated failures never trigger subdivision. An indivisible request that cannot fit stops analysis rather than losing source or required cast records. Subdivision provenance records failed parents, validated children, fingerprints, and replayed capacity hints; existing legacy caches remain readable.

## Character identity consensus

Identity consensus is enabled by default with `analysis.identity_consensus = true`. After discovery and reconciliation, a source scan looks for both directions of error: distinct people combined as one character, and one returning person represented by several characters. Shared or similar names are grounds for investigation, never proof. The audit examines aliases, descriptions, relationships, and source context. A proposed split includes replacement character, role, and voice records before judgment.

Each scan treats supplied cast records and their stored evidence as fallible reference material. Requests use the complete cast when it fits, otherwise an explicitly disclosed scoped view whose included records remain complete. `source_scope` identifies supplied text as a `source_batch` or `complete_source`; a single scan covering the entire book is complete-source regardless of old configuration switches. Every returned evidence quotation must occur verbatim in the matching block's supplied `source_units` text; a genuine quotation stored in the cast is not admissible when that passage is outside this request. One valid local quotation cannot authorize another hidden quotation. A region may reveal a connection to a character introduced elsewhere, and a locally supported concern need not contain all evidence needed for final judgment. An empty scan result means no supported concern in the supplied text, not that the entire cast is certified correct. Missing characters or context alone do not warrant an `uncertain` case.

Batch findings are combined before judgment. Discovery-proposed merges and the independent name-collision screen remain additional candidate sources. Case preparation and all three blind reviewers retain broader source context, including cast-cited blocks and neighboring passages; scan citation scoping does not trim their evidence or weaken exact-quotation validation.

With prior series books, a dedicated continuity scan gives every current character an explicit `new`, `match`, or `uncertain` decision, even when discovery proposed no match. A match requires current-source evidence and the same three-way review. An uncertain decision remains reviewable. An empty current cast needs no request.

Each concrete decision receives three independent blind verdicts. Only three approvals establish consensus. Adaptive name-collision reviews may batch disjoint cases into each reviewer's request, but every case still needs all three reviewers' source-bound approvals. Source-discovered conflicts retain their existing guard; a rejected or uncertain collision cannot be circumvented through a later transitive merge. Matching waves advance approved components and unpaired candidates, retire unapproved pair participants, and never enumerate every similar-name pair. This bounds nomination cases by the number of candidates, without claiming exhaustive identity certification. Accepted changes occur before passage attribution, so workers share one corrected cast. Proposals, evidence, verdicts, constraints, and cast fingerprints remain in `provenance.identity_consensus`.

The casting provenance retains complete discovery and coverage candidate snapshots. After approved identity merges, newly superseded local voice proposals are archived rather than sent for unnecessary voice design or auditions. Their source-supported traits are carried into the canonical voice without rewriting evidence; conflicting trait values remain intact and create a review note. Unrelated unused proposals and established series profiles are not discarded.

Continuity has at most two rounds. A second scan reconsiders every current character when a first-round match lacks three valid approvals, or when an uncertain result has a prior-character candidate. Rejection, uncertainty, error, and unresolved candidates trigger it. A `new` result, or uncertainty without candidates, does not. The second round uses prior feedback only as untrusted leads, performs fresh blind reviews for every proposed match, and enables only final-round accepted matches. The process stops after round two, even if uncertainty remains.

A failed scan or provider request stops analysis without publishing success, while valid cached work remains resumable. Consensus reduces dependence on one answer but does not certify identity accuracy. Outstanding identity notes are advisory; genuinely unassigned speech still requires a performer. Set `identity_consensus = false` to restore discovery and reconciliation behavior without the independent audit, exhaustive continuity scan, or three-way verdicts. No environment variable or separate CLI flag controls it.

Scan prompt and payload changes produce new keys for affected requests without changing the global analysis prompt version or deleting old cache entries. Keep the cache and rerun the same command and configuration to resume an interrupted `analyzed` stage; unchanged requests remain reusable, while affected scans run again. A running process does not hot-reload prompts, and an already-checkpointed `analyzed` stage or published book is not automatically re-audited by this prompt update.

## Whole-cast ensemble design and contrast audit

After identity consensus, source-backed coverage repair, and series reconciliation finalize the cast, and before any attribution, voice profiles are designed as a coordinated ensemble. The whole cast is supplied when it fits; capacity failures split the target voices into automatically sized groups. Each designable voice is targeted exactly once, with character and role context, source evidence, and current `speechTraits`. Identity splits and finalized inherited voice ownership are already applied, so attribution, auditions, and saved fingerprints all see one coherent cast.

Target profiles, traits, and role mappings are never trimmed to fit a group request. Voice-trait evidence is sent in full so an upsert can echo established text-supported traits exactly; character evidence uses the usual bounded exact quotations. Group requests disclose the full cast fingerprint, target IDs, included records and omitted counts; cross-group contrast findings remain read-only advisory context. Upserts cannot edit voices outside the group's targets. If one indivisible target and its mandatory evidence cannot fit, analysis stops clearly rather than dropping that evidence. A cast with fewer than two distinct voices, or with nothing but established series profiles, makes no request; provenance records the skip with unchanged fingerprints, zero request characters, and no updates.

Before subdividing an oversized request, the ensemble pass attempts a lossless encoding of its contrast findings: each voice ID appears in a lookup table, and comparisons with identical kinds and dimension lists share a group of indexed pairs. Every original pair and comparison detail remains present, and all cast records and source evidence are unchanged. The size limit includes decoding instructions. Requests already within budget retain their existing prompt, payload, and cache keys; compact and grouped requests have their own exact prompt/payload keys. Full, uncompressed before/after findings and actual target-group coverage remain in provenance. This does not raise provider limits, output budgets, or weaken acceptance rules.

Only complete upserts of existing local voices are accepted — each carries exactly `description`, `traits`, and `speechTraits` and a brief reason; stored profile metadata beyond those fields keeps its current values. The pass can coordinate casting-basis traits, descriptions, and `speechTraits`; it can never invent, remove, rename, or reorder voices, change character or role ownership, or touch a series-inherited profile: inherited descriptions, traits, and reference identity are reference-only context. Existing text-supported traits must be echoed exactly and may not be changed for contrast, and the pass never claims new text-supported traits — it coordinates casting, it does not discover evidence. The response is fully validated, including a rebuilt-cast check, before any successful cache entry is written, so a cached replay revalidates the complete cast. Changes, reasons, before/after cast fingerprints, and the guidance fingerprint are kept in `provenance.voice_ensemble`. Validated responses are cached and replayed like every other stage.

A deterministic contrast audit always accompanies the pass. It compares recognized audible dimensions declared in each profile's traits — with a few justified spelling and synonym normalizations — between distinct canonical voices only; aliases of one voice never pair. A dimension missing from either profile is unknown, never a difference, and no traits are invented. Pairs with identical known profiles, or fewer than two demonstrably different known dimensions, become advisory potential-confusion findings carrying voice IDs and dimension names. The before-design findings travel with the ensemble request; provenance keeps both the before-design findings and the findings recomputed on the final cast, so a collision the pass already resolved is not reported as unresolved. This is text analysis of profile records: it is not a measurement of synthesized audio, does not prove two voices are confusable, never blocks rendering, and never demands recasting where source evidence genuinely supports similar voices.

Vocal attributes follow a cue precedence shared by analysis and every refresh path: explicit vocal evidence, the user's deliberate choices, and established series profiles come first; specific narrative characterization and story context come second; only when none of those settles an attribute may names, ethnicity, gender, or descriptions inform a tentative creative casting guess. Such a guess is labeled `basis=casting` with empty evidence, its tentative basis is explained in reason/audit metadata rather than in engine-facing prose, and it is never a factual acoustic claim, a universal stereotype, an exaggerated caricature, or a fixed demographic-to-accent mapping. Gender and ethnicity are not acoustic facts.

Accent guidance considers upbringing, languages used, age at migration, and time in the current setting. Fluent English or the absence of an explicit accent label does not default a character to General American. A plausible Japanese or Mexican Spanish influence for a student who moved during adolescence is a creative casting decision; a similarly named person raised in Ohio can instead have a local American accent. Background evidence does not become evidence of an unmentioned audible accent. Local automatic defaults are revisable, while explicit source accents, deliberate user corrections, and inherited voices take precedence. Profile updates keep description, accent traits, and `speechTraits` consistent; a description-only refresh reports any remaining mismatch in its audit reason. The initial narrator seed leaves accent unspecified.

For a small manual model evaluation, `scripts/validate_voice_accents.py --config CONFIG.toml -o /tmp/accent-evaluation.json` exercises original synthetic contexts and records validated proposals plus review criteria. `--stage discovery`, `ensemble`, `profile`, or `prompt` selects a production prompt path; the default is `profile`. `--check-fixtures` is offline. These results test casting choices, not the acoustic quality of synthesized accents, and never modify book bundles.

The casting-description refresh follows the same cue precedence as analysis and carries one consistent `known_cast` snapshot plus that voice's contrast findings.

## Passage attribution consensus

Set `analysis.attribution_review = true` to review the completed script during `analyze`. It is disabled by default because it adds model calls. The review checks narration, speech separated by tags or actions, continuing and nested quotations, direct thoughts, narrator frames, and individual versus collective speech. It uses the existing cast and performance policy. It cannot add identities, change voices, or rewrite source text.

The scanner proposes complete replacements for affected source blocks. Three blind, separately cached reviewers inspect each proposal against source. A block needs three approvals and exact citations. Rejection, uncertainty, or invalid output leaves the original assignment and adds an unresolved item. Confirmed simultaneous assignment can resolve its corresponding unassigned flag. Human corrections cannot be overwritten. Bracketed delivery is reapplied after accepted changes. Source, cast, segments, verdicts, and fingerprints remain in `provenance.attribution_review`.

Every run performs mandatory speaker completion after this optional review, even when attribution review is disabled. It enumerates all unassigned spoken spans, assigns their surrounding complete source blocks through three independent reviewers, and applies only unanimous resolved decisions. Up to three rounds widen context, with the last adding relevant cast evidence from other chapters. Exact coverage, cast references, quotations, and protected human assignments are validated. Completion cannot invent a character or silently use the narrator. Decisions and remaining counts are retained in `provenance.speaker_completion` and resumable cache audits.

Remaining passages are shown separately with wrapped, paragraph-spaced context and assigned voice labels. The interactive menu offers up to three likely speakers, **Enter** accepts the first, a number selects one, **O** opens the full cast, **B** returns, and **Q** or end of input saves and exits with status `2`. Suggestions do not change assignments until selected. Unassigned speech blocks audio generation; advisory notes do not. JEV, when enabled, ranks plausible candidates but does not authorize assignment.

Rerun `analyze` with the same output directory to resume. Cached validated responses are reused, unresolved blocks receive a fresh bounded review, saved identities and unaffected assignments remain, and unresolved passages are never filled by guessing. Interactive speaker decisions are journaled under the request cache (`speaker-decisions/`) as each is made and replayed on a rerun that reaches the same pre-prompt state, including any voices defined during the prompt; a changed book or cast yields a different journal key and asks fresh. An in-progress analysis keeps a `.analysis-pending/` workspace in the output directory and checkpoints the document after each stage, so a rerun resumes from the latest completed stage without repeating its model calls or local work; the directory never counts as a completed bundle until publish finalizes it. Source must match the saved book. Conservative narration-boundary repair splits clear tags such as `"Hello," Mara said, "Again."` into character, narrator, character spans while preserving source text. Ambiguous scopes go through completion. Changes are recorded in `provenance.narration_boundaries`.

To resolve existing directories, use `python3.11 -m epub_tts_narrator resolve-speakers ...`. To review an analysis without rediscovery, use `review-attribution INPUT -o OUTPUT`. Review batches retain complete source blocks and chapter boundaries. Truncated semantic scans subdivide up to two levels; jury requests are not subdivided. Invalid scans or provider failures publish no output, while valid cached calls remain resumable.

## Optional corrections

A voice-profile refresh marks every selected profile invalid in `.voice-profile-refresh.json` before the first model request. Each validated proposal is durably checkpointed as prepared, not yet valid for generation. If a model call or publication fails, rerun the same refresh command: matching proposals are revalidated and reused without another model call, independently of the request cache. Changed inputs, guidance, or provider settings invalidate stale proposals. Rendering and casting are blocked while this book or an ancestor has a pending refresh. Successful publication clears the pending state; if clearing fails after publication, rerun reconciles the published fingerprints instead of repeating model work. Reference recordings are not invalidated or deleted. Stable `.lease` ownership files must not be deleted.

To update only voice profiles in an existing book, run `python3.11 -m epub_tts_narrator refresh-voice-profiles BOOK_DIRECTORY --workers 1`. This refreshes voice descriptions, casting traits, and `speechTraits` without reanalyzing source, identities, assignments, or spelling edits. Existing evidence-backed traits, audition text, reference recordings, and accepted audio files are retained. Changed performance traits require new affected speech during the next render, not new reference voices. Original analysis files and audition bindings are backed up in `.analysis-revisions/`; profile publication validates and refreshes descendant continuation links atomically. Process a series one book at a time, starting with the defining book. Parent profiles are supplied to later refreshes so their established accents are not independently reinvented.

When a parent profile declares an `accent` or `dialect` trait, automatic refresh must preserve that complete trait value and include its wording in both the description and `speechTraits`. Validation applies to fresh responses, cached responses, and resumed proposals; compatible pace, rhythm, or other delivery refinements remain possible. This checks declared continuity, not the acoustics or semantic consistency of every sentence. Background alone cannot authorize changing an inherited accent. Use an explicit casting/performance edit or a source-supported analysis correction for an intentional change.

`review` prints source and cast fingerprints, unresolved issues, and flagged segments. Copy both fingerprints into a corrections file:

```json
{
  "source_fingerprint": "COPY_FROM_REVIEW",
  "cast_fingerprint": "COPY_FROM_REVIEW",
  "assignments": [
    {"segment_id": "b000006:0:19", "role_id": "mara_dialogue"}
  ],
  "resolved_issues": []
}
```

Use IDs from the analysis, then apply and validate:

```sh
python3.11 -m epub_tts_narrator apply runs/book/analysis.json \
  --edits corrections.json -o book.reviewed.json
python3.11 -m epub_tts_narrator validate book.reviewed.json --ready
python3.11 -m epub_tts_narrator export book.reviewed.json -o book-bundle
```

Corrections are saved in provenance and leave the original intact unless `--force` is used. A file may also contain a complete replacement cast and exact per-block segment replacements. `analyze --edits corrections.json` replays them only when both source and cast fingerprints match. Stale corrections are rejected, never silently applied to a different identity. See [FORMAT.md](FORMAT.md) for the complete corrections contract.

## Performance policies

Defaults use a known thinking character for explicit thoughts, a known letter author for letters, the current speaking character for nested quotations, and the relevant narrator for headings. Unclear cases require review. Override selected strings in the configuration file; omitted strings retain defaults:

```toml
[performance_policy]
thoughts = "Use the relevant narrator for internal thoughts."
letters = "Read letters using the author's character voice when identified; otherwise flag for review."
nested_quotations = "Keep the currently speaking character's performed voice."
headings_and_nonstory_text = "Use the applicable narrator."
```

Policies guide annotation. They do not alter source wording or bypass validation.

## TTS handoff

Directory analysis creates a version 2 renderer bundle with a manifest, cast, ordered chapter scripts, canonical source, review data, full analysis, and cumulative series cast. The manifest supplies script order, source ranges, readiness, provenance, and hashes. The analyzer emits provider-independent voice references, not engine voice names or audio. See [FORMAT.md](FORMAT.md) for the bundle contract.

## Long books, resuming, and cost controls

Fresh analysis uses one automatic casting strategy, not a whole-book versus tiny-discovery-batch switch. The largest source region is attempted when it fits the serialized request budget. Oversized requests are bisected over existing source units; explicit provider context-limit errors and output truncation can also trigger subdivision. Generic timeouts, authentication, quota, transport, malformed output, and schema failures do not cause recursive fan-out. A single indivisible request that cannot fit stops clearly instead of deleting source or evidence.

For `N` original source units, a binary source partition tree contains at most `2N-1` nodes. Preflight-only split nodes make no provider call. There is no fixed depth, partition count, or book-level request quota: larger books receive proportionally more work without operator tuning. Output capacity is observed separately from input character size; character counts are not token estimates. Successful response caches and exact-request split hints let an interrupted run reuse validated work.

Initial discovery does not carry an ever-growing cast from one source region into the next. Local candidate IDs are namespaced by source region, and exact visible source quotations are checked before acceptance. Identity scans and later source checks may use explicit scoped cast views: included records remain complete, omitted membership is disclosed, and the full canonical cast remains intact locally. Scoped views are not permission to merge by name or invent an omitted identity.

Coverage repair follows distinct exact source anchors. A finding is processed once, and only affected source regions are rechecked after a repair; repeated findings or unchanged state terminate as unresolved review items. This proves scheduled source coverage, not that a model noticed every speaker or achieved perfect semantic understanding. The old `whole_book_discovery` and `cast_reconciliation_rounds` settings remain parseable for older configuration files but do not select a separate fresh-analysis algorithm or force repeated global passes.

`plan` reports a baseline estimate, source-derived partition bounds, and conditional work. Identity cases add three verdict requests per case and preparation requests for proposed splits. Prior series books add continuity checks; failed or uncertain matches can trigger the existing bounded second round. Voice-target grouping, coverage findings, speaker completion, repairs, retries, and auditions add data-dependent work. The ensemble request can be skipped when nothing needs design. Pronunciation preparation adds none.

Attribution review adds one scan per review batch and three jury requests for each batch with proposals. Semantic truncation can add bounded subdivision attempts. Initial passage tagging can likewise subdivide a truncated batch. Mandatory speaker completion adds three requests per unassigned source block in each required round, with at most `min(5, attribution_workers)` concurrent requests. Audition preparation adds one request per voice lacking usable source speech and a saved creative script.

`analysis.attribution_workers` accepts 1 to 8 and is the shared concurrency ceiling. Identity review uses at most `min(3, attribution_workers)` concurrent calls, passage review at most `min(5, attribution_workers)`, and these phases do not overlap. Defaults are 12,000 source characters per attribution batch, 200,000 serialized request characters, and one validation repair. Characters are not tokens, and cast metadata consumes context. `analysis.chunk_chars`, `analysis.max_request_chars`, `analysis.repair_attempts`, and `provider.max_output_tokens` bound request size and recovery.

Request retries are disabled by default. `provider.max_retries` enables bounded retries for transient HTTP and typed transport failures. Known quota, billing, authentication, TLS, invalid URL, permanent DNS, unrecognized transport, and invalid-output failures are not retried. Logs omit raw errors, URLs, response bodies, keys, and TLS diagnostics.

Validated responses are cached under `.epub-audiobook-cache` beside the output, or under `analysis.cache_dir`. Cache keys include exact prompts, inputs, model, endpoint, and relevant options. Changing request inputs invalidates affected work. Cache entries contain book text and derived annotations, never API keys. Keep the cache and rerun the same command after interruption. A failed analysis does not publish a completed directory. Failed responses remain in `failed/` for bounded repair and are never accepted as successful results.

Resuming the same book needs no flag. `--continue-from` selects prior series context and must remain the same when resuming. Changing timeout, key, or retry settings does not invalidate completed response caches; changing model, prompts, cast, policy, or other request inputs can. Advisory notes do not make a bundle a draft, but missing speakers prevent rendering.

## Optional pronunciation work and targeted attribution edits

There is no independent spoken-spelling scan, pronunciation candidate discovery, CMU suggestion, or name-occurrence classification during analysis. `discover-pronunciations` and `prepare-name-script` remain retired. Targeted, reviewed spoken edits can instead accompany the attribution call that already reads a passage; this does not create a pronunciation registry or a name script.

General pronunciation changes remain optional explicit edits after analysis. A later per-word name/JEV workflow is not implemented. Published books are not automatically reanalyzed or re-prepared on resume. Existing choices, name sidecars, and saved spelling edits remain readable; formats are documented in [FORMAT.md](FORMAT.md#spoken-spelling-normalization). Historical spelling checkpoint labels remain inert so interrupted analyses can resume without restoring the retired scan.

## Bracketed delivery

Square-bracket text such as `[Fireball]` receives a separate delivery cue, including inside dialogue. Speaker and voice stay unchanged. The script retains exact source text and supplies `spoken_text` without delimiters for synthesis. Configure `performance_policy.bracketed_text`; balanced nested and multi-paragraph groups are supported, while unmatched delimiters are flagged. See [the format contract](FORMAT.md#bracketed-delivery).

## Reusing a series cast

Directory analysis saves a cumulative series cast. Continue a series with `--continue-from`; the prior directory stays unchanged and the new output receives its history and a relative continuation link.

```sh
python3.11 -m epub_tts_narrator analyze book-one.epub -o runs/book-01
python3.11 -m epub_tts_narrator analyze book-two.epub \
  -o runs/book-02 --continue-from runs/book-01
python3.11 -m epub_tts_narrator analyze book-three.epub \
  -o runs/book-03 --continue-from runs/book-02
python3.11 -m epub_tts_narrator plan book-four.epub \
  -o runs/book-04 --continue-from runs/book-03
```

Voice lookup searches the current book, then continuation parents. A regenerated voice belongs to the book where it was generated and is inherited by descendants. Missing voices, casting edits, and replacements are written only to the current book. Keep linked directories together.

Author afterwords and explicit “From the Author” sections have a separate author narration identity. An unambiguous matching author can inherit the nearest prior author voice, but this never merges the author with a fictional narrator, guest, or editor. Explicit aliases and existing local recordings take precedence. Tentative gender presentation for a new author voice is casting guidance, not a claim about the author.

Manual sharing is book-specific: `audiobook-render voice-aliases BOOK --set SOURCE_VOICE_ID TARGET_VOICE_ID` writes `voice-aliases.json`, and later books inherit the nearest mapping. `--remove SOURCE_VOICE_ID` removes a local override. Global configuration must not contain `voices.aliases`.

The input directory must contain the `series-cast.json` produced by directory analysis. Directory mode rejects `--series` and `analysis.series_registry`; prior context is selected explicitly with `--continue-from`. Identity consensus requires current-source evidence and three approvals for continuity. Names and aliases find candidates but do not establish identity. Without consensus, matching remains restricted to the sole prior character sharing an exact normalized name or alias, with ambiguous matches reviewable.

### Legacy JSON workflow

For compatibility, `analyze -o book.analysis.json` writes one analysis file. It accepts `--series` or deprecated `analysis.series_registry`, then requires `export` for renderer input. The old registry workflow is:

```sh
python3.11 -m epub_tts_narrator series-import book-one-analysis.json \
  --series casts/my-series.json -o book-one-linked.json
python3.11 -m epub_tts_narrator analyze book-two.epub \
  --series casts/my-series.json -o book-two-analysis.json
```

`series-import` preserves the original analysis and does no fresh identity scan or three-way review. Legacy analysis can update the selected registry atomically after successful analysis. New directory runs do not use `series-import`.

## Extraction scope

EPUB reading order follows the package spine. XHTML is parsed without scripts or network fetches. EPUB 2 and EPUB 3 package and spine structures are supported when their content is parseable XHTML.

- Canonical text collapses display whitespace while retaining inline text and punctuation. Offsets refer to canonical text, not XML bytes.
- Optional `inline_spans` preserve emphasis cues from semantic tags, explicit inline styles, and simple local stylesheet classes. Complex CSS is not inferred. Empty paragraphs mark the next block with `break_before: "blank_paragraph"`.
- Navigation and non-linear spine items are excluded by default and recorded. `extraction.include_auxiliary = true` includes them when present in the spine. Other non-spine content is audited, not appended automatically.
- Footnotes inside selected documents remain in DOM order. Separate footnote documents require deliberate inclusion. Notes are not relocated to references.
- Fixed-layout order, CSS-hidden text, visual poetry, images, SVG, MathML, OCR, and renderer-level pronunciation normalization are not handled as a renderer would. Inspect warnings and exclusions.
- Primary resources without usable XHTML fallback, encrypted content, malformed XML, unsafe archives, and empty extraction fail explicitly. No DRM removal is attempted.
- Chapter entries correspond to selected spine resources. A chapter spanning files or several chapters in one file needs later structural refinement.

These extraction limits are separate from speaker-attribution uncertainty. Review extraction before spending model calls on unusual publications.

## Further reading

- [FORMAT.md](FORMAT.md), bundle and script contract
- [RENDERER.md](RENDERER.md), local rendering and casting
- [REMOTE_RENDERING.md](REMOTE_RENDERING.md), remote rendering
