# Analysis: how epub-tts-narrator reads a book

Analysis produces a validated, multi-voice script bundle whose source text and offsets remain traceable to the EPUB. It is independent of the TTS engine and contains no engine-specific voice names, SSML, audio parameters, or audio files. The pipeline is extract, discover, reconcile, identity, ensemble, attribute, review, complete, and auditions. The resulting directory can be rendered directly, or a legacy JSON analysis can be exported separately.

## Analysis stages

1. Extract canonical text from XHTML resources in EPUB spine order, preserving chapter membership and source references.
2. Discover characters, aliases, narrative roles, and proposed voices in bounded batches while retaining the accumulated cast.
3. Reconcile the book-wide cast using accumulated evidence and scene summaries. Additional passes continue from the preceding accepted cast. Proposed identity merges wait for the identity audit.
4. Scan for split and incorrectly combined identities, ambiguous names, and, when applicable, continuity with prior books. Require three independent unanimous approvals before applying an identity change or confirming continuity.
5. Design the voice profiles as one coordinated ensemble over the final resolved cast, before any attribution. See the ensemble section below.
6. Attribute every exact text span with the resulting cast and nearby context. Dialogue may split inside paragraphs and sentences. Character narration may share, or differ from, the dialogue voice.
7. Optionally review the completed script for attribution mistakes. Apply a proposed correction only after three unanimous block-level reviews.
8. Find every remaining unassigned spoken passage and request three independent source-based assignments. Apply only unanimous resolved decisions, and retain genuine uncertainty.
9. Prepare an audition script for each voice from clearly attributed speech, joining statements when useful, or ask the model for character-specific casting text.
10. After automatic completion, present remaining choices in a terminal, then validate and save the analysis, cumulative series cast, and renderer bundle. Notes are advisory. Missing performers are not.

A character, narrative or dialogue role, voice profile, and local delivery are separate records. Distinct characters receive distinct profiles. A narrator role may reference a character and share its profile. Profiles separate source-supported traits from creative casting choices; casting proposals are not claims of source evidence. Fresh profiles describe a few compatible audible traits, while temporary emotion, urgency, whispering, shouting, and changes of pace belong to segment delivery. Returning series characters retain their profiles and stable voice IDs through the continuation chain.

Voice design also requests `speechTraits`: one or two short, mood-independent phrases that the renderer applies to every passage for that voice. State the accent explicitly and specifically, such as Glaswegian Scottish rather than Scottish. Where the source establishes no specific accent, choose one as a labeled production decision, considering the supplied cast; neutral General American is an option, not a universal default or factual inference. Describe audible behavior such as clipped consonants, a persistent lisp, or slow habitual phrasing; omit personality and emotions. This field is distinct from the reference-voice creation description and passage-specific delivery. Existing analyses without it remain compatible; they are not silently assigned a new accent. A continuation attempting to add the field only to an inherited profile fails with the defining book and voice ID until that original definition is updated. Existing traits may improve in later books and are carried into subsequent series context without changing earlier snapshots or recreating reference recordings.

Fresh analysis prepares auditions of 25 to 60 words, normally 35 to 50, with a 650-character maximum. Clearly attributed ordinary speech is preferred. Separate statements may be joined with exact source ranges retained. Resolved simultaneous dialogue can serve as a solo audition for each participant. Ambiguous attribution, extreme delivery, repeated cries, and engine control syntax are excluded. If usable speech is insufficient, the model writes original character-specific casting text. That text is marked as a casting choice, never as a quotation or audiobook script, and is saved in `provenance.voice_samples`. Failed auditions remain pending rather than receiving generic text; rerunning analysis or opening casting retries them.

The model returns exact text fragments, not offsets. The application aligns them against immutable canonical source and rejects omissions, reordering, duplication, and rewritten words. Unknown speakers retain null role and voice IDs and remain reviewable. They are never relabeled as the narrator.

When identified characters say the same words together, a segment has a `simultaneous_speakers` list containing each existing role and voice. Its single `role_id` and `voice_id` are null, but the performance is resolved. Multiple candidates do not create a mixed performance. The renderer synthesizes and verifies one recording per performer, then mixes them at the same start time. Current support is unison of the same words, not overlapping speech with different words or staggered entrances.

An empty delivery object such as `{}`, `{"direction":""}`, or `{"evidence":[]}` is completed locally as `{"direction":"","evidence":[]}`. Only omitted empty defaults are supplied. Missing objects, invalid types, unknown partial fields, and nonempty directions without evidence still fail. The same rule applies to proposed semantic corrections before review.

One narrow recovery restores copied repeated-letter counts from the authoritative EPUB. The complete unit must have the same run-character sequence. Runs must be alphabetic, counts must be 2 to 32, at least one count must be 3 or more, the difference must be at most 8, and at most 16 runs may change per unit. Segment boundaries may not split a changed run. Restoration preserves canonical source and speaker assignments, records `source_restoration` metadata, and flags affected segments. It is not fuzzy spelling repair.

If initial passage tagging receives an explicit output-truncation error, the batch may be divided into halves, up to two split levels and four final groups. Splits use whole target units. Children retain original neighboring source and read-only context, run sequentially in the existing worker, and are revalidated together for exact coverage and order. Other failures do not trigger subdivision. A single-unit request or a request at the depth limit stops analysis. Subdivision provenance records failed parents, validated children, fingerprints, and replayed truncation hints.

## Character identity consensus

Identity consensus is enabled by default with `analysis.identity_consensus = true`. After discovery and reconciliation, a source scan looks for both directions of error: distinct people combined as one character, and one returning person represented by several characters. Shared or similar names are grounds for investigation, never proof. The audit examines aliases, descriptions, relationships, and source context. A proposed split includes replacement character, role, and voice records before judgment.

With prior series books, a dedicated continuity scan gives every current character an explicit `new`, `match`, or `uncertain` decision, even when discovery proposed no match. A match requires current-source evidence and the same three-way review. An uncertain decision remains reviewable. An empty current cast needs no request.

Each concrete decision receives three separately cached, blind verdicts. Only three approvals establish consensus. Disagreement, uncertainty, insufficient evidence, invalid quotations, or invalid references leaves the case unresolved and withholds the change. Accepted changes occur before passage attribution, so workers share one corrected cast. Proposals, evidence, verdicts, and cast fingerprints remain in `provenance.identity_consensus`.

Continuity has at most two rounds. A second scan reconsiders every current character when a first-round match lacks three valid approvals, or when an uncertain result has a prior-character candidate. Rejection, uncertainty, error, and unresolved candidates trigger it. A `new` result, or uncertainty without candidates, does not. The second round uses prior feedback only as untrusted leads, performs fresh blind reviews for every proposed match, and enables only final-round accepted matches. The process stops after round two, even if uncertainty remains.

A failed scan or provider request stops analysis without publishing success, while valid cached work remains resumable. Consensus reduces dependence on one answer but does not certify identity accuracy. Outstanding identity notes are advisory; genuinely unassigned speech still requires a performer. Set `identity_consensus = false` to restore discovery and reconciliation behavior without the independent audit, exhaustive continuity scan, or three-way verdicts. No environment variable or separate CLI flag controls it.

## Whole-cast ensemble design and contrast audit

After identity consensus and series reconciliation finalize the cast, and before any attribution, one whole-cast request designs the voice profiles as a coordinated ensemble. Every profile is visible with its character and role context, bounded source evidence, and the current `speechTraits`; identity splits and finalized inherited voice ownership are already applied, so attribution, auditions, and saved fingerprints all see one coherent cast.

The request carries the complete cast: records, traits, and role mappings are never dropped or summarized to fit the request budget. Voice-trait evidence is sent in full so an upsert can echo established text-supported traits exactly; character evidence uses the usual bounded exact quotations. If the request would exceed `analysis.max_request_chars`, analysis fails before making any model call and names the setting to raise. A cast with fewer than two distinct voices, or with nothing but established series profiles, makes no request; the provenance records the skip with unchanged fingerprints, zero request characters, and no updates.

Only complete upserts of existing local voices are accepted — each carries exactly `description`, `traits`, and `speechTraits` and a brief reason; stored profile metadata beyond those fields keeps its current values. The pass can coordinate casting-basis traits, descriptions, and `speechTraits`; it can never invent, remove, rename, or reorder voices, change character or role ownership, or touch a series-inherited profile: inherited descriptions, traits, and reference identity are reference-only context. Existing text-supported traits must be echoed exactly and may not be changed for contrast, and the pass never claims new text-supported traits — it coordinates casting, it does not discover evidence. The response is fully validated, including a rebuilt-cast check, before any successful cache entry is written, so a cached replay revalidates the complete cast. Changes, reasons, before/after cast fingerprints, and the guidance fingerprint are kept in `provenance.voice_ensemble`. Validated responses are cached and replayed like every other stage.

A deterministic contrast audit always accompanies the pass. It compares recognized audible dimensions declared in each profile's traits — with a few justified spelling and synonym normalizations — between distinct canonical voices only; aliases of one voice never pair. A dimension missing from either profile is unknown, never a difference, and no traits are invented. Pairs with identical known profiles, or fewer than two demonstrably different known dimensions, become advisory potential-confusion findings carrying voice IDs and dimension names. The before-design findings travel with the ensemble request; provenance keeps both the before-design findings and the findings recomputed on the final cast, so a collision the pass already resolved is not reported as unresolved. This is text analysis of profile records: it is not a measurement of synthesized audio, does not prove two voices are confusable, never blocks rendering, and never demands recasting where source evidence genuinely supports similar voices.

Vocal attributes follow a cue precedence shared by analysis and every refresh path: explicit vocal evidence, the user's deliberate choices, and established series profiles come first; specific narrative characterization and story context come second; only when none of those settles an attribute may names, ethnicity, gender, or descriptions inform a tentative creative casting guess. Such a guess is labeled `basis=casting` with empty evidence, its tentative basis is explained in reason/audit metadata rather than in engine-facing prose, and it is never a factual acoustic claim, a universal stereotype, an exaggerated caricature, or a fixed demographic-to-accent mapping. Gender and ethnicity are not acoustic facts.

The casting-description refresh follows the same cue precedence as analysis and carries one consistent `known_cast` snapshot plus that voice's contrast findings.

## Passage attribution consensus

Set `analysis.attribution_review = true` to review the completed script during `analyze`. It is disabled by default because it adds model calls. The review checks narration, speech separated by tags or actions, continuing and nested quotations, direct thoughts, narrator frames, and individual versus collective speech. It uses the existing cast and performance policy. It cannot add identities, change voices, or rewrite source text.

The scanner proposes complete replacements for affected source blocks. Three blind, separately cached reviewers inspect each proposal against source. A block needs three approvals and exact citations. Rejection, uncertainty, or invalid output leaves the original assignment and adds an unresolved item. Confirmed simultaneous assignment can resolve its corresponding unassigned flag. Human corrections cannot be overwritten. Bracketed delivery is reapplied after accepted changes. Source, cast, segments, verdicts, and fingerprints remain in `provenance.attribution_review`.

Every run performs mandatory speaker completion after this optional review, even when attribution review is disabled. It enumerates all unassigned spoken spans, assigns their surrounding complete source blocks through three independent reviewers, and applies only unanimous resolved decisions. Up to three rounds widen context, with the last adding relevant cast evidence from other chapters. Exact coverage, cast references, quotations, and protected human assignments are validated. Completion cannot invent a character or silently use the narrator. Decisions and remaining counts are retained in `provenance.speaker_completion` and resumable cache audits.

Remaining passages are shown separately with wrapped, paragraph-spaced context and assigned voice labels. The interactive menu offers up to three likely speakers, **Enter** accepts the first, a number selects one, **O** opens the full cast, **B** returns, and **Q** or end of input saves and exits with status `2`. Suggestions do not change assignments until selected. Unassigned speech blocks audio generation; advisory notes do not. JEV, when enabled, ranks plausible candidates but does not authorize assignment.

Rerun `analyze` with the same output directory to resume. Cached validated responses are reused, unresolved blocks receive a fresh bounded review, saved identities and unaffected assignments remain, and unresolved passages are never filled by guessing. Source must match the saved book. Conservative narration-boundary repair splits clear tags such as `"Hello," Mara said, "Again."` into character, narrator, character spans while preserving source text. Ambiguous scopes go through completion. Changes are recorded in `provenance.narration_boundaries`.

To resolve existing directories, use `python3.11 -m epub_tts_narrator resolve-speakers ...`. To review an analysis without rediscovery, use `review-attribution INPUT -o OUTPUT`. Review batches retain complete source blocks and chapter boundaries. Truncated semantic scans subdivide up to two levels; jury requests are not subdivided. Invalid scans or provider failures publish no output, while valid cached calls remain resumable.

## Optional corrections

A voice-profile refresh marks every selected profile invalid in `.voice-profile-refresh.json` before the first model request. Each validated proposal is durably checkpointed as prepared, not yet valid for generation. If a model call or publication fails, rerun the same refresh command: matching proposals are revalidated and reused without another model call, independently of the request cache. Changed inputs, guidance, or provider settings invalidate stale proposals. Rendering and casting are blocked while this book or an ancestor has a pending refresh. Successful publication clears the pending state; if clearing fails after publication, rerun reconciles the published fingerprints instead of repeating model work. Reference recordings are not invalidated or deleted. Stable `.lease` ownership files must not be deleted.

To update only voice profiles in an existing book, run `python3.11 -m epub_tts_narrator refresh-voice-profiles BOOK_DIRECTORY --workers 1`. This refreshes voice descriptions, casting traits, and `speechTraits` without reanalyzing source, identities, assignments, or spelling edits. Existing evidence-backed traits, audition text, reference recordings, and accepted audio files are retained. Changed performance traits require new affected speech during the next render, not new reference voices. Original analysis files and audition bindings are backed up in `.analysis-revisions/`; profile publication validates and refreshes descendant continuation links atomically. Process a series one book at a time, starting with the defining book. Parent profiles are supplied to later refreshes so their established accents are not independently reinvented.

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

Let `batch_count` be the number of bounded discovery batches and `R` the configured number of reconciliation passes. With identity consensus, the baseline is `3 × batch_count + R + 2`: discovery, reconciliation, an identity scan of every batch, ensemble design, attribution, and one spoken-spelling scan. Whole-book mode sends complete source for discovery, reconciliation, and one identity scan, giving `batch_count + R + 4`. Without identity consensus, the corresponding formulas are `2 × batch_count + R + 2` and `batch_count + R + 3`. These baselines include one ensemble request; subtract it when the pass is skipped. These are baselines, not totals. Conditional juries, repairs, retries, continuity scans, completion, and auditions add requests.

Identity cases add three verdict requests per case and preparation requests for proposed splits. Prior series books add a dedicated continuity scan when the current cast is nonempty. A failed or uncertain first-round candidate can trigger the bounded second round, with fresh juries, but that conditional work is outside the baseline. `plan` reports guaranteed scan overhead and explains conditional cost.

Attribution review adds one scan per review batch and three jury requests for each batch with proposals. Semantic truncation can add bounded subdivision attempts. Initial passage tagging can likewise subdivide a truncated batch. Mandatory speaker completion adds three requests per unassigned source block in each required round, with at most `min(5, attribution_workers)` concurrent requests. Audition preparation adds one request per voice lacking usable source speech and a saved creative script.

`analysis.attribution_workers` accepts 1 to 8 and is the shared concurrency ceiling. Identity review uses at most `min(3, attribution_workers)` concurrent calls, passage review at most `min(5, attribution_workers)`, and these phases do not overlap. Defaults are 12,000 source characters per attribution batch, 200,000 serialized request characters, and one validation repair. Characters are not tokens, and cast metadata consumes context. `analysis.chunk_chars`, `analysis.max_request_chars`, `analysis.repair_attempts`, and `provider.max_output_tokens` bound request size and recovery.

Request retries are disabled by default. `provider.max_retries` enables bounded retries for transient HTTP and typed transport failures. Known quota, billing, authentication, TLS, invalid URL, permanent DNS, unrecognized transport, and invalid-output failures are not retried. Logs omit raw errors, URLs, response bodies, keys, and TLS diagnostics.

Validated responses are cached under `.epub-audiobook-cache` beside the output, or under `analysis.cache_dir`. Cache keys include exact prompts, inputs, model, endpoint, and relevant options. Changing request inputs invalidates affected work. Cache entries contain book text and derived annotations, never API keys. Keep the cache and rerun the same command after interruption. A failed analysis does not publish a completed directory. Failed responses remain in `failed/` for bounded repair and are never accepted as successful results.

Resuming the same book needs no flag. `--continue-from` selects prior series context and must remain the same when resuming. Changing timeout, key, or retry settings does not invalidate completed response caches; changing model, prompts, cast, policy, or other request inputs can. Advisory notes do not make a bundle a draft, but missing speakers prevent rendering.

## Spoken spelling and delivery

Analysis normalizes intentionally stylized speech for synthesis while preserving original text and offsets. `Wheerrre issss sheeee?` can become `Where is she?`, with stretched delivery recorded separately. Context-supported dialect spelling may become readable words, but grammar, repetitions, names, numbers, and negation remain. Three reviewers check edits, each requiring two approvals plus local checks for ordered coverage, bounds, spelling correspondence, and protected content. Uncertain edits remain unchanged. Decisions are stored in `provenance.speech_normalization` and segment `speech_normalization`; `spoken_text` is derived from them. Existing prepared books skip unchanged work, and reruns retain cast, voices, and audio. See [FORMAT.md](FORMAT.md#spoken-spelling-normalization).

## Bracketed delivery

Square-bracket text such as `[Fireball]` receives a separate delivery cue, including inside dialogue. Speaker and voice stay unchanged. The script retains exact source text and supplies `spoken_text` without delimiters for synthesis. Configure `performance_policy.bracketed_text`; balanced nested and multi-paragraph groups are supported, while unmatched delimiters are flagged. See [the format contract](FORMAT.md#bracketed-delivery).

## Pronunciation candidates

After publishing a directory analysis, `analyze` collects character names and aliases from the resolved cast, checking each word against the actual source. It also scans the source in bounded batches for place names, other proper names, invented terms, foreign words, and ambiguous spellings. Names are included even when their spelling looks ordinary; generic titles, roles, and function words are excluded from automatic cast-name seeding. Source scanning can still miss a word, so the review command also accepts a manually specified source word.

Each candidate in `BOOK/pronunciations.json` has an exact written word, source quotation, and reason to check it. Pending character and place names automatically receive a suggested CMU Arpabet pronunciation, including vowel stress. The existing configured analysis model proposes it from the name's spelling and book context. These are guesses for listening review, not canonical pronunciations or acoustic verification. Confirmed and keep-default choices, including inherited ones, are reused without new proposals.

Existing books can add this queue without repeating character or passage analysis:

```sh
epub-tts-narrator discover-pronunciations BOOK --config epub-tts-narrator.toml
audiobook-render pronunciations BOOK --config epub-tts-narrator.toml
```

Discovery and phoneme proposals use the existing analysis provider and cached requests, with at most five concurrent calls. Each completed name proposal is saved immediately. A failure leaves the published analysis, completed discovery batches, and saved proposals intact; rerun the command to finish. Rerunning `analyze BOOK` also completes missing discovery or proposals. A completed source/cast-bound scan is skipped. Local corrections survive discovery and do not rewrite the canonical analysis, scripts, cast, or reference recordings. Legacy single-JSON analysis output does not have this per-book sidecar workflow.

Pending candidates do not affect synthesis. Confirmed respellings or CMU Arpabet choices are loaded at render time, independently of expressive delivery planning. Choices follow the saved continuation chain dynamically: a correction confirmed in Book 1 after Book 2 was analyzed is available to Book 2 on its next render. A Book 2 correction or explicit keep-default decision overrides Book 1 for that word in Book 2 and its descendants. Matching uses complete words with exact spelling and case, never name resemblance or fuzzy identity merging. See [pronunciation review](RENDERER.md#pronunciation-review) for audition and correction commands.

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
