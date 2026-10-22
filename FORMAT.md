# Audiobook analysis version 1 and bundle version 2

The executable contract validator is `epub_tts_narrator/contracts.py`. Validate before consuming a file; a saved `readiness` value is informational and must be recomputed after edits. Fields may gain optional metadata, but incompatible changes require a new integer `version`.

## Analysis document

```text
format: "epub-audiobook-analysis"
version: 1
source: Source
cast: { characters: Character[], roles: Role[], voices: Voice[] }
segments: Segment[]
issues: Issue[]
provenance: object
readiness: { ready: boolean, errors: string[], review_count: integer }
```

`readiness.ready` means the document is valid and every spoken passage has a performer. `review_count` retains advisory notes for optional corrections and may be nonzero in a ready bundle. Older readiness flags remain readable; consumers validate current structure and speaker assignments.

`source` contains book metadata, ordered chapters and blocks, exclusions, warnings, and the extraction policy. `book.id` is the SHA-256 of the input EPUB. Blocks contain `id`, `chapter_id`, exact canonical `text`, and `source: {href, element_id, tag}`. Chapters contain `id`, `title`, `source_href`, and ordered `block_ids`. EPUB source references identify a resource and nearest containing block element; they are not XML byte offsets. When an element has no ID, its `element_id` is null.

Blocks may contain `inline_spans: [{start, end, kind, origin}]`, with zero-based, end-exclusive codepoint offsets into canonical block text. `kind` is `emphasis` or `strong`; `origin` is `tag` or `style`. These are source cues from supported markup and simple local style rules, not a complete rendered CSS cascade or speaker labels. Optional `break_before: "blank_paragraph"` records an empty paragraph before the block within the same spine resource.

An `Evidence` record is `{block_id, quote}`. The quote must be a nonempty exact substring of that source block. A repeated quotation points to its block, not a uniquely selected occurrence; source spans on segments are precise.

| Record | Required fields and meaning |
| --- | --- |
| Character | `id`, `name`, `aliases` (strings), `description`, `evidence` (Evidence[]), `voice_id` |
| Role | `id`, `kind` (`narration` or `dialogue`), `character_id` (string or null), `voice_id`, `description` |
| Voice | `id`, `description`, `traits` (Trait[]) |
| Trait | `name`, `value`, `basis` (`text` or `casting`), `evidence` (Evidence[]) |
| Issue | `id`, `kind`, `target`, `reason`, `resolved` (boolean) |

Text-supported traits need at least one evidence quotation. Casting traits are proposals. `Voice.description` is a proposed synthesis/casting direction, not an independent claim that all its qualities appear verbatim in the book. Distinct characters must have distinct voice IDs. A dialogue role attached to a character uses that character's voice; a narrator role attached to a character may share it or select a separate profile. An external narrator has `character_id: null`.

## Identity consensus provenance

New analyses enable identity consensus by default through the TOML setting `analysis.identity_consensus`. The audit runs after cast discovery and before passage attribution. Proposed discovery merges are deferred, source scans seek both incorrectly combined identities and duplicated characters, and each concrete identity decision receives three independent verdict requests. Cases without enough evidence for a concrete decision remain unresolved. Any accepted split is judged as a concrete cast patch with distinct character and voice records. Proposed matches to prior-series characters also require this review before series canonicalization.

`provenance.identity_consensus` stores audit metadata with `version: 1` and `policy: "unanimous_3"`, the `source_fingerprint`, complete `original_cast`, `original_cast_fingerprint`, `final_cast_fingerprint`, `scans`, `cases`, and `series_cases`. Case records retain proposals, evidence, verdicts, and the resulting decision. Series votes additionally bind to `series_registry_fingerprint`. Fingerprints describe the source and cast states used by the audit; later series canonicalization or human corrections may change cast IDs or records. Consumers should preserve this provenance with the analysis and bundle rather than infer fresh approval from an unchanged voice ID.

Each review has three distinct reviewer slots, so an identical cached response cannot stand in for three independent calls. Reviewers receive the proposal and evidence without seeing peer verdicts. Only three approvals permit the proposed decision; uncertainty, disagreement, and conflicting changes remain reviewable. Evidence quotations and the proposed resulting cast must pass local validation. Invalid responses can be repaired within the configured bound. A patch or verdict that exhausts validation repairs leaves its case unresolved and its change withheld; a failed source scan or provider request aborts analysis. All requests use the configured model, whose errors can be correlated. Consensus is evidence of a completed procedure, not a calibrated probability of correctness or human approval.

When `identity_consensus = false`, the additional scans and verdicts are skipped. Discovery merges can then be applied with advisory uncertainty flags, as in earlier analyses. Older saved analyses and offline `series-import` results need not contain consensus provenance; importing an existing analysis does not run a new model audit. Export preserves advisory notes without requiring approval. `--allow-review` is a compatibility option; valid, fully assigned speech is ready even when notes remain.

## Segments

```json
{
  "id": "b000003:0:7",
  "block_id": "b000003",
  "start": 0,
  "end": 7,
  "text": "“Wait,”",
  "kind": "dialogue",
  "role_id": "mara_dialogue",
  "voice_id": "voice_mara",
  "delivery": {"direction": "", "evidence": []},
  "attribution": {
    "basis": "explicit",
    "reason": "Attributed by the following first-person speech tag.",
    "evidence": [{"block_id": "b000003", "quote": "I said"}],
    "candidates": [],
    "review_required": false
  }
}
```

Offsets are zero-based **Unicode code points**, with exclusive `end`, relative to the canonical block text. They are not UTF-8 byte offsets or JavaScript UTF-16 code units. Consumers in JavaScript should use code-point iteration (`Array.from(text)`) when verifying ranges. The analyzer-generated ID derives from the block ID and range. Preserve IDs when applying local corrections; changing segmentation changes affected range IDs.

`kind` is one of `narration`, `dialogue`, `thought`, `letter`, `heading`, or `other`. `attribution.basis` is `explicit`, `context`, `unresolved`, or `user`; only the corrections path can create a `user` attribution. Reasons summarize evidence or uncertainty and are not model reasoning traces. There are no uncalibrated probability scores.

For unresolved text, `role_id` and `voice_id` are both null, `basis` is `unresolved`, `review_required` is true, and `candidates` lists possible role IDs (possibly empty). Unknown character identity is not automatically conflated with a narrator. A known role may also be flagged for review.

For identified people saying the same words together, the segment instead has an optional `simultaneous_speakers` field:

```json
{
  "role_id": null,
  "voice_id": null,
  "simultaneous_speakers": [
    {"role_id": "mara_dialogue", "voice_id": "voice_mara"},
    {"role_id": "lena_dialogue", "voice_id": "voice_lena"}
  ]
}
```

This fragment supplements the normal segment fields. It requires at least two distinct existing dialogue roles and distinct voice IDs, with each voice matching its role in the cast. The segment kind is `dialogue`, `thought`, or `letter`; attribution is resolved (`explicit` or `context`, or `user` through corrections), has no candidates, and has `review_required: false`. Model assignments require exact evidence for the shared delivery. Null single-performer IDs therefore do not by themselves mean an unassigned segment: consumers must also inspect `simultaneous_speakers`. Each listed performer says the complete segment text. The list represents actual performers, never alternative identities or an uncertain crowd. Different words spoken over one another and staggered entrances are not represented by this version.

The mandatory completion pass in `analyze` enumerates all remaining unassigned spoken spans and obtains three independent assignments using the final cast and source context. It updates only unanimous resolved decisions at common source boundaries, preserving source text, existing local delivery, protected human assignments, and unrelated review concerns. Up to three rounds supply wider context, including relevant cast evidence from other chapters in the final round, without exposing peer answers. `provenance.speaker_completion` retains input fingerprints, cases, reviewer results, decisions, and remaining unassigned counts. Unassigned speech requires an explicit speaker choice before generation; it is never filled with a fallback narrator.

The interactive CLI then offers a decision for each remaining passage. Explicit choices use `attribution.basis: "user"`; `provenance.speaker_decisions` binds those choices to source, cast, input/output script fingerprints, segment ranges, text fingerprints, previous attribution, and selected roles/voices. Earlier runs remain in `speaker_decision_history`. Blank input asks again; stopping retains earlier choices. Redirected input uses the controlling terminal when available. If input ends or no terminal is available, the CLI saves progress and returns status `2` while speakers remain unassigned. Saved directory analyses can be resumed with `analyze BOOK_DIRECTORY` without rediscovering the cast.

Local `delivery` is optional in meaning but always present structurally. Empty direction means no special instruction. Model-produced nonempty directions require evidence. User-authored block replacements can specify creative direction without textual evidence. Delivery never creates a new persistent voice by itself.

At model ingestion only, empty partial delivery objects (`{}`, `{"direction":""}`, or `{"evidence":[]}`) are expanded to the complete empty default. Nonempty or malformed partial objects are rejected. This structural completion changes no source text, speaker, evidence, or delivery cue; the raw model response is retained in the cache. Published files and explicit user edits still require the complete delivery schema above.

All source blocks must be covered exactly once and in order, without gaps or overlaps. For each segment:

```text
segment.text == source_block.text[segment.start:segment.end]
segment.voice_id == cast.roles[segment.role_id].voice_id   (when assigned)
speaker.voice_id == cast.roles[speaker.role_id].voice_id   (each simultaneous speaker)
```

Concatenating a block's segment texts reconstructs that canonical block exactly. Paragraph/chapter boundaries are structural; do not concatenate unrelated blocks without considering those boundaries. The renderer may create separate pronunciation or speakable-text transformations, but should retain the source text and offsets for auditing.

## Bracketed delivery

Balanced square-bracket passages are split at their source boundaries after speaker attribution. They keep the selected role and voice. Each resulting segment has optional `performance` metadata:

```json
{"kind":"bracketed","group_id":"bracket:b000099:381","direction":"Give this text a slight audible change in delivery.","delimiters":"omit","enter":true,"exit":true}
```

`text` always retains the exact brackets and source offsets. Without speech normalization, `spoken_text` supplies the same text with square brackets omitted; no words or other punctuation may change. Approved spelling edits below compose with bracket omission. The TTS consumer should synthesize `spoken_text` when present, otherwise `text`, and apply the delivery direction using its own engine settings. An empty `spoken_text` is a control-only segment and needs no audio. Do not speak the words “open bracket” or “close bracket.”

`group_id` joins pieces of one bracket passage across existing speaker segments or paragraphs. `enter` and `exit` identify its outer boundaries. Nested brackets form one delivery group. The cue does not create a new character, force a different voice, or impose fixed silence around a skill name inside a sentence. Unbalanced brackets remain literal and create review issues; groups never cross chapter boundaries. Direction is configurable in `[performance_policy]` as `bracketed_text`.

## Spoken spelling normalization

A segment may include `speech_normalization: {"version": 1, "edits": [...]}` together with `spoken_text`. Each edit has `start`, `end`, `original`, `replacement`, `direction`, and `reason`. Edit offsets are Unicode code points relative to the segment's exact `text`, ordered, nonoverlapping, and end-exclusive; `original` must equal that source slice. Replacements contain ordinary spelling, while `direction` contains a plain acoustic instruction. They cannot contain engine markup. The original `text`, source block, and segment offsets never change.

The complete `spoken_text` must be exactly reproducible from approved edits and any bracket-delimiter omission. It cannot be an arbitrary rewritten passage. Analysis records proposals, three independent verdicts per edit, decisions, and input/output fingerprints in `provenance.speech_normalization`. Boundary changes invalidate affected segment edits so they can be reconsidered against the new spans.

The renderer combines distinct edit directions with existing `delivery.direction`, supplies the derived words to both TTS and verification, and retains source ranges. Unchanged spoken characters map precisely to the original source. A replacement maps to its entire original source span, including when a speech chunk cuts through that replacement; it does not invent character-level alignment between spellings of different lengths. Synchronization retains both the normalized spoken text and original source ranges.

## Series continuity

The cumulative `series-cast.json` in each analysis directory, or a legacy explicitly selected registry, uses `format: "epub-audiobook-series"`, `version: 1`, a persistent UUID `series_id`, and a `books` array. Each book snapshot includes `source_fingerprint`, book metadata, a complete cast, review issues, computed readiness, and `series_provenance`. Quotations inside a snapshot belong only to that snapshot's source fingerprint. Loading a registry never interprets those quotations as evidence from the current book.

Series-linked analyses keep the normal version-1 analysis contract. Character IDs use `series_char_…` and voice IDs use `series_voice_…`; accepted recurring identities retain their prior IDs. Role IDs remain local to each analysis. A renderer can reuse its `voice_id → engine voice` mapping across books within the selected series. For directory analysis, a fresh series namespace derives from the final output directory; continuation retains it from the preceding book. Legacy registry identities derive from the explicitly selected registry path until its first save. Saved IDs persist when files move. An unrelated book starts a new history by omitting `--continue-from`.

Optional current character `series_identity` is a model proposal with an existing prior `character_id`, `basis` (`context` or `uncertain`), a brief `reason`, exact current-source `evidence`, and boolean `review_required`. A shared name alone cannot confirm a match. Conflicting or uncertain current proposals get separate identities and unresolved review issues. A unique current-source-supported match can retain established IDs despite prior casting or identity concerns. Relevant earlier issues carry forward unresolved with `series_origin.kind: "inherited_series_review"`, their original source fingerprint, book, issue ID, kind, target, and reason. Repeated inheritance retains that original provenance and does not duplicate the same concern. Stable IDs do not imply human approval; these issues remain advisory and do not block readiness. Registering the exact same analysis requires no model and preserves its existing mapping and review state.

Reused prior voice descriptions and traits are production casting references. In the current analysis, inherited traits use `basis: "casting"`, empty `evidence`, and `series_origin` containing the original source fingerprint, voice ID, trait index, and original basis. Current character evidence remains current-source evidence. Same-source profile corrections retain their own current evidence. Identified narrators can reuse a separate established narration voice; an unidentified main frame may reuse default narration casting without asserting a character identity. Distinct POV and afterword scopes remain separate, and casting conflicts produce review issues. Role-specific narration concerns carry only to the matching role scope; sharing a character and voice does not make an afterword concern apply to the main narrative frame.

`provenance.series` records the series ID, source and reference-context fingerprints, the input cast fingerprint, local-to-global character/voice mappings, accepted identity matches, unresolved candidates, inherited review origins, and `human_approved: false`. Identity matches record `prior_review_required` when their character or voice has inherited review issues. Every import refreshes applicable review history while retaining its selected cast; a changed review set records `review_context_fingerprint` separately from the original matching context. An explicitly resolved issue saved in a registry snapshot closes that exact original source-and-issue ID for future inheritance; a new concern needs a new issue ID. Matching never resolves issues itself. Registry persistence is atomic and rejects concurrent changes. Series provenance and matching do not resolve unrelated review flags or alter source text, offsets, segmentation, or bracketed performance metadata.

## Audition scripts

Fresh CLI analysis records `provenance.voice_samples` with version, source/cast fingerprints, a `scripts` map keyed by voice ID, and a `samples` map of selected audition text and provenance. Auditions contain 25–60 words and at most 650 characters. `scripts` holds only generated text with `{text, basis: "casting", reason}`. These invented lines are production casting material and never replace a source segment or claim textual evidence for a character trait.

Each selected sample has `reference_text` and `source`. `source.kind: "book_excerpt"` identifies one exact selection using chapter, block, segment, spoken offsets, and canonical source ranges. `source.kind: "book_excerpts"` instead has a literal `separator`, `segment_ids`, and an `excerpts` array. Each excerpt contains its exact `text`, independent source coordinates, and `sample_start`/`sample_end` offsets into the assembled audition. Joining excerpt texts with the separator reconstructs the complete audition. There is no invented top-level contiguous source range. All excerpts belong to the same canonical voice, including explicit aliases; a resolved simultaneous passage may supply an excerpt for each actual participant.

`source.kind: "casting_script"` identifies original character-specific audition text, with `basis: "casting"`, voice identity, and rationale. It has no purported source segment or range. The renderer also accepts an optional voice-profile `sample_script` with the same creative-script shape for compatible producers; the included analyzer stores these scripts in provenance so cast identity fingerprints do not change.

For existing books, the renderer can save generated scripts in book-local `audition-scripts.json`, with version and exact source/cast fingerprints. This sidecar supplies `audition_scripts` at runtime and leaves immutable analysis, cast, continuation links, and saved voices unchanged. A mismatched sidecar is rejected. Full source provenance remains in persisted casting drafts and voice-creation metadata. Replacing an automatic draft does not change an existing reference recording or its verified transcript; regeneration creates a new candidate. Explicit user scripts retain `source.kind: "user"` and are not automatically replaced for length.

## Corrections

Every correction file requires `source_fingerprint` and `cast_fingerprint`, both printed by `review`. Fingerprints use SHA-256 over UTF-8 JSON with sorted keys, no insignificant separators, and literal Unicode. They bind corrections to the exact extracted source and input cast. Use the CLI to create/apply them instead of independently approximating canonicalization in another language.

Optional fields:

- `assignments`: `[{segment_id, role_id, kind?}]`. Resolve/confirm an existing segment with an existing role. Setting the same role confirms a flagged assignment.
- `blocks`: `[{block_id, segments:[{text, kind, role_id, delivery?, simultaneous_role_ids?}]}]`. Replace a complete block's segmentation, with exact text reconstruction. This supports adjusting dialogue boundaries. A simultaneous piece sets `role_id: null` and supplies the distinct existing dialogue roles; the application derives their voice IDs.
- `cast`: a complete replacement `{characters, roles, voices}` object. Supports corrected identities, names, narrator links, new missing characters, and voice descriptions. Keep existing referenced role IDs or reassign their segments in the same edit.
- `resolved_issues`: existing issue IDs explicitly accepted or resolved by the editor. This does not bypass unresolved segment checks.

Unknown correction keys fail. Edited documents retain all correction payloads in `provenance.corrections`. Reapplying the same correction file to an independently changed cast fails; inspect and refresh the edits rather than bypassing the check.

## Exported bundle

The directory has `manifest.json`, `cast.json`, `source.json`, `review.json`, and chapter scripts named `scripts/chapter-NNNN.jsonl`. The manifest has `format: "epub-audiobook-bundle"`, `version: 2`, `book`, `chapters`, `source_fingerprint`, `readiness`, `files`, `sha256`, `offsets`, and `provenance`. Analysis documents and series registries remain version 1.

```json
{
  "files": {
    "source": "source.json",
    "cast": "cast.json",
    "review": "review.json",
    "scripts": [
      {
        "chapter_id": "ch0001",
        "title": "Chapter 1",
        "heading_block_id": "b000001",
        "path": "scripts/chapter-0001.jsonl",
        "segment_count": 12,
        "sequence_start": 0,
        "sequence_end": 12
      }
    ]
  }
}
```

Directory analysis additionally saves `analysis.json` and cumulative `series-cast.json`. Its manifest has `analysis_directory: {"version": 1, "sha256": {"analysis.json": "…", "series-cast.json": "…"}}`, binding those two files to the bundle. The renderer files retain their separate checksums in `sha256`.

New directory outputs also have a top-level `continuation` field. It is `null` for the first book, or an object describing the preceding analysis directory:

```json
{
  "continuation": {
    "version": 1,
    "path": "../book-01",
    "series_id": "persistent-series-id",
    "source_fingerprint": "parent-source-fingerprint",
    "analysis_sha256": "parent-analysis-sha256",
    "series_cast_sha256": "parent-series-cast-sha256"
  }
}
```

`path` is relative to the current book's final directory, not its staging directory or the global configuration. The identity fields bind the link to the intended parent book and series; the hashes bind its analysis and cumulative cast. Runtime resolution validates the parent and follows its own continuation recursively. Broken links, cycles, changed analysis or history, and mismatched identities fail explicitly. Voice recordings are intentionally excluded from these immutable analysis hashes: an ancestor can acquire voices after a descendant was analyzed.

The renderer derives its voice namespace from `provenance.series.series_id` and stores new assets beneath the current book's `voices/<series-hash>/` directory. Lookup searches the current book, parent, grandparent, and so on; the nearest selected reference for the canonical voice ID wins. Current-book regeneration, imports, and selection changes create local overrides and never mutate ancestor stores. Descendants can inherit the override; earlier books and sibling branches cannot. Continuation preserves access to earlier references without copying them into every book. Older bundles without directory metadata retain their explicitly configured shared-store behavior.

Manual voice aliases are mutable rendering metadata in the book root, outside the immutable analysis checksums. The optional `voice-aliases.json` file has this shape:

```json
{
  "format": "audiobook-book-voice-aliases",
  "version": 1,
  "series_id": "persistent-series-id",
  "book_id": "current-source-book-id",
  "aliases": {
    "source-voice-id": "target-voice-id"
  }
}
```

The book and series fields bind the mappings to their intended analysis. Keys and values are exact known voice IDs, never character names. Automatic role sharing comes from explicit character identities in each cast. Manual aliases are resolved from the continuation chain at rendering time; the nearest explicit mapping overrides earlier mappings and automatic role sharing. Cycles and invalid IDs fail before a voice can be used. A child does not snapshot inherited manual mappings, so aliases added later in an ancestor remain available. Removing a local mapping restores inherited or automatic behavior. The global configuration contains no alias mappings. Voice aliases select shared reference audio without changing character identities in the analysis.

`files.scripts` follows book order. Each file contains a contiguous span belonging to one native source chapter. A native chapter is an EPUB resource; when it contains multiple numbered story headings, it can produce multiple script entries with the same `chapter_id`. The manifest's `chapters` records and each row's `chapter_id` remain the original source values. Display the script entry's `title`; `heading_block_id` identifies the canonical block used for that title, or is null for a source-title fallback or detected contents list.

Additional boundaries require a whole-block English `Chapter` label (Arabic digits, Roman numerals, or supported English number words), an HTML heading or `heading` segment classification, and intervening non-heading prose. Dense or consecutive chapter-label lists remain one `Contents` section. Other resources retain their existing boundaries, including empty resources; named chapters, labels in other languages, arbitrary scene headings, and uncertain chapter formats are not automatically split. This conservative partition is reviewable metadata, not a guarantee that every EPUB's intended chapter structure has been recovered.

`sequence_start` is inclusive and `sequence_end` exclusive; their difference equals `segment_count`. The first entry starts at zero, and each following entry starts at the preceding entry's end. An empty source chapter has an empty file, `segment_count: 0`, and equal sequence bounds at the current global cursor. Filenames use the one-based output position, padded to at least four digits; titles and IDs never become path components.

The manifest's `sha256` object maps every relative data filename to its SHA-256, including every chapter script and empty file. Concatenate parsed rows in `files.scripts` order to reconstruct the full script. There is no root `script.jsonl` in version 2. Older version-1 bundles use `files.script: "script.jsonl"` and one complete script; readers must explicitly support their version instead of assuming one layout. Re-exporting an analysis changes the bundle layout without changing that analysis or its annotations.

Each chapter script line is a Segment plus:

- `sequence`: zero-based position in the complete book script, continuing across chapter files.
- `chapter_id`: its source chapter.
- `break_after`: `paragraph` at the end of a source block, otherwise `none`.
- Optional `break_before`: `blank_paragraph` when the source has an empty paragraph before this block. It is a literal spacing cue, not a claim that a new scene begins.

`paragraph` means an end-of-block pause hint, including headings; the renderer should use `kind` and script-file boundaries to choose actual pauses. Native chapter IDs follow selected spine resources, while script entries may divide a resource into multiple logical chapters. Validate final chaptering against the book before creating `.m4b` chapter markers.

The audio generator owns `voice_id → engine/model/voice/settings` mapping, audio chunking, pronunciation adjustments, timing, pauses, mixing, and container assembly. Multiple roles with one voice ID must retain one vocal identity. Distinct voice profiles should remain distinguishable unless the production explicitly overrides the casting.

Require the manifest, supported version, valid file checksums, valid source coverage and cast references, and assigned voices for every selected spoken passage before generation. Advisory notes and historical `ready: false` flags do not require an override. Current readiness means structurally valid and fully assigned; it does not certify perfect literary interpretation or that voices have already been auditioned.

## Text and audio synchronization

The pair `source_fingerprint` and segment `id`, together with `block_id`, `start`, and `end`, identifies an exact range in the canonical extracted source. Chapter-file partitioning preserves all those values. This supports attaching audio timing to the same segments later without guessing which text was synthesized.

The audio renderer must add actual clip and assembled-audiobook start/end times. Word timing requires timestamps from the TTS engine or a subsequent alignment pass. It must map any `spoken_text` bracket omissions and later pronunciation transformations back to the source range, and account for inserted pauses, trimming, and mixing when placing timestamps on the final audio timeline. This analyzer does not yet emit audio times or a synchronization sidecar.

The included renderer writes `sync.json` with source ranges and sample-frame boundaries at chapter and book level. For simultaneous speech, it synthesizes and verifies each performer independently before mixing; it does not use recognition of the mixed recording as proof that every voice spoke correctly. All lanes begin at relative frame zero, their original durations are retained, and the mixed chunk lasts until the longest lane ends. Shorter lanes are zero-padded, and each receives a fixed `1 / speaker_count` weight. Chapter loudness normalization follows composition and preserves the frame timeline.

Each simultaneous synchronization entry retains its source text once and adds `simultaneous_speakers`, `speaker_lanes`, and `mix`. Each lane includes `role_id`, `voice_id`, accepted-audio hash, request hash, validation key, relative `start_frame`/`end_frame`, and chapter/book start/end frames. The enclosing entry ends at the longest lane. `verification.json` carries each lane's full successful verification report; the combined entry records `method: "independently_verified_simultaneous_lanes"`. Mixture and chapter caches bind all performer assets, request identities, verification policy, audio hashes, and frame counts. Saved reference selection, including per-book voice aliases and recursive ancestor lookup, works the same way as for solo speech.

Canonical block coordinates are normalized Unicode code-point offsets, not offsets into the original EPUB DOM. `source.href` and optional `element_id` provide resource and coarse element references; the current output does not contain DOM ranges, EPUB CFIs, or SMIL media overlays. Reliable synchronization to the original ebook therefore requires a verified mapping back to the exact EPUB identified by `book.id`. In particular, EPUB CFI character offsets count UTF-16 code units, which differ from these code-point offsets for supplementary characters. See the [W3C EPUB CFI character-offset definition](https://w3c.github.io/epub-specs/epub33/epubcfi/#character-offset).
