# EPUB audiobook analyzer

A Python CLI that turns an EPUB into a **cast and voice roster** and a **text-preserving performance script**. Its output is independent of the TTS engine. A separate renderer can consume the exported bundle, choose actual engine voices, synthesize audio, and assemble an `.m4b`.

The separate **`audiobook-render` CLI** provides local Drama 3 inference through MLX, interactive voice casting with playback and regeneration, persistent reference voices, speech verification, loudness-normalized M4A chapters, and chapterized M4B assembly. New voice references default to Gemini when its API key is configured; Drama 3 uses those saved references to synthesize the audiobook locally. `render BOOK` runs the full pipeline; `render BOOK --chapters-only` saves playable chapter checkpoints, and `assemble BOOK` packages completed chapters without loading speech models. Use `--chapters 3-5,7` for a per-book selection. See [RENDERER.md](RENDERER.md) for installation, file-based configuration, commands, and current validation limits.

Optional JEV confirmation checks whether the requested passage and recognized speech would sound roughly the same aloud, for both voice samples and chapter audio. Every nonexact transcript reaches JEV; no edit-distance or substring rule can veto it. Acoustic failures such as silence still fail locally. JEV sees the complete requested chunk or sample and each distinct recognized transcript, never audio. Its actual score controls acceptance. Accepted audio is reused across verification-policy changes; an outage retains unaccepted audio so the same command can retry verification. Configure JEV once in the global file as described in [RENDERER.md](RENDERER.md#optional-jev-confirmation).

This is a first working implementation. Extraction, validation, API handling, caching, corrections, and export have offline tests. Live-model work has included a complete book and selected excerpts from another book, plus focused attribution regression cases. Identity consensus has also passed four small original-fiction regression scenarios using the configured Neuralwatt model: identical-name people, a nickname after a long absence, similar names, and one person whose role and voice change. These checks do not establish accuracy on a representative set of books. Ambiguities remain visible for review; structural validation does not prove a speaker assignment is semantically correct.

## Run

Requires **Python 3.11 or newer**. There are no third-party runtime dependencies. Run from this directory:

```sh
python3.11 -m epub_audiobook --help
```

Optionally install the command into your own virtual environment with `python3.11 -m pip install -e .`; the installed command is `epub-audiobook`.

On a new checkout, create one global configuration file and set your model and credentials once:

```sh
cp epub-audiobook.example.toml epub-audiobook.toml
```

Set `provider.model` to your chosen model. If the endpoint requires authentication, uncomment `provider.api_key` and put its key in that file. No default model is chosen. Keys are read directly from the configuration file and are never written into analysis files or caches. The template is tracked; `epub-audiobook.toml` and `*.local.toml` are ignored by Git.

If you already have a configured `epub-audiobook.toml`, keep it rather than replacing it with the template. Run both commands from the project directory and omit `--config`. From another working directory, pass `--config /path/to/epub-audiobook.toml` to use the same file. Book-specific configuration files are not needed for the directory workflow.

`extract`, `plan`, `analyze`, `review-attribution`, and `resolve-speakers` load `epub-audiobook.toml` from the current working directory when it exists. Each also accepts `--config PATH` to select another file. Extraction and planning work with built-in defaults when no configuration file exists; model calls require a configured model. Keep global provider, analysis, extraction, and performance settings in that one TOML file. Add the renderer sections from [audiobook-render.example.toml](audiobook-render.example.toml) to the same file when setting up audio generation; `audiobook-render` also loads it automatically. Choose the book, output directory, and optional previous-book directory on each command; no book or series needs its own configuration file. See [epub-audiobook.example.toml](epub-audiobook.example.toml) for the global settings and their defaults.

```sh
# Inspect extraction and request counts without model calls or file writes.
python3.11 -m epub_audiobook plan book.epub

# Analyze one book and create its renderer bundle in ./book.audiobook.
python3.11 -m epub_audiobook analyze book.epub

# Or choose an output directory.
python3.11 -m epub_audiobook analyze book-one.epub -o runs/book-01

# Continue the series, using the preceding output directory as context.
python3.11 -m epub_audiobook analyze book-two.epub \
  -o runs/book-02 --continue-from runs/book-01

python3.11 -m epub_audiobook validate runs/book-02/analysis.json
python3.11 -m epub_audiobook review runs/book-02/analysis.json
```

Omitting `-o` uses `./<input-stem>.audiobook` in the current directory. The output directory is itself the renderer bundle: it contains `cast.json`, chapter scripts, `manifest.json`, the full `analysis.json`, and a cumulative `series-cast.json`. No separate export step is needed. Every first-book analysis creates stable series IDs, even when no continuation is planned yet. When review items remain, the bundle is saved as a draft with `readiness.ready=false`; success means analysis finished, not that all speakers are resolved.

Git tracks the application, tests, documentation, configuration templates, and original-fiction examples. Local configurations, model downloads, input EPUBs, generated audio, caches, and run directories (`runs/`, `old-runs/`, `dist-runs/`, and `*.audiobook/`) are ignored. The small `examples/last-lantern/the-last-lantern.epub` is an intentional test fixture. Keep custom output directories outside the repository or add them to `.gitignore`; their JSON files can contain the full book text. The historical seven-book migration utilities and their tests are kept under `scripts/legacy_series/`, separate from private run data.

Use a new output directory for a new analysis. An existing completed output is refused, and directory analysis does not accept `--force`. After a failed analysis, repeat the command to reuse successful cached requests. `plan` accepts the same `-o DIRECTORY` and `--continue-from DIRECTORY` options to inspect the intended run without changing files or calling the model.

The same configuration applies to every command. If it lives elsewhere, pass `--config /path/to/epub-audiobook.toml`; this selects a global settings file, not a book-specific setup. To use a local compatible server, change the provider section in that file:

```toml
[provider]
model = "YOUR_LOCAL_MODEL"
base_url = "http://127.0.0.1:8000/v1"
json_mode = false
```

Saved extraction is also a valid analysis input:

```sh
python3.11 -m epub_audiobook extract book.epub -o book.source.json
python3.11 -m epub_audiobook analyze book.source.json -o runs/book
```

Set `provider.json_mode = true` or omit it when the server supports `response_format: {"type":"json_object"}`. Add `provider.api_key` if that server requires one. The client calls `POST <base-url>/chat/completions`; compatibility requires textual JSON content and a successful `finish_reason: "stop"`. Refusals and truncated outputs are never accepted; eligible attribution requests can use the bounded subdivision described below. JSON mode does not enforce the application schema, so all responses undergo local validation. These fields follow the [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) and [JSON-mode guidance](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).

HTTP is supported for loopback servers; remote endpoints require HTTPS. Redirects are refused. Book text is sent when `analyze`, `review-attribution`, or `resolve-speakers` runs, to the configured endpoint. The CLI prints that endpoint before starting. Configuration values are literal; keys and other settings are not interpolated from external variables.

## Analysis stages

1. Extract canonical text from XHTML resources in EPUB spine order. Preserve chapter/resource membership and source element references.
2. Discover characters, aliases, narrative roles, and proposed voices in bounded batches while retaining the accumulated cast.
3. Reconcile the book-wide cast using accumulated textual evidence and scene summaries. Optional additional passes continue from the preceding accepted cast. Defer proposed identity merges until the identity audit.
4. Scan the source for duplicated characters, incorrectly combined identities, and ambiguous names. When a series registry has prior books, separately check every current character for continuity, even if discovery proposed no matches. Require three independent, unanimous approvals before applying an identity change or confirming a proposed earlier-series identity. Keep unresolved cases visible for review.
5. Attribute each exact text span with the resulting cast and neighboring context. Dialogue can be split inside paragraphs and sentences. Character narration can share its dialogue voice or use another voice.
6. Optionally scan the completed script against the source for passage attribution mistakes. Require three unanimous block-level reviews before applying a proposed correction.
7. Find every remaining unassigned spoken passage deterministically and request three independent source-based assignments. Apply only unanimous resolved decisions. Revisit unresolved cases with wider source context; keep genuine uncertainty visible rather than choosing a fallback voice.
8. Prepare a useful audition script for each voice: select enough clearly attributed source speech, joining separate statements when needed, or ask the configured model to write character-specific casting text.
9. If speech remains unassigned after three automatic review rounds, ask for a final speaker decision when running in a terminal. Validate and save the analysis, cumulative series cast, and renderer bundle together. Analysis notes are advisory; no separate review or acceptance step is required.

A character, a narrative/dialogue role, a voice profile, and a passage's local delivery are separate records. Distinct characters receive distinct voice profiles. A narrator role may reference a character and share that character's profile. Voice profiles contain text-supported traits separately from creative casting choices. Accent, perceived age, and other unspecified traits need not be invented.

Fresh analysis creates reusable acoustic casting directions as part of discovery, cast reconciliation, and any identity split that adds a voice. The model is instructed to write a complete voice description in one or two concise sentences, choosing a few compatible audible traits: typical pitch, resonance, vocal weight, texture, accent, articulation, and natural cadence. Age, gender presentation, and fictional context belong there when they help define the sound. Personality becomes vocal choices, such as precise consonants and measured phrasing for a meticulous character. Essential harsh or unusual qualities are preserved while keeping speech intelligible; collective roles use a representative reference performer. Temporary emotion, urgency, whispering, shouting, and changes of pace belong in each passage's delivery.

The description should cover every chosen audible trait. Default voice creation uses only that description plus the audiobook context; structured trait fields remain evidence metadata and are not appended to the prompt. Fresh analysis needs no separate prompt-repair command. An older profile whose important vocal details exist only in trait fields may need its creation prompt edited before casting. Custom casting drafts, selected recordings, and inherited voices remain unchanged. Returning series characters retain their profiles and stable voice IDs; casting and rendering look up their saved references through the book's continuation chain. These instructions guide model proposals without guaranteeing their form or the resulting sound; listening is still needed.

Fresh CLI analysis also prepares audition scripts of 25–60 words, normally aiming for 35–50, with a maximum of 650 characters. Clearly attributed ordinary speech is preferred. Several separate statements by the same canonical voice can be joined, retaining the chapter, segment, and exact source ranges for each excerpt. Resolved simultaneous dialogue is eligible as a solo audition for each identified participant. Ambiguous attribution, extreme delivery, repeated cries, and engine control syntax are excluded. If there is insufficient usable speech, the configured analysis model writes original character-specific audition text using the character, voice profile, and available examples. This text is marked as a casting choice and never enters the audiobook script or masquerades as a book quotation. Prepared selections and creative scripts are saved in `provenance.voice_samples`.

Older books acquire missing scripts automatically when opening `audiobook-render cast`, preparing samples, or creating missing voices during rendering. The same global model configuration is used. Generated scripts for these books are stored in the book's `audition-scripts.json`, bound to its source and cast fingerprints, so no reanalysis or changes to series history are required. Once prepared, they are reused without another model call. Old short automatic drafts are upgraded; custom user scripts and useful established auditions are retained. Updating a draft does not regenerate or alter its saved cloning reference: the revised script is used when a voice is subsequently generated or regenerated.

The model returns exact text fragments, not numeric offsets. The application aligns those fragments with the immutable source, calculates offsets, and rejects omissions, reordering, duplication, or rewritten words. Unknown speakers use null role/voice IDs and remain reviewable; they are never automatically relabeled as the narrator.

When identified characters say the same words together, a segment instead carries an explicit `simultaneous_speakers` list containing each existing role and voice. This is a resolved performance, even though its single `role_id` and `voice_id` are null. Possible alternative speakers remain unresolved `candidates`; they never become a mixed performance merely because more than one candidate exists. The renderer synthesizes the passage separately for each performer using their saved reference voice, verifies each recording, and only then mixes them at the same start time. Shorter recordings are padded to the longest, with equal fixed weights to prevent clipping. Source text appears once in synchronization data, with separate timing and verification records for each performer. Current support covers unison of the same words; overlapping speech with different words or staggered entrances is not yet represented.

An empty model delivery object (`{}`, `{"direction":""}`, or `{"evidence":[]}`) is completed locally to `{"direction":"","evidence":[]}`. This supplies only omitted empty defaults; it changes no text, evidence, speaker, or performance direction and avoids an unnecessary model repair. Missing delivery objects, invalid types, unknown partial fields, and nonempty directions without evidence still fail. Raw model responses remain unchanged in the cache, and published analyses retain the complete required schema. The same rule applies to proposed semantic corrections before the three reviewers evaluate them.

One narrow recovery restores a model's copied repeated-letter count from the authoritative EPUB source. The entire unit must have the identical sequence of run characters. Each altered run must be alphabetic, both counts must be 2–32, at least one must be 3 or more, and the difference must be at most 8; at most 16 runs may change per unit. No segment boundary may fall inside an altered run. Accepted restoration preserves the exact canonical source, keeps speaker assignments, and flags affected segments for review with `source_restoration` metadata recording counts, offsets, and fingerprints. Other differences still fail validation; this is not fuzzy spelling repair.

If initial passage tagging receives an explicit model-output truncation error, it can divide that batch into halves, with at most two split levels and four final groups. Splitting uses whole existing target units; it does not subdivide a unit further. Each child retains the original neighboring source and the other target units as read-only context. Children run sequentially inside the existing worker, preserving the configured concurrency ceiling. Serial attribution carries preceding accepted child segments forward as context; independent workers do not share those annotations. Every child is validated, and the combined result is revalidated for exact original source coverage and order, retaining any repeated-letter restoration audit records. Other failures do not trigger subdivision. Truncation of a single-unit request or at the depth limit stops analysis.

Normal tagging prompts, payloads, and cache keys remain unchanged. Validated child responses are cached separately. Exact-request truncation hints allow resumption without repeating an already observed failed parent request; a successful parent cache takes precedence. Hints never supply accepted annotations. `provenance.attribution_subdivisions` records failed parent requests, validated children, request fingerprints, and whether a truncation hint was replayed or imported from a verified prior observation.

## Character identity consensus

Identity consensus is enabled by default through `analysis.identity_consensus = true`. After discovery and reconciliation, an additional source scan looks for both error directions: two distinct people being represented as one character, and a returning person being represented by multiple characters. Shared or similar names are grounds to investigate, never sufficient evidence to merge. The audit examines aliases, descriptions, relationships, and source context; a proposed split includes the concrete replacement character, role, and voice records before it is judged.

When the selected series registry contains prior books, a dedicated continuity scan must account for every character in the current cast with an explicit `new`, `match`, or `uncertain` decision. This check runs even when discovery omitted all optional series matches. A proposed match needs current-source evidence and receives the same three-way review before identity reuse; a shared or similar name alone cannot establish continuity. An uncertain decision remains reviewable. No request is needed when the current cast is empty.

Each concrete identity decision receives three separately cached model verdicts. Reviewers see the proposed decision and source evidence, without seeing each other's responses. Only three approvals establish consensus; disagreement or an uncertain answer leaves a review item. Cases without enough evidence for a concrete decision remain unresolved. Proposed merges are not applied while consensus is pending. Accepted changes happen before passage attribution, so all attribution workers receive the same corrected cast. Proposals and verdicts, including source evidence and cast fingerprints, are retained in `provenance.identity_consensus`.

Series continuity has a fixed limit of two rounds. One additional continuity scan reconsiders every current character if a first-round match lacks three valid approvals, or if the first scanner marks a character `uncertain` and identifies at least one prior-character candidate. A rejection, uncertain verdict, or error vote therefore triggers the recheck, as does an unresolved candidate that never reached a jury. A `new` decision or `uncertain` decision without prior candidates does not trigger it. Prior scan and jury feedback is supplied to the second scanner as untrusted leads to verify against the source.

The second scan and its reviewers distinguish recurrence of a specific established performer from unresolved questions about older identity links or the nature of that performer. For example, a recurring protocol voice can be considered for reuse while its independence from the narrator remains an open question. Existing uncertainty flags remain visible, and current-source evidence is still required. Every match proposed in the second round receives three fresh blind jury reviews, including matches approved in the first round; reviewers do not see prior verdicts or peer responses. Only the final round's accepted matches enable reuse. Anything still unresolved remains visible in the draft.

The final scan and cases are retained in `provenance.identity_consensus`, with the first round's cast, issues, scan, and verdicts in its `series_review_history`. The broader retry trigger and second-round guidance preserve the first-round scan and jury request bytes and cache keys; discovery and within-book identity caches also remain compatible. Second-round requests use their own guidance and round discriminator. All cache reuse still requires unchanged prompts and inputs. The tool stops after the second round even if uncertainty remains; there is no setting to increase this limit.

Responses with invalid quotations or invalid references undergo bounded repair. If a proposed patch or verdict still fails validation, the case stays unresolved and its change is withheld. A failed source scan or provider request stops analysis without publishing a successful result. Valid cached work remains available to resume.

All three calls use the configured model, so their errors can be correlated. A scan can also miss a candidate. Consensus reduces reliance on one answer; it does not certify identity accuracy. Directory analysis retains outstanding uncertainties as advisory notes. Those notes do not block export or rendering; genuinely unassigned speech still needs a performer.

To explicitly disable the extra identity calls, set `identity_consensus = false` under `[analysis]`. This restores the earlier discovery/reconciliation behavior: proposed merges can be applied to the draft and are flagged for review, with no independent identity audit, exhaustive series continuity scan, or three-way verdict. Series matching then depends on optional proposals made during discovery/reconciliation and the deterministic matching safeguards. No environment variable or additional CLI flag controls this setting.

## Passage attribution consensus

Set `analysis.attribution_review = true` to review the completed script during `analyze`. This extra pass is disabled by default because it adds model calls. It checks who performs each passage, including speech separated from narrative tags and actions, continuing and nested quotations, directly rendered thoughts, narrator frames, and individual versus collective speech. It uses the existing cast and the analysis's performance policy; it cannot add identities, change voices, or rewrite source text.

The scanner proposes complete replacements for source blocks needing correction. Three separately cached reviewers inspect every proposed replacement against the source without seeing peer votes or scanner notes. Each block requires three approvals and exact source citations. A rejection, uncertainty, or invalid jury response leaves the original assignment intact and adds an unresolved review item. A confirmed simultaneous assignment can resolve the corresponding unassigned-speaker flag; unrelated unresolved issues remain, and explicit human corrections cannot be overwritten. Bracketed delivery is reapplied after accepted changes. Source, cast, original and proposed segments, verdicts, and fingerprints remain available in `provenance.attribution_review`; repeat reviews preserve earlier audit provenance. The report distinguishes rejected corrections from the actual number of unassigned passages remaining.

Every `analyze` run also performs mandatory speaker completion after this optional scan, including when `analysis.attribution_review = false`. It enumerates all unassigned spoken spans rather than relying on a scanner to notice them. Three reviewers independently assign the surrounding complete source block without seeing each other's answers. Only source spans with unanimous resolved decisions are updated; disagreement elsewhere in the block cannot erase an agreed assignment. Up to three rounds expand source context, with the last round adding relevant cast evidence from other chapters, while retaining the same evidence requirements. Exact source coverage, existing cast references, quotations, and protected human assignments are validated before changes are applied. Existing local delivery remains unchanged. Completion cannot invent a missing character or silently use the narrator. Per-case decisions, source/cast/script fingerprints, reviewer results, and remaining counts are retained in `provenance.speaker_completion` and resumable cache audit files.

After automatic speaker assignment, `analyze` and `resolve-speakers` show each remaining passage separately from its wrapped, paragraph-spaced context. Surrounding text is labeled with its assigned voice names, splitting paragraphs where the voice changes and marking unresolved context as **Unassigned**. Color terminals show context in cyan and the passage to assign in bold yellow; plain output keeps the target heading and `>` marker. Redirected output, `TERM=dumb`, and `NO_COLOR` disable color. The menu lists up to three likely speakers, best guess first, with short explanations. Press **Enter** to accept that first suggestion, choose a number, or use **O (Other)** to open the full cast; **B** returns to the shortlist. Comma-separated dialogue roles are only for characters saying the same words together. **Q** or end of input preserves earlier decisions and exits with status `2` while speakers remain missing. Suggestions do not change assignments until selected. Redirected standard input uses the controlling terminal when one is available. With no terminal, the command saves progress and explains how to resume. These choices are recorded as source-bound user decisions. Unassigned speech blocks audio generation; advisory notes do not.

The shortlist uses existing candidate speakers, the latest automatic reviewers, and nearby dialogue and cast references. When `[jev] enabled = true` in the same global config, JEV also evaluates the plausible candidates using the target passage, surrounding source text, and relevant cast details. This is a separate speaker-ranking judgment, not the speech-verification prompt; its scores rank suggestions rather than authorize automatic assignments. An unavailable evaluator falls back to the local shortlist, and unlikely characters remain accessible through Other. JEV makes no ranking requests when there are no missing speakers or no interactive terminal.

To resume a saved book, rerun the original `analyze` command with the same output directory, or pass that directory directly:

```sh
.venv-renderer/bin/epub-audiobook analyze runs/book-06
```

Each explicit rerun automatically reconsiders every remaining unassigned block before asking for manual choices, including cases whose earlier checks ended without an answer. Three independent reviewers reconsider narration versus speech and split sequential speakers or narration into separate segments when supported by the source. For unassigned spans, retry requests omit earlier inferred labels and speaker guesses so reviewers start from the source. Existing segment boundaries are not treated as authoritative for unassigned text. Exact text coverage and three-way agreement remain required; unresolved passages are never filled by guessing. Saved cast identities and already-assigned spans are preserved.

A completed but unresolved check gets a fresh attempt on the next invocation. An API interruption within that attempt reuses its validated cached responses when you rerun; unrelated manual decisions do not invalidate the other blocks' request keys. Each invocation still has a bounded automatic review before terminal fallback. A fully assigned book needs no model calls or rewrite. Supplying an EPUB with an existing output requires it to match the saved source; an unrelated book cannot overwrite that directory. Updates preserve voices and refresh linked descendant metadata.

If a speaker-completion response is truncated twice, that reviewer cannot approve an assignment. The affected block stops automatic retries and is presented for your choice; completed agreements for other passages are retained. Repeated truncation no longer prevents the speaker prompt. Transport failures still stop the run with its validated requests cached for resumption.

To resolve several existing book directories in one command:

```sh
python3.11 -m epub_audiobook resolve-speakers runs/book-01 runs/book-02 runs/book-03
```

This uses the same global configuration and cached model requests. It stages and validates the revised analyses, chapter scripts, review data, and cumulative series history, backs up the original managed files under `.analysis-revisions/` in the first affected book, and refreshes descendant continuation hashes. Linked descendants in the same parent folders are discovered automatically; include descendants stored elsewhere explicitly so their links can also be refreshed. Cast identities, saved voice references, per-book aliases, and rendered audio are retained. Existing rendered checkpoints whose source bundle changed must be rendered again before assembly; unchanged verified clip requests can still be reused. Do not run this while rendering an affected book. The command reports the backup location and returns status `2` when unassigned speech remains, reporting that speaker selection is incomplete.

Review an existing analysis without rediscovering characters or updating its series registry:

```sh
python3.11 -m epub_audiobook review-attribution runs/book/analysis.json \
  -o book.attribution-reviewed.json
```

This explicit command runs the review regardless of the `analysis.attribution_review` setting. It validates the input, writes only after the review succeeds, and requires `--force` to replace an existing output. It cannot replace the configured series registry or configuration file. Export the reviewed analysis separately to regenerate its TTS bundle.

`analysis.attribution_review_chunk_chars` defaults to 60,000 source characters and accepts integers of at least 256. Batches retain complete source blocks and chapter boundaries, so a single long block can exceed that target. Surrounding context and script/cast metadata add to request size. Optional `analysis.attribution_review_notes` supplies source observations for the scanner to verify; an empty string falls back to `analysis.cast_review_notes`. Notes are leads, not approved corrections. The review retains the policy saved in the analysis being reviewed.

If a semantic review scan exceeds the model's output limit, its target blocks are divided into halves, up to two subdivision levels and four final groups. Child scans run sequentially inside the same worker and retain all source context visible to the original scan. Their validated proposals still receive three blind jury requests per original batch. Ordinary cache entries stay compatible; child responses and separate hints recording prior truncation support resumption. Other errors do not trigger subdivision. Truncation of a single-block scan or at the depth limit stops review. Jury responses are not subdivided; initial passage tagging has the separate unit-based recovery described above.

All reviewers use the configured model, so shared errors remain possible, and the scanner can miss mistakes. Three-way agreement does not guarantee semantic accuracy. Invalid scans or provider failures stop the review without publishing output; valid cached calls remain resumable. Invalid jury responses undergo configured validation repairs, then remain recorded as error votes if still invalid. Outstanding notes remain available for optional corrections and do not block export or rendering once every spoken passage has a voice.

## Optional corrections

`review` prints the current source and cast fingerprints, plus unresolved or flagged segments and issues. Examine the surrounding source in the analysis JSON. Copy both fingerprints into a corrections file:

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

The IDs above illustrate the supplied example. Use IDs from your analysis.

```sh
python3.11 -m epub_audiobook apply runs/book/analysis.json \
  --edits corrections.json -o book.reviewed.json
python3.11 -m epub_audiobook validate book.reviewed.json --ready
python3.11 -m epub_audiobook export book.reviewed.json -o book-bundle
```

Corrections are saved in provenance. The original file stays intact unless you explicitly choose `--force`. A corrections file can also contain a complete replacement `cast` and exact per-block segment replacements; see [FORMAT.md](FORMAT.md). Changes to the cast update every segment's resolved voice reference.

`analyze --edits corrections.json` replays corrections **only when both source and cast fingerprints match**. Cached runs retain the same IDs. An independent analysis may discover a different cast or allocate IDs differently; stale corrections are rejected and must be reviewed against that result. Source- or cast-bound edits are never silently applied to a different identity.

## Performance policies

The defaults use a known thinking character for explicit thoughts, a known letter author for letters, the currently speaking character for nested quoted dialogue, and the relevant narrator for headings. Unclear cases require review. Override selected policy strings in the configuration file; omitted strings keep their defaults:

```toml
[performance_policy]
thoughts = "Use the relevant narrator for internal thoughts."
letters = "Read letters using the author's character voice when identified; otherwise flag for review."
nested_quotations = "Keep the currently speaking character's performed voice."
headings_and_nonstory_text = "Use the applicable narrator."
```

Policies guide annotation. They do not alter the source wording or bypass validation.

## TTS handoff

Directory analysis creates the renderer bundle directly. Its files include:

| File | Purpose |
| --- | --- |
| `manifest.json` | Format/version, book/chapter metadata, readiness, source fingerprint, provenance, filenames, and SHA-256 hashes |
| `cast.json` | Characters, aliases, narrative/dialogue roles, and provider-independent voice profiles |
| `scripts/chapter-0001.jsonl`, … | Chapter scripts with ordered text spans, `voice_id`, `role_id`, local delivery, source ranges, and paragraph breaks |
| `source.json` | Canonical source blocks, EPUB locations, extraction policy, warnings, and exclusions |
| `review.json` | Advisory notes and any unassigned passages; notes do not require approval |
| `analysis.json` | Full validated analysis, including source evidence and audit provenance |
| `series-cast.json` | Cumulative series history for the next book's `--continue-from` |

Bundles use format version 2. Read chapter scripts in the order of `manifest.files.scripts`; each entry gives its display title, source chapter ID, relative path, segment count, and global sequence range (start inclusive, end exclusive). Empty source chapters have an empty file and a zero-length range. Filenames use output positions rather than titles. A source resource containing several numbered story headings is split when intervening prose supports separate chapters; contents lists remain one section. Source chapter IDs and the manifest's original `chapters` metadata remain unchanged, so several script files may reference one source chapter. Check the displayed titles and boundaries before using them for final audiobook chapter markers.

The audio CLI should require the manifest and supported format version, check file hashes and readiness, and map each `voice_id` to one of its engine's voices. The renderer keeps that mapping beneath each book's `voices` directory and searches earlier books through the saved continuation link. Older version-1 bundles have one `script.jsonl`; version-2 bundles contain only the chapter files. Both retain the same segment IDs, global order, source offsets, and voice references. The analyzer emits no engine-specific voice names, SSML, audio parameters, or `.m4b` files.

The separate `export` command is for legacy JSON analyses or a new bundle after optional corrections. It writes the renderer files above, without the directory workflow's `analysis.json` and `series-cast.json`. It publishes the manifest last and refuses existing output directories. An interrupted manual export without a manifest is incomplete; retry using a new directory. `--allow-review` is retained for compatibility but is not required. `readiness.ready` reflects valid structure and assigned spoken passages, not whether advisory notes remain. Export saves unresolved work too, but returns status `2` when speakers are missing.

## Long books, resuming, and cost controls

With identity consensus enabled, the default baseline is `3 × batch_count + 1` model requests: discovery, reconciliation, an identity scan of every batch, and attribution. `analysis.whole_book_discovery = true` sends the complete source for discovery, reconciliation, and one identity scan, making the baseline `batch_count + 3`. Use it only with a model context and `analysis.max_request_chars` large enough for the full source and metadata. Disabling identity consensus reduces those baselines to `2 × batch_count + 1` and `batch_count + 2`, respectively.

`analysis.cast_reconciliation_rounds` accepts integers from `1` to `3` and defaults to `1`. Each extra pass adds one sequential baseline model request. With `R` passes, the baselines above become `3 × batch_count + R` and `batch_count + R + 2` with identity consensus, or `2 × batch_count + R` and `batch_count + R + 1` without it. `plan` includes the configured number of passes; conditional jury calls, validation repairs, and transport retries remain additional.

Identity decisions add conditional requests beyond that baseline: allow three verdict calls per case, including proposed matches to an earlier series character, and one additional preparation call for each proposed split. The number of cases is unknown until discovery and scanning have run. `plan` includes the guaranteed scan overhead in `baseline_model_requests`, reports it separately as `identity_scan_requests`, and explains the additional case-dependent cost. It makes no model requests and does not present the baseline as a total-cost estimate.

With identity consensus enabled and prior series books present, `plan` also budgets one dedicated continuity scan in `baseline_model_requests`, reported as `series_identity_scan_requests_minimum`. This assumes discovery finds at least one character; an empty cast needs no continuity request. Whole-book mode uses one complete-source continuity scan per round and requires enough request budget. Bounded discovery mode also uses the full source if it fits; otherwise continuity is checked in character/context batches, so additional calls depend on the discovered cast and request size. Every resulting proposed match still needs three jury verdicts. If a first-round match lacks unanimous approval or the scanner leaves an uncertain prior-character candidate, the second round adds another scan (or bounded-mode scan batches) and three fresh jury requests for every match it proposes, including previously approved matches. This conditional second round is excluded from the baseline and explained in the plan's note. An empty registry or disabled identity consensus adds no dedicated continuity scan.

Enabling passage attribution review adds one scan request per review batch and three jury requests for each batch that proposes at least one correction. Each jury request contains separate verdicts for every proposed block in that batch. `plan` includes these scans in the baseline and reports `attribution_review_scan_requests`; conditional jury calls and validation repairs are additional. Output-truncated semantic scans can add up to six scan attempts per original batch through bounded subdivision, before cache reuse and validation repairs; the three jury requests per original batch remain unchanged. A different review batch size can change this count independently of initial attribution batches.

Initial passage-tagging subdivision can likewise add up to six attribution requests per original batch in one run: seven including the failed parent, before cache reuse, validation repairs, and transport retries. This conditional recovery is excluded from the baseline request count. It does not add jury requests itself.

Mandatory speaker completion adds three independent assignments per source block containing unassigned speech in each required review round. These conditional requests are additional to the plan's baseline. No completion requests are needed when every spoken span already has an assigned performer. Completion uses at most `min(5, attribution_workers)` concurrent requests across all cases and reviewers.

Audition preparation adds one model request per canonical voice lacking enough usable source speech and a saved creative script, plus bounded validation repairs and transport retries. It uses the same cache, configured provider, and five-request ceiling. These calls prepare text only; local TTS and audio verification happen later.

`analysis.attribution_workers` (1–8; default 1) is the shared concurrency ceiling for attribution and identity-review requests. Reviewing an identity case uses at most `min(3, attribution_workers)` concurrent calls; passage review uses at most `min(5, attribution_workers)`. These phases do not overlap. Setting it to `5` therefore keeps this analyzer at no more than five concurrent requests. Concurrent attribution batches use actual neighboring source context, and outputs remain in source order. Choose a worker count within the provider's concurrency allowance.

The defaults are 12,000 source characters per attribution batch, a 200,000-character maximum serialized request, and one validation-repair attempt. **Characters are not tokens**, and the cast also consumes model context. Select a suitable model and adjust `analysis.chunk_chars` and `analysis.max_request_chars` in the configuration for its limits. There is no automatic cost estimator. Optional `provider.reasoning_effort` is sent only when configured; supported values depend on the model. Reasoning models may consume output tokens before emitting the final JSON.

Use `provider.max_output_tokens` to set an output token cap. It uses `max_completion_tokens` by default; servers requiring the older field can set `provider.token_limit_field = "max_tokens"`. A cap that is too small causes rejected truncated responses and can exhaust the bounded recovery available to tagging and semantic scans. Set `analysis.repair_attempts` from `0` to `5` to bound semantic validation retries after the initial response. The default is `1`; higher values allow more repairs without weakening validation.

Request retries are disabled by default. Set `provider.max_retries` (0–10) to enable bounded retries for HTTP 408, 429, 500, 502, 503, and 504, plus typed transient transport failures such as timeouts, connection resets/refusals, incomplete reads, and temporary DNS failures. Retries occupy the same worker, preserving the configured concurrency ceiling, and do not change cache keys. `provider.retry_initial_delay` (default 5 seconds) grows exponentially with jitter up to `provider.retry_max_delay` (default 60 seconds); both settings must be from 1 to 60 seconds. A valid `Retry-After` delay is respected; if it exceeds the configured maximum, analysis stops for a later resume instead of retrying too early.

Known quota, billing, and authentication failures, explicit nonretryable responses, TLS/certificate errors, invalid URLs, permanent DNS failures, unrecognized transport errors, and invalid model output are not retried by the request layer. Transport classification uses exception types and error numbers, never message text. Retry logs use a generic transport failure message or safe HTTP status, recognized codes, and allowlisted numeric TPM/concurrency headers; they omit raw errors, URLs, partial response bodies, and TLS diagnostics.

Successfully validated responses are cached under `.epub-audiobook-cache` beside the output directory (for example, `runs/.epub-audiobook-cache` for `-o runs/book-01`), or under the global `analysis.cache_dir` override. A relative configured cache path is resolved against the configuration file's directory. For directory analysis, the cache cannot be inside either the current output or the previous output selected by `--continue-from`. Cache keys include the exact prompt, input, model, endpoint, and relevant options. Repeating the same command reuses successful work after a failure. Changing model/prompt/cast/policy inputs invalidates affected requests. Cache entries contain book text and derived annotations, never API keys.

`analyze`, `review-attribution`, and `resolve-speakers` print the resolved request-cache path when model work begins and repeat it with recovery instructions after a failure or interruption. After a timeout or other error, keep that cache and rerun the same command with the same input, output, and configuration once the cause is resolved. Previously completed responses are loaded and revalidated locally, so discovery, reconciliation, and review stage lines appear again even when no new API call is needed. `Cache reuse` messages count validated responses reused in the current invocation. Identity juries also report completed reviewer responses out of their total: 28 identity matches require 84 responses, including any restored from cache. The incomplete request runs again; a response that was never received cannot be recovered from the provider. A failed analysis does not publish a completed book directory.

Resuming the same book needs no resume flag and does not require adding `--continue-from`. That option supplies an earlier book's series context: if the interrupted command already used it, keep the same parent directory. Preserve the earlier book and its continuation chain unchanged. Changing output location can select a different default cache or, for the first book, a different series identity. Timeout, API-key, and transport-retry settings can be changed without invalidating completed response caches; changing model, prompt, cast, policy, or other request inputs can require new calls.

Invalid model responses are retained in the cache's `failed/` subdirectory for diagnosis and are never accepted as successful results. When resuming the exact same request, the latest failed JSON is revalidated under current rules. If it remains invalid, a new model request receives fresh validation feedback and, when it fits, the previous response. Corrupt, oversized, mismatched, or now-valid failed files are ignored. This repair resume stays within the existing per-run call and input-size limits; it adds no CLI flags.

Attribution repairs explicitly request the complete original batch, including unaffected units, and include its ordered unit IDs when the request budget permits. Evidence feedback asks for short, sufficient quotations copied verbatim from the referenced visible block. The repair request drops redundant hints or the previous response if needed to fit the budget; it never trims source data or accepts a partial answer. These repair instructions and empty-delivery defaults leave normal prompt and request cache identities unchanged.

`Repairing validation` means a proposed model response was rejected and a bounded retry is underway. A later successful result has passed validation; these messages are distinct from the final review-item count, which includes unresolved identity, continuity, and attribution decisions. Extraction notices about an SVG-only cover page or DOM text order are informational. A fatal validation error still stops the run. Advisory notes do not make a bundle a draft or require a review step; missing speakers require a selection before those passages can render.

Citation diagnostics identify the failing field and evidence index, with bounded record and source-block IDs where available. They distinguish an unknown block ID from a quotation that is not an exact substring, and report when additional invalid locations were omitted from the feedback budget. Repairs must preserve the source's exact spelling, case, punctuation, whitespace, and block-ID leading zeros.

For a book with known attribution pitfalls, optional `analysis.cast_review_notes` can contain source observations and block references to check during cast reconciliation. These are claims for the model to verify, not approved assignments. Whole-book mode reviews the complete source again during reconciliation to check for missing speakers. The notes are retained in analysis provenance.

Additional cast reconciliation passes receive the preceding accepted cast, source context, and pending identity merges. This lets a later pass extend earlier corrections while checking that solo speakers, unison roles, and voice scopes remain complete. Identity review starts only after all configured passes finish. Additional passes do not guarantee completeness or resolve an identity disagreement without the required review.

Optional `analysis.cast_reconciliation_notes` is a list of strings with at most `cast_reconciliation_rounds` entries. The first entry applies only to pass 1, the second to pass 2, and the third to pass 3. These observations supplement `cast_review_notes` for the selected pass and must be checked against the source; they do not approve an identity change or bypass the three reviewers. For example, with three passes, `cast_reconciliation_notes = ["", "", "Check the shared words at b000818 against their explicit speaker tag."]` supplies a lead only to the third pass.

The default discovery and first reconciliation request inputs and cache keys remain unchanged. An omitted list, missing entry, or empty string adds nothing to that pass's prompt or payload. Later passes use fresh allocated IDs and their own cache entries. Changing shared review notes or other request inputs still invalidates affected cached requests; adding only third-pass notes preserves the unchanged first two requests.

## Bracketed delivery

Square-bracket text such as `[Fireball]` receives a separate delivery cue automatically, including inside dialogue. Speaker and voice selection stay intact. The script retains exact source text and also includes `spoken_text` without bracket delimiters for synthesis. Configure the suggested delivery with `performance_policy.bracketed_text`; the audio-generation tool decides how to realize the contrast. Balanced nested and multi-paragraph groups are supported; unmatched delimiters are flagged for review. See [the format contract](FORMAT.md#bracketed-delivery).

## Reusing a series cast

Every directory analysis saves a cumulative series cast automatically. To continue a series, pass the preceding book's output directory with `--continue-from`. That directory stays unchanged; the new output receives the full prior history plus the new book. Analyze books in order and point each run at the latest completed output, so returning characters can retain their established voice IDs even after an absence of several books. An unrelated book simply omits `--continue-from`.

```sh
python3.11 -m epub_audiobook analyze book-one.epub -o runs/book-01
python3.11 -m epub_audiobook analyze book-two.epub \
  -o runs/book-02 --continue-from runs/book-01
python3.11 -m epub_audiobook analyze book-three.epub \
  -o runs/book-03 --continue-from runs/book-02

# Inspect the next run without calls or writes.
python3.11 -m epub_audiobook plan book-four.epub \
  -o runs/book-04 --continue-from runs/book-03
```

The new output also records a relative link to the preceding book directory, bound to its series and book identity. This lets the renderer discover voice recordings created in an earlier book even after later books have already been analyzed. Keep the linked book directories available; moving the whole series together preserves their relative links.

Voice lookup starts in the current book, then follows `--continue-from` parents in order until it finds that voice. For example, book 3 searches book 3, book 2, then book 1. A voice regenerated in book 2 is saved in book 2 and is inherited by book 3; book 1 and branches continued directly from book 1 keep their existing voice. Missing voices, edited casting drafts, and replacements are written only to the current book. No per-book configuration or advance voice creation is required. See [RENDERER.md](RENDERER.md) for casting and rendering commands.

Author afterwords and “From the Author” sections have a separate author narration identity. New analyses reuse the nearest prior author voice when the book metadata names the same unambiguous author and the narration scope explicitly belongs to that author. The cast menu names it, for example, **Nathan Thompson (author)**. This does not merge the author with a fictional narrator, guest contributor, or editor. Existing books with older, separate author voice IDs can also inherit missing author recordings through their continuation chain; existing local recordings and failed-regeneration markers retain precedence, and explicit book aliases override automatic author links.

For a new author voice whose presentation is unspecified, the analysis and prompt-refresh guidance uses the author's name to propose a tentative masculine or feminine casting choice when reasonably clear. Initials or ambiguous names can remain unspecified. This is labeled as casting, not a claim about the author's gender or actual voice, and existing deliberate choices take priority. `audiobook-render cast BOOK --action refresh-prompts` applies the current prompt guidance to saved casting drafts without regenerating audio.

Manual voice sharing is also book-specific. `audiobook-render voice-aliases BOOK --set SOURCE_VOICE_ID TARGET_VOICE_ID` saves the mapping in that book's `voice-aliases.json`; later books inherit it through the continuation chain. The nearest explicit mapping wins, so a later book can override it without changing earlier books or sibling branches. `--remove SOURCE_VOICE_ID` removes a local override and restores inherited or automatic behavior. Running the command without edits lists the effective mappings. Global configuration must not contain `voices.aliases`; no reanalysis is needed to add or change a book's voice mappings.

The input directory must contain the `series-cast.json` produced by directory analysis. Merely choosing a similar output directory name never selects another book's cast. Use the same global configuration for every book. Directory mode rejects `--series` and `analysis.series_registry`; choose prior context explicitly with `--continue-from` instead.

For compatibility, `analyze -o book.analysis.json` still writes a single analysis file. This legacy mode accepts an explicit registry file through `--series` or the deprecated `analysis.series_registry` setting; the CLI option takes precedence. It needs a separate `export` command for renderer input. New directory runs do not need either setting or a `series-import` step.

```sh
# Legacy JSON/registry workflow only.
python3.11 -m epub_audiobook series-import book-one-analysis.json \
  --series casts/my-series.json -o book-one-linked.json
python3.11 -m epub_audiobook analyze book-two.epub \
  --series casts/my-series.json -o book-two-analysis.json
```

`series-import` preserves the original analysis. Its optional output copy uses stable series character and voice IDs, which the audio renderer can map to its established engine voices. This offline import does not perform a fresh identity scan or three-way model review; it registers the existing analysis and its review state. Reimporting preserves those IDs and casting while refreshing applicable prior review concerns, including when the input is already linked. Legacy `analyze --series` creates an absent registry after successful analysis, and `plan --series` inspects its book and character counts without writes or model calls.

The model must propose continuity using exact evidence from the current book and an explanation beyond a shared name. With identity consensus enabled, three source-supported approvals can establish a returning character even when their name has changed, they were previously unnamed, or several earlier characters share an alias. The linker validates the full audit against the current source, registry, cast, and concrete proposal before reusing that identity; a model-supplied approval flag cannot enable reuse. Rejected or uncertain cases cannot fall back to name matching, and multiple current characters claiming the same prior identity remain separate and flagged for review.

Without that validated consensus, `series-import` and analysis with `identity_consensus = false` retain the stricter matching rule: a source-supported context proposal must point to the sole prior character sharing an exact normalized name or alias. Missing or ambiguous alias matches remain reviewable. Importing an existing analysis does not perform fresh model review or infer a new match from saved verdicts; an already linked analysis retains its established series IDs.

Prior casting details or identity concerns do not automatically allocate new IDs when current-source evidence supports an accepted match: relevant earlier issues carry forward unresolved with their original book and issue provenance. Reuse does not approve those issues, and readiness remains false while they remain unresolved. Identified narrators can reuse their own established narration voice independently of their dialogue voice. An unidentified main narrator can retain the series' default narration casting; interpretation concerns carry forward without changing that voice. Separate POV and afterword roles keep their own scopes, and conflicting casting relationships stay reviewable.

Registry snapshots retain book fingerprints, cast profiles, and unresolved cast issues. Earlier quotations stay attached to their original book. Reused voice traits appear as casting directions in the current analysis, with `series_origin` provenance; they are never presented as evidence from the new book. Registry changes affect request inputs and therefore cache reuse. Directory analysis saves a new cumulative registry in its own output and never updates the prior output. In the legacy JSON workflow, successful analysis updates the selected registry atomically; concurrent updates are rejected instead of overwriting each other. A registry is production reference material, not human approval of its contents.

With identity consensus enabled and prior books present, analysis performs a dedicated source-based continuity scan covering every current character, then requires three independent approvals before a proposed earlier-series match can reuse the established identity. Optional discovery proposals cannot substitute for this complete scan. These scans and continuity cases contribute to the request costs described above. Names and aliases help identify candidates; validated contextual consensus can establish continuity without a unique alias match. Exact source evidence, audit bindings, and duplicate-claim safeguards still apply. Disabling identity consensus skips the dedicated scan and jury and retains the strict alias-based matching path described above.

## Extraction scope

EPUB reading order follows the package spine, as defined by the [EPUB specification](https://www.w3.org/TR/epub-33/#sec-spine-elem). XHTML is parsed without running scripts or fetching network resources. EPUB 2 and EPUB 3 package/spine structures are supported when their content is parseable XHTML.

- The canonical text collapses display whitespace. Inline text and punctuation are retained. Offsets refer to this canonical text, not original XML bytes.
- Optional `inline_spans` preserve emphasis and strong-text cues from semantic tags, explicit inline styles, and simple local stylesheet class rules. Complex or conflicting CSS is not inferred. These cues can help interpretation but do not establish a speaker. Empty paragraphs add `break_before: "blank_paragraph"` to the next text block in the same resource.
- Navigation and non-linear spine items are excluded by default and recorded. Set `extraction.include_auxiliary = true` to include them when present in the spine. Other non-spine content is audited but not appended automatically.
- Footnotes inside selected documents remain in DOM order. Separate footnote documents need deliberate inclusion. The tool does not relocate notes to their references.
- Fixed-layout reading order, CSS-hidden text, visual poetry/layout, images, SVG/MathML narration, OCR, and pronunciation normalization are not handled as a renderer would. Warnings and exclusions must be inspected for such books.
- Primary unsupported resources without a usable XHTML fallback, encrypted primary content, malformed XML, unsafe archives, and empty extraction fail explicitly. No DRM removal is attempted.
- Chapter entries currently correspond to selected spine resources; a publisher's chapter spanning several files or several chapters inside one file needs a later structural refinement before final audiobook chaptering.

These limitations are separate from speaker-attribution uncertainty. Review extraction before spending model calls on unusual publications.

## Offline example and tests

`examples/last-lantern/` contains an original fictional EPUB and **hand-authored annotations**, not model-generated results. Mara is both a narrator and a character using one voice; Lena has a distinct voice. One line has no explicit speaker. The included correction selects Mara as a production decision and produces a ready example bundle.

```sh
python3.11 -m epub_audiobook review examples/last-lantern/analysis.json
python3.11 -m epub_audiobook validate examples/last-lantern/reviewed.json --ready
python3.11 -m unittest discover -s tests -q

# Rebuild the example in a new directory without any API calls.
python3.11 examples/build_example.py /tmp/new-audiobook-example
```

Tests use synthetic EPUBs and mocked/deterministic model responses. They cover EPUB order/fidelity, malformed archives, API failures and credential handling, cross-book context, exact coverage, cached resumption, narrator/character voice sharing, ambiguous attribution, identity consensus and disagreement, source-bound corrections, and export integrity. Separate live-model work has exercised a full book, selected excerpts, focused attribution regression cases, and the four identity scenarios above. A representative evaluation corpus and calibrated attribution or identity accuracy remain future work.

To run the small identity regression suite against your configured model, choose a new report path:

```sh
python3.11 scripts/validate_identity.py --config epub-audiobook.toml \
  -o runs/identity-check/results.json
```

This checks the identity audit against deliberately incorrect or potentially confusing cast records; it does not evaluate full-book discovery or passage attribution. Reports retain proposals and verdicts. Add `--check-fixtures` for local fixture validation without API calls, or `--case same_name_distinct_people` to run one scenario. `--cache-dir` can reuse independently validated model responses within the report directory. Invalid split proposals remain unresolved and cannot change the cast.

Exit codes: `0` success; `1` invalid input, contract, I/O, or API failure; `2` saved analysis with unassigned spoken passages after `analyze`, `resolve-speakers`, or `export`, or when `validate --ready` detects missing speakers; `130` interrupted analysis. Advisory notes alone do not change a successful exit status.
