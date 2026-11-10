# epub-tts-narrator

Turn an EPUB into a **verified multi-voice audiobook**. The analyzer reads a book, casts characters to voices, and writes a text-preserving performance script; the renderer synthesizes, verifies, and assembles the audio into a chapterized `.m4b`. Analysis output is independent of the TTS engine.

## Requirements

- **Analysis:** Python 3.11+, no third-party dependencies. An OpenAI-compatible model endpoint for the LLM stages (local or hosted).
- **Rendering:** Python 3.12 (via uv), FFmpeg + FFprobe, and the renderer dependency set — see [RENDERER.md](RENDERER.md#install). Local synthesis uses a vendored Drama 3 MLX port (Apple Silicon); remote rendering uses a Runpod GPU worker.

## Install

```sh
git clone https://github.com/taxilian/epub-tts-narrator.git
cd epub-tts-narrator

# Analyzer (zero runtime deps)
python3.11 -m venv .venv
.venv/bin/pip install -e .

# Renderer (separate environment; details in RENDERER.md)
uv venv --python 3.12 .venv-renderer
uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt
uv pip install --python .venv-renderer/bin/python -e .
```

Two commands come from one package: `epub-tts-narrator` (analysis; `epub-audiobook` remains a legacy alias) and `audiobook-render` (rendering). The module forms `python -m epub_tts_narrator` and `python -m epub_tts_narrator.renderer` work without installing.

## Configure

```sh
cp epub-tts-narrator.example.toml epub-tts-narrator.toml
```

Set `provider.model` (and `provider.api_key` if the endpoint requires one) in that file. Keys are read from the file, never from environment variables, and never written into analysis files or caches. The template is tracked; the real config is git-ignored. Both CLIs load `epub-tts-narrator.toml` from the current directory, or `--config PATH`; the legacy filename `epub-audiobook.toml` still works when the new one is absent — the two files are never merged.

For a local OpenAI-compatible server:

```toml
[provider]
model = "YOUR_LOCAL_MODEL"
base_url = "http://127.0.0.1:8000/v1"
json_mode = false
```

Remote endpoints require HTTPS; HTTP is loopback-only; redirects are refused.

## Use

From this checkout, `just` provides short commands without activating a virtual environment or remembering binary paths. Install it once with `brew install just` if needed; the recipes use the existing `.venv-renderer` environment described above.

```sh
just                         # list available commands
just analyze "path/to/book.epub"
just plan "path/to/book.epub"
just cast runs/book-01
just render runs/book-01 --chapters 3 --chapters-only
just render runs/book-01 runs/book-02   # queue several books through one process
just assemble runs/book-01
just runpod-cleanup          # delete every Runpod Pod this deployment created (asks first)
just test
just test -p test_epub.py     # optional unittest discovery arguments
```

Arguments are forwarded unchanged, including quoted paths containing spaces. Recipes do not install dependencies automatically. The underlying commands remain available directly:

```sh
# Estimate batches and baseline model requests; no model calls, no writes.
epub-tts-narrator plan book.epub

# Analyze into ./book.audiobook/ — a renderer-ready bundle.
epub-tts-narrator analyze book.epub

# Continue a series; the earlier output directory supplies the cast.
epub-tts-narrator analyze book-two.epub -o runs/book-02 --continue-from runs/book-01

# A serial got new chapters or revisions: analyze only the new and edited text
# into the same book, report new characters, and keep unchanged audio reusable.
epub-tts-narrator analyze serial-latest.epub -o runs/serial

# Refresh existing voice profiles in a completed book directory under the
# current profile guidance; only each voice's description, traits, and
# speechTraits change, atomically with its series continuation metadata.
epub-tts-narrator refresh-voice-profiles runs/book-01 --workers 1
```

Re-check a finished book's riskiest speaker assignments (context-based dialogue in
multi-speaker scenes) with the semantic review engine, then decide what to save:

```sh
# Fresh audit; a terminal offers Accept all, Edit-review, or Leave unchanged.
epub-tts-narrator verify-attribution runs/book-01

# Explicit noninteractive modes: report only, apply all, or select each finding.
epub-tts-narrator verify-attribution runs/book-01 --chapter 5 \
  --instructions "the twins finish each other's sentences: alternate them, tags win" \
  --interactive
epub-tts-narrator verify-attribution runs/book-01 --apply
```

Every ordinary invocation performs a fresh audit in a new run-local request cache; a completed earlier audit is never silently replayed. After an interruption, the printed `Fresh verification run: RUN_ID` can be used as `--resume RUN_ID` to reuse compatible validated responses from that explicitly selected run. Without `--resume`, even an unchanged book is checked again. Non-TTY invocation without `--apply` or `--interactive` remains report-only and never applies findings.

Edit-review redisplays each finding's text, original/proposed speakers, and evidence at its decision point. You can accept the proposal, leave it unchanged, finish, or open a staged manual edit. Manual editing can choose any existing-cast speaker (including one outside both proposals), change the delivery direction within 240 characters, and set the review flag; explicit confirmation is required, and cancel or EOF discards the unconfirmed staged edit. Publication preserves source text, performance controls, cast/voice assets, and descendant bindings. The final summary distinguishes applied findings from those left unchanged, breaks down accepted proposals versus manual edits, and names the backup/revision journal whenever publication occurs.

Browse and correct a finished book by hand with `review-ui` (or `just review-ui`):

```sh
epub-tts-narrator review-ui runs/book-01
```

It opens the book's rendered chapter list — the same chapters the exported scripts and final M4B use, so one EPUB file holding several story chapters splits at its chapter headings — shows each passage with its reader in a distinct per-reader color, and on Enter reveals the passage's full metadata with edits: reassign any existing cast member (multi-select composes simultaneous speech), mark a passage unresolved, edit its delivery direction, or toggle the review flag. Saving publishes the same receipt-bound attribution revisions as `verify-attribution --apply`, with backups and series descendant rebinding; the published cast never changes and no model calls are made.

For a passage that has been rendered, the metadata dialog also plays its audio (`P`, any key stops) and `T` lists every render take with its verification outcome; there you can play any take and, after confirmation, accept any eligible one — including a failed take, which publishes under the audited `manual_take_selection` basis with the original failure report retained and the previous acceptance archived. Accepting a different take supersedes the chapter checkpoint; rerun `audiobook-render render BOOK --chapters-only N` and `audiobook-render assemble BOOK` to rebuild the chapter and audiobook from cached clips. Playback uses `[casting] player` (default `/usr/bin/afplay`).

Chapters whose render failed (`ChaptersRenderFailed`) are marked `FAILED` in the chapter list with the saved error; `F` jumps to the first failed passage when the renderer recorded exact segment linkage, and failed passages say no audio exists instead of offering playback. Editing a failed chapter still saves the usual attribution revision, but the recorded failure remains until a successful rerender.

Analysis is resumable: after a failure, rerun the same command. Validated model responses and journaled interactive speaker choices are reused. The output directory contains a `.analysis-pending/` workspace until publication, so an interrupted run resumes completed stages rather than starting over. Unassigned speech requires an explicit speaker choice and blocks rendering; advisory notes do not.

`refresh-voice-profiles` never touches source text, segments, attribution, characters, roles, voice identity or order, audition scripts, reference recordings, or render caches. It rebinds the hidden `cast_fingerprint` records (analysis provenance and the book-local `audition-scripts.json`) so casting keeps working, backs up the originals, and refreshes descendant continuation links in the same transaction. The next render picks up the new profiles automatically; rerunning after everything is current makes no changes.

Before model work, refresh durably marks all profiles invalid and checkpoints each prepared proposal. Rendering is blocked for that book and its descendants until publication completes. Rerun the same command after a failure to resume prepared work without repeating completed model calls; reference recordings stay intact. Pending state is cleared only after the published profiles and bindings are complete.

Cast and render (see [RENDERER.md](RENDERER.md) for the full workflow):

```sh
audiobook-render cast book.audiobook        # create/audition voices
audiobook-render render book.audiobook      # synthesize + verify chapters
audiobook-render assemble book.audiobook    # package the .m4b
```

`render` also queues several books through one process — `audiobook-render render runs/book-01 runs/book-02`. Books use the configured worker pool, admitting the next book when earlier books leave synthesis slots free, including during final assembly. One shared pool bounds concurrency and loads models once. A configured `[remote_chapters].url` instead uses that chapter service with the same ordered admission policy. See [multi-book render queue](RENDERER.md#multi-book-render-queue).

`audiobook-render doctor` checks the local setup (models, FFmpeg, verification stack) before a long render.

`audiobook-render compare-performance BOOK --chapter N --directory COMPARISON --local` renders an isolated legacy/expressive chapter comparison using frozen copies of the same selected voices. Open its `index.html` to listen and inspect changed controls. Existing rendering remains unchanged; see [chapter comparison](RENDERER.md#compare-legacy-and-expressive-chapter-generation).

Optional [audio censorship](RENDERER.md#optional-audio-censorship) preserves a short opening sound and silences the rest of configured vulgar words without changing the dialogue's timing. Use `render BOOK --censor` for new output, or `assemble BOOK --censor --chapters 13 --chapters-only` to rebuild a saved chapter from its original verified clips. `--no-censor` overrides the global setting for an uncensored output. The feature is off by default; saved speech and voices remain unchanged.

Analysis designs the whole voice cast and assigns speakers using compact, source-linked annotations rather than copying the book into model responses. Attribution can propose targeted spoken-spelling edits and local delivery, emotion, or vocalization controls where the passage calls for them. Canonical text remains unchanged; semantic changes receive a focused review. There is no separate whole-book spelling scan, pronunciation discovery, phoneme suggestion, or name-annotation stage. General pronunciation overrides remain optional explicit edits afterward; see [manual pronunciation overrides](RENDERER.md#pronunciation-review). To fix how a name or invented word is read throughout a book (and its series continuations), run `just pronounce BOOK WORD`, optionally limited to some parts of speech with `--as`; see [names and invented words](RENDERER.md#names-and-invented-words-pronounce).

Casting adapts automatically to the book. It groups source into bounded 60,000-character regions, runs those independent regions concurrently up to `analysis.attribution_workers`, subdivides only when request or response capacity requires it, resolves independent evidence-backed candidates, and separately checks the source for overlooked speakers. Work bounds grow with the actual source units and distinct findings; there is no partition quota to tune for longer books. Exact evidence and independent identity review remain mandatory when identity consensus is enabled. `plan` reports source-derived bounds and conditional work, not a promise of an exact model-call count.

## What analysis produces

The output directory is the renderer bundle: `manifest.json`, `cast.json`, `source.json`, `review.json`, per-chapter scripts in `scripts/chapter-NNNN.jsonl`, plus the full `analysis.json` and a cumulative `series-cast.json` for continuations. The exact contract is [FORMAT.md](FORMAT.md). Every character, role, voice, and passage attribution carries source evidence; unknown speakers stay explicitly unassigned rather than being guessed.

## Verification

Ordinary speech verification checks written-content equality ignoring case, spacing, and punctuation; sound-alike equivalence; optional JEV judgment; and acoustic quality. With analysis-model recovery enabled, broader clip-local repairs can alter phonetic input, delivery, traits, and splitting after two parallel independent approvals. If ordinary recovery is exhausted, an LLM chooses the best physically usable retained full-clip take under an explicit best-effort policy. Its selection is decisive, including when ordinary lexical checks failed; reports preserve the original failure and identify `llm_exhaustion_selection` rather than claim verbatim verification. Corrupt, empty, and effectively silent audio is excluded. Reference creation retains its separate verification requirements. Details and tuning: [RENDERER.md](RENDERER.md).

## Status

A first working implementation: extraction, validation, caching, corrections, and export have offline tests; live-model runs have covered complete books and focused regression cases. Ambiguities surface as review items — structural validation does not prove a speaker assignment is semantically correct.

## Offline example and tests

`examples/last-lantern/` is an original fictional EPUB with hand-authored annotations (not model output); `examples/identity-consensus/` holds the identity-regression fixtures used by `scripts/validate_identity.py --check-fixtures`.

```sh
epub-tts-narrator review examples/last-lantern/analysis.json
epub-tts-narrator validate examples/last-lantern/reviewed.json --ready
python3.11 examples/build_example.py /tmp/new-audiobook-example   # rebuilds the bundle offline
```

Run the full offline unittest suite in `.venv-renderer`. Its test requirements include the renderer runtime, server/storage SDKs, and the worker's real audio decoder; no model downloads or cloud credentials are needed:

```sh
uv venv --python 3.12 .venv-renderer  # first setup only
uv pip install --python .venv-renderer/bin/python -r test.requirements.txt
.venv-renderer/bin/python -m unittest discover -s tests -q
```

For an existing environment, rerun only the install command to update missing dependencies. The full suite should finish without skips on this supported environment. Tests use tracked fixtures or create their own temporary bundles, never saved books under `runs/`.

## Documentation

| Document | Contents |
|---|---|
| [RENDERER.md](RENDERER.md) | Renderer install, config, casting, rendering, verification, caching |
| [REMOTE_RENDERING.md](REMOTE_RENDERING.md) | Chapter servers, remote workers, Runpod deployment |
| [ANALYSIS.md](ANALYSIS.md) | How analysis works: stages, consensus passes, corrections, cost controls |
| [FORMAT.md](FORMAT.md) | The analysis/bundle/script JSON contract (normative) |
| [deploy/runpod/README.md](deploy/runpod/README.md) | Managed GPU pod lifecycle |
| [AGENTS.md](AGENTS.md) | Project knowledge base for agentic work |

## Notes

- Run directories (`runs/`, `dist-runs/`, `*.audiobook/`) and their in-output `.analysis-cache/` can contain full book text; keep them out of version control.
- Exit codes: `0` success; `1` invalid input, contract, I/O, or API failure; `2` saved analysis with unassigned spoken passages (after `analyze`, `resolve-speakers`, or `export`, or when `validate --ready` finds missing speakers), or after `analyze` when the model was unsure about some uses of a word whose pronunciation you defined; `130` interrupted.
