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

```sh
# Estimate batches and baseline model requests; no model calls, no writes.
epub-tts-narrator plan book.epub

# Analyze into ./book.audiobook/ — a renderer-ready bundle.
epub-tts-narrator analyze book.epub

# Continue a series; the earlier output directory supplies the cast.
epub-tts-narrator analyze book-two.epub -o runs/book-02 --continue-from runs/book-01

# Refresh existing voice profiles in a completed book directory under the
# current profile guidance; only each voice's description, traits, and
# speechTraits change, atomically with its series continuation metadata.
epub-tts-narrator refresh-voice-profiles runs/book-01 --workers 1
```

Analysis is resumable: after a failure, rerun the same command — validated model responses are reused from the request cache. Interactive speaker choices are journaled to the request cache as you make them and replayed on a rerun, so an interrupted analysis resumes your decisions instead of re-asking, and the dependent spoken-spelling work then reuses the cache instead of recomputing. If speech remains unassigned after bounded automatic work, the CLI asks you to pick speakers in the terminal and exits with status 2; unassigned speech blocks audio generation, advisory notes do not.

`refresh-voice-profiles` never touches source text, segments, attribution, characters, roles, voice identity or order, audition scripts, reference recordings, or render caches. It rebinds the hidden `cast_fingerprint` records (analysis provenance and the book-local `audition-scripts.json`) so casting keeps working, backs up the originals, and refreshes descendant continuation links in the same transaction. The next render picks up the new profiles automatically; rerunning after everything is current makes no changes.

Before model work, refresh durably marks all profiles invalid and checkpoints each prepared proposal. Rendering is blocked for that book and its descendants until publication completes. Rerun the same command after a failure to resume prepared work without repeating completed model calls; reference recordings stay intact. Pending state is cleared only after the published profiles and bindings are complete.

Cast and render (see [RENDERER.md](RENDERER.md) for the full workflow):

```sh
audiobook-render cast book.audiobook        # create/audition voices
audiobook-render render book.audiobook      # synthesize + verify chapters
audiobook-render assemble book.audiobook    # package the .m4b
```

`audiobook-render doctor` checks the local setup (models, FFmpeg, verification stack) before a long render.

`audiobook-render compare-performance BOOK --chapter N --directory COMPARISON --local` renders an isolated legacy/expressive chapter comparison using frozen copies of the same selected voices. Open its `index.html` to listen and inspect changed controls. Existing rendering remains unchanged; see [chapter comparison](RENDERER.md#compare-legacy-and-expressive-chapter-generation).

Optional [audio censorship](RENDERER.md#optional-audio-censorship) preserves a short opening sound and silences the rest of configured vulgar words without changing the dialogue's timing. Use `render BOOK --censor` for new output, or `assemble BOOK --censor --chapters 13 --chapters-only` to rebuild a saved chapter from its original verified clips. `--no-censor` overrides the global setting for an uncensored output. The feature is off by default; saved speech and voices remain unchanged.

Directory analysis also collects character and place names and prepares CMU Arpabet suggestions for [pronunciation review](RENDERER.md#pronunciation-review). Run `audiobook-render pronunciations BOOK` to listen and accept them, correct a name with a readable hint, or edit CMU directly. Existing books can add candidates and proposals with `epub-tts-narrator discover-pronunciations BOOK`, without reanalysis. Confirmed choices are per-book and inherit recursively through `--continue-from`; guesses never change narration automatically.

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

The test suite (unittest, offline) exercises the renderer verification stack, so install the extras first:

```sh
.venv/bin/pip install -e ".[chapter-server,verification,remote]" packaging soundfile
.venv/bin/python -m unittest discover -s tests -q
```

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

- Run directories (`runs/`, `dist-runs/`, `*.audiobook/`) and the request cache can contain full book text; keep them out of version control.
- Exit codes: `0` success; `1` invalid input, contract, I/O, or API failure; `2` saved analysis with unassigned spoken passages (after `analyze`, `resolve-speakers`, or `export`, or when `validate --ready` finds missing speakers); `130` interrupted.

### Speech-spelling output limits

`Repairing validation` means a complete model answer failed local checks; it is different from `The model output was truncated`, which means the provider exhausted its response budget. Speech-spelling scans recover from that output limit by dividing the target segments in half, at most two levels (four final groups). Child scans retain all original context and run sequentially within the existing worker limit. Every result still needs exact-source validation and the normal independent per-edit review; source text and attribution do not change. Successful requests keep their existing cache keys. Separate truncation hints allow a rerun to skip known failed parent scans and reuse validated children.

If a single-segment scan, a depth-limited child scan, or a jury response still exceeds the output limit, analysis stops without applying unvalidated edits. The error identifies the stage and batch. Check `provider.max_output_tokens` and the provider/model output ceiling; no token limit is raised automatically. Keep the reported resume cache and rerun the same command with the same input, output, and configuration to reuse completed work. Raising the provider token cap changes provider request identity and can invalidate existing request-cache entries, so do not discard the cache or change settings merely to retry the new subdivision recovery.

### Speech-spelling request efficiency

New spelling requests use compact, stage-specific instructions and JSON. Repeated role records are sent once in a role table. A context block references existing segment text only when those segments reconstruct the complete original block exactly and the reference form is smaller; partial, repeated, or truncated context stays verbatim. Reviewers receive proposed-edit segments as targets and all unchanged segments as read-only context, retaining their full text, roles, and nonempty delivery evidence. Established names and all three independent review calls are preserved. Reviewer-specific fields follow the shared request content, allowing prefix caching where supported by the provider.

Validated v1 scan and review caches are checked before making compact v2 requests. Failed responses are never promoted, old truncation hints remain usable, and already-completed normalization does not rescan merely because the wire representation became smaller. Output schemas, validation, retry limits, token ceilings, and acceptance rules are unchanged. These changes reduce redundant input; actual token savings and model accuracy depend on the endpoint and should be measured separately from serialized request size.

### Contextual spelling review

The local spelling heuristic is not a pronunciation dictionary. A plausible unfamiliar normalization now reaches contextual review instead of repeatedly asking the scanner to repair it. Exact source spans, nonoverlap, bounds, protected names/numbers/negation, and complete ordered word coverage remain local requirements. Unknown one-to-one word correspondences cannot absorb unchanged words; mechanically recognized joins/splits retain their existing rules. No book-specific spelling exceptions are added.

Only proposals needing this extra judgment use the extended review contract: all three blind reviewers receive the exact ordered word pairs, and at least two complete reviews must approve every pair as well as the edit. A generic approval without word evidence cannot apply it. Rejection or uncertainty leaves the source spelling unchanged with an explicit unresolved audit/progress message, rather than a scanner repair loop. Reviewer judgments can still share errors; word coverage alone does not prove lexical equivalence. Applied edit format and already-validated ordinary scan/review caches stay unchanged; new contextual reviews have distinct request identities.
