# RENDERER PACKAGE

## OVERVIEW

Flat renderer subpackage: 75 Python modules plus `ENGINE_NOTES.md`.
Resumable, quality-gated audiobook rendering, assembly, casting, and remote execution.

Keep this file current: when modules, entry points, or conventions change, update it in the same commit.

## ENTRY

`audiobook-render` maps to `renderer/cli.py:main`; `python -m epub_tts_narrator.renderer` also works.
Subcommands: `plan`, `cache-report`, `render`, `assemble`, `cleanup`, `voices`, `create-voices`, `import-voice`.
More commands: `cast`, `voice-aliases`, `doctor`, `serve`, `serve-chapters`, `runpod`.
`serve` exposes a Fish-compatible `/v1/tts` service through `server.py`.
`serve-chapters` runs the resumable job server in `chapter_server.py`.
Normal flow: plan, cast or bind voices, render chapters, verify, assemble, publish.

## STRUCTURE

1. Orchestration/config: `workflow.py`, `cli.py`, `config.py`, `book_context.py`, `book_aliases.py`, `common.py`, diagnostics, locks, progress, recovery identity.
2. Speech verification/recovery: `quality.py`, `verification.py`, `speech_resolver.py`, `verification_prompts.py`, `speech_identity.py`, `ordinal_words.py`, coverage, cache, recovery, text, phonetic, ASR, retry modules.
3. Chapter/distributed pipeline: `bundle.py`, `chapter_jobs.py`, `chapter_protocol.py`, `chapter_records.py`, `chapter_executor.py`, `chapter_client.py`, servers, cache, storage, verification, pool.
4. Casting/voice design: `casting.py`, `audition.py`, `voices.py`, `voice_design.py`, provider design modules, prompts, bindings, samples, letter names.
5. TTS engines: `engines.py`, `fish_http.py`, `mlx_audio_engine.py`, local runtime, model setup, Drama 3 model/design/sampling modules.
6. Assembly/publication: `assembly.py`, `publication.py`, `cleanup.py`, paragraphs, paragraph recovery, loudness, pacing.
7. Runpod deployment: `runpod.py`, `runpod_api.py`, `runpod_config.py`.

## WHERE TO LOOK

| Task | Module |
| --- | --- |
| Render chapters and resume work | `workflow.py`, especially `render()` |
| Inspect acceptance or transcript failures | `quality.py`, `verification.py`, `chapter_verification.py` |
| Resolve recurring speech failures | `speech_resolver.py`, `recurring_resolution.py`, `delivery_recovery.py` |
| Configure engines or add a plugin | `engines.py`, `ENGINE_NOTES.md` |
| Cast voices and manage aliases | `casting.py`, `voices.py`, `audition.py`, `voice_bindings.py` |
| Run or debug remote chapter jobs | `chapter_client.py`, `chapter_server.py`, `chapter_protocol.py` |
| Cache, lock, or diagnose a run | `common.py`, `chapter_cache.py`, `run_locks.py`, `diagnostics.py` |
| Assemble and publish an audiobook | `assembly.py`, `publication.py`, `loudness.py`, `pacing.py` |
| Operate Runpod workers | `runpod.py`, `runpod_api.py`, `runpod_config.py` |

## CONVENTIONS

- Renderer imports are direct submodule imports; `__init__.py` exports nothing.
- Heavy ML and HTTP imports stay inside functions, never at module scope.
- Use the separate Python 3.12 uv environment and root `renderer.requirements.txt`.
- FFmpeg and FFprobe are required; paths come from TOML configuration.
- `engines.py:create_engine` supports entry-point groups `epub_tts_narrator.tts` and legacy `audiobook_renderer.tts`.
- The default engine is `drama3_mlx`; unknown or ambiguous engines raise, with no silent fallback.
- Engine plugins provide typed `identity` and `capabilities` dictionaries.
- Required identity fields: `engine`, `model`.
- Capability fields: `voice_design`, `reference_voice`, `preset_voice`, `delivery_control`.

## ANTI-PATTERNS

- Do not add renderer dependencies to the analyzer environment or import heavy packages at module load.
- Do not use the Runpod SDK; `runpod.py` calls the Runpod REST API through `urllib`.
- Do not enable HTTP redirects in `chapter_client.py`; configure a direct URL.
- Do not accept remote artifacts before size and SHA-256 verification, then atomic rename.
- Do not transfer reference WAVs unless they are mono, 16-bit audio.
- Do not trust compressed voice transfers until their decoded PCM hashes match.
- Do not run `audition.py` regenerate-all casually: it replaces voices in the shared series library and does not ask for confirmation.
- Do not replace immutable audition history or voice assets in place.

## NOTES

- `common.py` has only 51 lines but 33 internal consumers; treat changes as high blast radius.
- `workflow.py` is 1,826 lines and the largest maintenance risk; keep changes local.
- `workflow.render()` restores completed chapters when the context hash matches.
- It writes `status.json` and `plan.json` under `work_dir`, with directory locks from `run_locks.acquire_directory_lock`.
- See `ENGINE_NOTES.md` for engine-specific details.
- See root `RENDERER.md` for renderer configuration and operation.
- See root `REMOTE_RENDERING.md` for distributed rendering contracts.
- `reference_cache.py` owns bounded validated WAV reuse for `serve`; `fish_http.py` negotiates it via successful response headers, never for pinned native/SGLang protocols.
- Local `workflow.render()` overlaps one chapter assembly with the next chapter's synthesis; drain assembly before releasing locks. Chunk/recovery algorithms remain shared with chapter jobs.
- `diagnostics.py` owns fatal frame formatting and payload-free SDK/HTTP error details; managed Runpod failures skip optional stop delays.
- `speech_resolver.py` reconsiders newly valid, previously blocked rewrites through fresh sound reviews with an origin-bound `reconsidered` audit. `recurring_resolution.py` resumes that candidate without new failed takes or pending-history growth; consonant-fragment shortening is rewrite-only, never direct audio acceptance.
- `speech_recovery.py` tries em-dash boundaries only when sentence/newline splitting has no valid plan, preserving existing plan identities. `workflow.verified_take` routes planner stop to executable untried splitting only with enough shared starts for all parts; reports distinguish planner stop from runtime split, and failed-join recovery keeps splitting disabled.
- All speech joins embed child acceptance evidence independently of plain-delivery eligibility. `recovery_provenance.py` reconstructs older missing evidence read-only from hash-bound child records and audio for `SpeechCache.identity`; normal generation-identity validation still proves voice, words, and delivery.
- `quality.word_spacing_equivalence` proves whole-transcript canonical written equality before fuzzy checks: ignore case, arbitrary word boundaries, and apostrophes after lexical/number normalization. No minimum piece length or maximum join size; preserve numeric syntax, signs, currency content, character order, and acoustic gates. `speech_coverage` uses the same comparator, not a separate boundary whitelist.
