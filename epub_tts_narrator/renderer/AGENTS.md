# RENDERER PACKAGE

## OVERVIEW

Flat renderer subpackage: 90 Python modules plus `ENGINE_NOTES.md`.
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
3. Chapter/distributed pipeline: `bundle.py`, `chapter_jobs.py`, `chapter_sessions.py`, `chapter_protocol.py`, `chapter_records.py`, `chapter_executor.py`, `chapter_client.py`, servers, cache, storage, verification, pool.
4. Casting/voice design: `casting.py`, `audition.py`, `voices.py`, `voice_design.py`, provider design modules, prompts, bindings, samples, letter names.
5. TTS engines: `engines.py`, `fish_http.py`, `mlx_audio_engine.py`, local runtime, model setup, Drama 3 model/design/sampling modules.
6. Assembly/publication: `assembly.py`, `publication.py`, `cleanup.py`, paragraphs, paragraph recovery, loudness, pacing.
7. Runpod deployment: `runpod.py`, `runpod_api.py`, `runpod_gpu.py`, `runpod_config.py`, `runpod_progress.py`.

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
| Operate Runpod workers | `runpod.py`, `runpod_api.py`, `runpod_config.py`; GPU selection policy in `runpod_gpu.py` |

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
- `workflow.py` is about 2,060 lines and the largest maintenance risk; keep changes local.
- `workflow.render()` restores completed chapters when the context hash matches.
- It writes `status.json` and `plan.json` under `work_dir`, with directory locks from `run_locks.acquire_directory_lock`.
- See `ENGINE_NOTES.md` for engine-specific details.
- See root `RENDERER.md` for renderer configuration and operation.
- See root `REMOTE_RENDERING.md` for distributed rendering contracts.
- Chapter execution requires a connected client. Scheduling and event cursors are current-run only; ACK or disconnect discards run rows after accepted work drains. Persist validated audio/cache manifests, never past-run receipts or scheduling permissions.
- Managed workers remain warm between books and stop after 300 idle seconds. Reuse a matching ready service without model restart; maintenance smoke checks hold a live setup lease, and activation starts consumers without reloading inference.
- `runpod progress` observes without the deployment lock or any allocation/session mutation. `runpod_api.py` reports bounded credential-redacted JSON server diagnoses; allocation requires a standard-volume-capable datacenter as well as GPU stock.
- `runpod_gpu.select_placement` searches US standard-volume locations by GPU class, price, then stock; known-volume regions and the configured location hint only break ties. `runpod.py` checkpoints exact allocation/replacement requests, retains regional volumes without migration or deletion, and forces full bootstrap after an active-storage switch. Resume pending replacements before ordinary provisioning; only confirmed 404 proves a source worker is gone.
- `reference_cache.py` owns bounded validated WAV reuse for `serve`; `fish_http.py` negotiates it via successful response headers, never for pinned native/SGLang protocols.
- Local `workflow.render()` overlaps one chapter assembly with the next chapter's synthesis; drain assembly before releasing locks. Chunk/recovery algorithms remain shared with chapter jobs.
- `diagnostics.py` owns fatal frame formatting and payload-free SDK/HTTP error details; managed Runpod failures skip optional stop delays.
- `speech_resolver.py` reconsiders newly valid, previously blocked rewrites through fresh sound reviews with an origin-bound `reconsidered` audit. `recurring_resolution.py` resumes that candidate without new failed takes or pending-history growth; consonant-fragment shortening preserves the complete ordered consonant-run skeleton and is rewrite-only, never direct audio acceptance.
- `speech_recovery.py` tries em-dash boundaries only when sentence/newline splitting has no valid plan, preserving existing plan identities. `workflow.verified_take` routes planner stop to executable untried splitting only with enough shared starts for all parts; reports distinguish planner stop from runtime split, and failed-join recovery keeps splitting disabled.
- `fragment_recovery.py` gives a failed bound single-speech recovery part one extra fragmentation level for a short final em-dash tail. `fragment_provenance.py` validates exact pieces, the inherited parent plain-delivery audit, and final join identity; a rejected join is not repeated. Depth and shared episode/sibling reserves remain bounded.
- All speech joins embed child acceptance evidence independently of plain-delivery eligibility. `recovery_provenance.py` reconstructs older missing evidence read-only from hash-bound child records and audio for `SpeechCache.identity`; normal generation-identity validation still proves voice, words, and delivery.
- `quality.word_spacing_equivalence` proves whole-transcript canonical written equality before fuzzy checks: ignore case, arbitrary word boundaries, and apostrophes after lexical/number normalization. No minimum piece length or maximum join size; preserve numeric syntax, signs, currency content, character order, and acoustic gates. `speech_coverage` uses the same comparator, not a separate boundary whitelist.
- `speech_coverage` can nominate one-to-one non-numeric protected homophones using matching nonempty full codes, including short codes, for contextual review only. Direct phonetic acceptance is unchanged; numeric protection, initialisms, complete ordered coverage, and review-vote requirements remain intact.
- `letter_names.letter_name_rewrite_equivalent` proves single-letter English name respellings, including repeated-letter name variants, for `speech_resolver` rewrite eligibility only. Keep all sound votes and final verification; do not broaden direct letter matching or authorize changed/missing letters. Existing blocked-proposal reconsideration applies.
- `speech_resolver._prosody_span` permits bounded punctuation-only synthesis repairs with exact normalized word-list equality and all sound reviews. `verification_prompts._PRE_PROSODY_PROMPT` freezes the predecessor bytes; `RecurringResolution` retains its reviewed spelling/delivery history across the upgrade. Historical policies cannot authorize the new repair.
- `retained_candidates.py` nominates complete older candidates only after a saved stop and an ineligible latest transcript. Require fresh ASR, unchanged request/WAV hashes, and normal sound votes; bound scans/reviews and retain negative audits. Do not preempt an untried approved spelling or delivery repair.
- `speech_traits.py` binds current cast speech traits to performance plans and composes them before passage directions. Keep traits in cache/checkpoint identities and recovery, including plain delivery; never mutate voice-reference assets or add traits to voice-creation prompts. Profiles omitting the field keep legacy behavior.
- `clip_repair_*` permit broad clip-local synthesis text, delivery, trait, and split plans after two parallel blind approvals. `clip_recovery.py` routes repeated/terminal failures without changing canonical source or voice identity; existing budgets still apply.
- `exhaustion_candidates.py` nominates physically usable, bound full-span takes; lexical/syllable differences are chooser evidence, not vetoes. `exhaustion_selection.py` honors the decisive LLM choice with `llm_exhaustion_selection`, retaining original failed reports and hash-bound audits through cache and assembly. Never re-veto the choice with ordinary lexical guards.
- `gemini_voice_design.py` treats an HTTP-answered creation failure as knowable: reconcile by exact-name listing, adopt a voice created despite the error, or confirm nothing was created and allow retry. Only transport failures leave a pending state; the first empty listing keeps it repairable, and a repeated empty listing after a recorded failed cleanup becomes a `confirmed_absent` record instead of blocking forever.
