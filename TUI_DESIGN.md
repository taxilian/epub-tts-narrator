# Terminal UI design (draft for approval)

Status: approved 2026-11-12 with these decisions:

- **Library:** Textual, installed by the normal setup (`renderer.requirements.txt`; the analyzer installs `.[tui]`) while the core package keeps `dependencies = []`. Without it (or with `--plain`, or off a terminal) every command keeps its line output unchanged.
- **Order:** casting TUI first, then the render shell, then the other commands. Build shared components first and compose screens from them.
- **On failure:** the screen stays open on the errors view until `q`.
- **Reference clips:** a per-voice override (`voice-clips.json`), nearest book first.

Done: per-voice clip counts (`voice_clip_counts.py`, `book_voice_settings.py`); shared components (`tui_support.py`, `tui_jobs.py`, `tui_widgets.py`); casting TUI (`renderer/cast_tui_model.py`, `renderer/cast_tui.py`). Run shell for `render` (single and multi-book), `assemble`, `import-voices`, `create-voices`: `run_events.py`, `run_shell_model.py`, `run_shell.py`, `progress.shell_board`, `interrupts.keyboard_interrupts`; it holds open on failure and resumes with `R` by rerunning the command. Analyzer model commands and bulk casting run in the shell too, with their terminal questions as dialogs. In-place chapter retry (`r`) while the rest keeps running, with the render holding for retries until `F`. Render UI: an at-a-glance Summary (health, books, generating/verifying blocks, accepted rate, rejections, failures, ETA, per-book rows) that drills into chapters, workers and pipeline stages; every stop confirmed (Escape/Enter never stop); failure repairs: retry chapter, listen to a failed passage's takes and accept one (manual selection, optional heard-variant learning) and continue, resume only unfinished books. `pronounce` stays a line tool: its listening loop reads raw keys.

## What exists today (survey)

| Area | Today | Reusable seam |
|------|-------|---------------|
| Render progress | `renderer/progress.py`: `TerminalBoard` (pinned rows + scrolling log, ANSI), `ChapterProgress` (per-chapter blocks, per-chunk phase, worker/slot activity, ETA from character-weighted timings), `sched` footer from `render_schedule_status` | Already structured: `set_row`, `_TaskNotifier.phase/worker_activity`, `report_book_outcome` |
| Logging | Every module takes `notify=print` (a one-argument string callback); ~130 call sites in `workflow.py` alone. Multi-book queue prefixes `[book]` via `PrefixedBoardNotify`; chapter rows prefix `Ch N:` | Callback shape is uniform, so a structured sink can stand in for `print` without touching call sites |
| Render errors | Chapter-scoped `RenderBlocked`/`SpeechVerificationFailed` recorded while other chapters finish; `ChaptersRenderFailed` after the drain; persisted in `render-work/status.json` `failed_chapters` (+ `segment_ids`/`chunk_id`/`request_hash`). Run-fatal: `VerificationUnavailable`, interruption, integrity faults | Failure dicts are already the "saved detail" an errors view needs |
| Recovery | Every command resumes by rerunning (checkpoints, request caches); `verify-attribution`/`refresh-directions` resume by `--resume RUN_ID` | Retry = rerun the unit, never new state |
| Analyzer progress | `progress(message)` strings + `stream_progress.StreamConsole` (live provider-stream lines, stage fraction via `meter`) | `observe(event)` dict events and `set_progress(fraction, stage)` |
| Curses | `review_tui.py`: curses-free `ReviewApp` state machine, `CursesScreen` adapter (`size/write/refresh/read_key`), `_drive` loop; offline tests drive the app with a fake screen | Same split (UI-free model, thin screen) reused with Textual; `review-ui` itself stays curses for now |
| Casting | `renderer/audition.py:run_casting`: numbered line menu over `CastingSession` (`rows`, `regenerate`, `accept`, `update_prompt`, `share_voice`, `ensure_ready_many`, `refresh_prompts`, `recover_attempt`) | Session API covers everything; only the front end changes |
| Interactive prompts inside commands | `speaker_decisions` (own full-screen chooser `speaker_tui.SpeakerScreen` via `tui_support.screen_host`; readline with `--plain`), `verify-attribution` accept prompt, `pronounce` listening loop, `terminal.play_sample`/`wait_key`, casting menu | Must suspend the full-screen UI around them |

## Principles

- **Plain stays byte-identical.** Non-TTY, `TERM=dumb`, or `--plain` take exactly today's code path: no event bus is installed, `notify is print`, `live_board`/`StreamConsole` behave as now. A regression test captures today's output for representative commands and compares it after every phase.
- **No new state.** The UI displays and requests; resumable work, checkpoints and caches stay where they are. Retry never bypasses acceptance policy and never invents a voice or speaker choice.
- **Textual-free models, thin Textual screens** built from the shared components in `tui_widgets.py`. Models are tested offline; screens are tested headlessly with Textual's `run_test()` pilot.
- **Flat modules**, stdlib only in the core. Heavy imports stay lazy.
- **No secrets or book passages in persisted logs.** The in-memory log shows what the terminal shows today; an optional log file is scrubbed with `safe_error_text`/`configured_secrets`.

## Phase 1 — event model (`epub_tts_narrator/run_events.py`)

- `Event(time, kind, severity, scope, text, data)`. `kind`: `log`, `progress`, `unit` (start/phase/done/failed), `worker`, `failure`, `outcome`. `scope`: `book`, `chapter`, `chunk`, `voice`, `worker`, `stage`. `data` holds IDs and numbers only.
- `EventBus`: thread-safe fan-out to subscribers plus a bounded ring buffer (about 20k events).
- Scope comes from a `ContextVar` set at existing boundaries (queue book thread, `process(chapter_index)`, `ChapterProgress.task(index)`, worker assignment, casting voice), so the ~130 `notify(...)` sites need no new arguments. `bus.notify` is a print-compatible callable: a plain string becomes a `log` event with the current scope; severity comes from explicit emit calls at failure sites, falling back to recognized markers (`=== Failed`, `Error:`, `Could not`).
- Structured emitters at the seams that already exist: `ChapterProgress` (block counts, per-task phase, worker activity, ETA), `report_book_outcome`, chapter failure recording in `workflow._render`, queue book start/finish, `render_schedule_status` snapshots (polled read-only), analyzer `progress`/`meter`/`StreamConsole.observe`, casting per-voice results.
- Python-level `sys.stdout`/`sys.stderr` are captured into `log` events while the UI owns the screen. Child processes already capture their output; I'll audit for any that inherit fd 1/2.

## Phase 2 — run shell (`run_shell_model.py`, `run_shell.py`, built from `tui_widgets.py`)

Long-running commands run inside it on a TTY: `analyze` (and book updates, `resolve-speakers`), `verify-attribution`, `refresh-directions`, `render` (single book and multi-book queue, local and remote pool), `assemble`, `import-voices`, `cast --action generate-missing|regenerate-all|prepare-samples`, and the labelling stage of `pronounce`. Quick read-only commands (`validate`, `review`, `list`, `plan`, `doctor`) keep printing, because their output is meant to be piped.

Layout:

```
┌ render  3 books  0:42:10 elapsed  ~1:05:00 left  ✓1 ✗0 ▶2 ┐
│ Book / Chapter        blocks       active                  │
│ ▸ Book A    ch 12/38  41/96 43%   #42 gen w2 0:31 · #43 asr │
│ ▸ Book A    ch 13/38   3/88  3%   #4 waiting for w1         │
│   Book B    ch  1/40  12/70 17%   #13 verify 0:04           │
│ Workers  local-0 gen 0:31 · w1 idle · w2 gen 0:12 (remote)  │
│ sched gen 2q/3a asr 0q/1a logical 6/16 backlog 2 pend       │
├ Log [book=Book A   sev≥info]                     follow ●  ┤
│ 12:01:13 Book A   ch12 #40 accepted (attempt 2)            │
│ 12:01:15 Book B   ch1  Processing chapter 1/40 …           │
├ Errors (1) ───────────────────────────────────────────────┤
│ ✗ Book A   ch 9  SpeechVerificationFailed  [r]etry  [Enter] │
└ Tab view · f filter · / search · e errors · r retry · q quit┘
```

- Views: Status, Log, Errors (list → detail with the saved failure record, chunk/segment IDs, failed-take directory), Workers.
- Filters: book, chapter, worker, voice, severity, and free-text `/`; they apply to Log and Errors and dim unrelated status rows.
- Threading: Textual must own the main thread (its driver installs signal handlers), so the command runs on a worker thread. The shell's interrupt key (and Ctrl-C) requests a stop through the command's existing stop checks (`stop_if_requested` in render) and, where a command has none, delivers `KeyboardInterrupt` into the command thread (`PyThreadState_SetAsyncExc`) so the existing interruption handling, checkpoints and resume messages run unchanged. Blocking calls finish first, as with Ctrl-C today. Line prompts from the command thread use `tui_widgets.PromptBridge`, and stray `print` output is captured into the log (`CapturePrintMixin`), as the casting screen already does.
- Interactive prompts: `run_shell.interactive()` suspends the screen (`endwin`, UI stops reading stdin), lets the line prompt run, then resumes. It wraps `terminal.play_sample`/`wait_key`, `speaker_decisions`, the `verify-attribution` accept prompt, and the `pronounce` listening loop.
- On finish: success exits the screen and prints the plain summary lines today's command prints, so scrollback and tmux keep a record. On failure the screen stays open on the Errors view until `q` (see question 2). Exit codes are unchanged.
- Flags: `--plain` on both CLIs; plain is automatic when stdout or stdin is not a TTY.

## Phase 3 — retry and resume semantics

| Command | Unit | In-place retry (rest keeps running) | After a run-fatal error |
|---------|------|-------------------------------------|-------------------------|
| `render` (one book) | chapter | `r` on a failed chapter resubmits `process(index)` with a fresh retry budget on a retry lane that shares the run's scheduler and coordinator pool. Verified clips are reused, so only failed chunks regenerate. When the main drain finishes with failures and the UI is attached, the run holds ("2 failed · r retry · q finish") before raising `ChaptersRenderFailed`. Plain mode never holds. | `R` resumes: rerun the same argv in-process (checkpoints make this cheap). Applies to `VerificationUnavailable`, provider/setup faults and lock conflicts. |
| `render A B C` (queue) | chapter, book | Chapter retry as above, inside its book. A failed book (preparation or fatal) can be re-enqueued at its original priority while other books continue. | Same `R`. |
| `assemble` | book | none (single unit) | `R` reruns. |
| `analyze` / book update / `resolve-speakers` | stage | Provider retries are already automatic; the shell shows them live (`StreamConsole.observe` `retry` phase). | `R` reruns: request cache and `.analysis-pending/` checkpoints resume from the last stage. |
| `verify-attribution`, `refresh-directions` | run | none | `R` reruns with `--resume RUN_ID` of the failed run, which the shell captures from the run's event. |
| `cast` bulk actions | voice | `r` on a failed voice calls `session.regenerate`/`ensure_ready` for that voice while the others continue. | `R` reruns. |
| `import-voices` | voice | `r` re-imports one failed voice. | `R` reruns. |

Retry never touches lock or lease artifacts, never skips verification, and stays chapter- or voice-scoped. Interruption (Ctrl-C) behaves exactly as today.

## Phase 4 — casting TUI (`renderer/cast_tui_model.py`, `renderer/cast_tui.py`) — done first

`cast` (interactive) opens it on a TTY; `--plain` keeps today's numbered menu.

```
┌ Cast — Book A    18/24 ready  filter: status=missing ──────────────────────┐
│ ● Ava                ready    gemini  3 takes │ Ava                          │
│ ○ Mentor             missing                  │ Status  ready · local voice  │
│ ◐ Scribe             generating… 0:22         │ Description  (e) …           │
│ ↳ Ava (older)        derived from Ava         │ Script  (s) from ch 3 …      │
│ ⇢ Guard 2            shares Guard 1           │ Speech traits (t) …          │
│ ⊙ Narrator           inherited (Book 1)       │ Shares  (d) — · used by: —   │
│ ✗ Hound              FAILED: verification …   │ Reference clips (c) 3        │
│                                               │ Takes   ▸ #3 accepted ▶      │
│                                               │         ▸ #2          ▶ [a]  │
└ ↑↓ select · Enter play · g/G/E regenerate local/Gemini/ElevenLabs · a accept · m mark A/B · / filter · q ┘
```

- Statuses from `CastingSession.rows()`: missing (no candidates), pending (draft or candidates but none selected), ready (accepted), inherited, derived (`derived_from`, with clone-stale flag), shared (alias), failed (`regeneration.state == failed`, with its error), generating (a background job is running).
- Filter by text and status; sort by status or name.
- Play: the accepted take, or any candidate (`P` stops; reuses `AudioPlayer` and the normalized playback copies).
- Compare: mark two candidates (or the accepted take and one candidate) and alternate them with `Tab`; `a` accepts the highlighted one through `session.accept`.
- Generate/regenerate: `g`/`G`/`E` per voice; `M` generates every missing voice. Jobs run on a background worker, so the list stays live and shows progress and failures; several jobs queue up (Gemini runs in parallel up to `gemini_concurrency`). Regenerating a shared or inherited voice, or all voices, asks for confirmation first: today `regenerate-all` does not ask.
- Edit (a multi-line editor dialog, saved through `session.update_prompt`): description/prompt (clone direction for derived voices), audition script, speech traits (`voice-traits.json`; empty clears, `-` restores the inherited value). Validation errors from the session appear inline.
- Aliases: `d` share another voice (`share_voice`), `D` stop sharing (`edit_book_aliases(removals=…)`).
- Refresh: `R` refreshes this voice's description and script with the analysis model (`refresh_prompts(voice_ids=[…])`) after confirmation.
- Recover: open a saved attempt file (`recover_attempt`).
- Reference clip count: see question 3.
- Manual choices stay the user's: the TUI acts only on keypresses and never accepts a take or creates a performer on its own.

## Commit plan

1. Done: per-voice reference clip counts.
2. Done: shared Textual components and the casting TUI.
3. Done: event model and the render run shell with status, workers, log, errors, filters, hold-on-failure and resume. Plain output stays on the untouched code path; existing live-board, progress, queue and CLI tests guard it.
4. Done: in-place chapter retry on a retry lane while other chapters keep running (render and the multi-book queue), with a hold for retries before the run reports its failures.
5. Done: analyzer commands (`analyze`, book updates, `resolve-speakers`, `review-attribution`, `refresh-voice-profiles`), `verify-attribution`/`refresh-directions` (resume with `--resume RUN_ID`) and cast bulk actions in the shell. `pronounce` stays a line tool.

Documentation (`RENDERER.md`, `ANALYSIS.md`, `README.md` and the `AGENTS.md` files) is updated with each phase.
