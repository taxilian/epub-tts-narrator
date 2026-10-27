"""Interactive terminal casting and local playback, with no UI dependencies."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unicodedata

from .workflow import VerificationUnavailable
from .model_setup import ModelSetup, ModelSetupError


def display(value):
    """Keep names and model metadata from becoming terminal escape sequences."""
    return "".join(c if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) < 160)
                   else " " for c in str(value))


class AudioPlayer:
    def __init__(self, command, *, normalizer=None):
        self.command = list(command)
        self.normalizer = normalizer
        self.process = None
        self.errors = None

    def prepare(self, path):
        """Prepare a playback copy without changing the saved clone reference."""
        source = Path(path).resolve()
        return self.normalizer.prepare(source) if self.normalizer is not None else source

    def start(self, path):
        self.stop()
        path = self.prepare(path)
        if not path.is_file():
            raise ValueError("The voice sample is missing")
        executable = shutil.which(self.command[0])
        if executable is None:
            raise ValueError("Audio player not found; set casting.player in the TOML configuration")
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                [executable, *self.command[1:], str(path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self.errors,
                shell=False,
            )
        except BaseException:
            self.errors.close()
            self.errors = None
            raise

    def _finish(self, status, *, stopped=False):
        self.errors.seek(0)
        error = self.errors.read(4096).decode("utf-8", errors="replace").strip()
        self.errors.close()
        self.errors = None
        self.process = None
        if status and not stopped:
            raise ValueError(f"Audio playback failed ({status}): {display(error)}")

    def poll(self):
        if self.process is None:
            return False
        status = self.process.poll()
        if status is None:
            return True
        self._finish(status)
        return False

    def wait(self):
        if self.process is not None:
            self._finish(self.process.wait())

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self._finish(self.process.returncode, stopped=True)


def _label(row):
    if row.get("label"):
        return row["label"]
    names = row["characters"]
    return ", ".join(names) if names else ", ".join(row["roles"]) or row["voice_id"]


def _cell_width(text):
    return sum(0 if unicodedata.combining(character) else
               2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
               for character in text)


def _wrap_menu_text(text, width, *, prefix="", indent=""):
    """Wrap complete sanitized text by terminal cells, including long names."""
    lines, line = [], prefix
    for word in display(text).split():
        addition = ("" if line.endswith(" ") else " ") + word
        if line.strip() and _cell_width(line + addition) <= width:
            line += addition
            continue
        if line.strip() and line != prefix:
            lines.append(line.rstrip())
            line = indent
        while _cell_width(line + word) > width:
            available = max(1, width - _cell_width(line))
            end, used = 0, 0
            for character in word:
                cells = _cell_width(character)
                if end and used + cells > available:
                    break
                used += cells
                end += 1
            lines.append(line + word[:end])
            word, line = word[end:], indent
        line += word
    if line.strip():
        lines.append(line.rstrip())
    return lines


def _menu_lines(rows, width):
    """Use available columns without truncating labels or changing numbering."""
    if not rows:
        return []
    width = max(8, width)
    labels = {row["voice_id"]: _label(row) for row in rows}
    entries = []
    for index, row in enumerate(rows, 1):
        status = "" if row["selected"] else " — no sample"
        regeneration = row.get("regeneration")
        if regeneration:
            status = (" — FAILED — no active voice" if regeneration["state"] == "failed"
                      else " — needs regeneration — no active voice")
            if regeneration.get("inherited"):
                status += " — inherited"
        if row["selected"] and row["selected"].get("inherited"):
            status += " — inherited"
        canonical = row.get("canonical_voice_id", row["voice_id"])
        if canonical != row["voice_id"] and canonical in labels:
            status += " — shares " + labels[canonical]
        entries.append((index, _label(row) + status, (regeneration or {}).get("error")))
    number_width = max(3, len(str(len(rows))))
    preferred = max(24, min(36, max(_cell_width(display(text)) + number_width + 2
                                  for _, text, _ in entries)))
    gap = 3
    columns = max(1, min(len(rows), (width + gap) // (preferred + gap)))
    cell_width = (width - gap * (columns - 1)) // columns
    height = (len(rows) + columns - 1) // columns
    cells = []
    for index, text, error in entries:
        prefix = f"{index:{number_width}}. "
        indent = " " * min(len(prefix), cell_width // 2)
        lines = _wrap_menu_text(text, cell_width, prefix=prefix, indent=indent)
        cells.append(lines)
    output = []
    for row_index in range(height):
        row_cells = [cells[row_index + column * height]
                     for column in range(columns) if row_index + column * height < len(cells)]
        for line_index in range(max(map(len, row_cells))):
            parts = [cell[line_index] if line_index < len(cell) else "" for cell in row_cells]
            output.append((" " * gap).join(
                part + " " * max(0, cell_width - _cell_width(part))
                for part in parts).rstrip())
    for index, _, error in entries:
        if error:
            output.extend(_wrap_menu_text(
                f"{index}. {_label(rows[index - 1])}: {error}", width, prefix="  ", indent="  "))
    return output


def _menu_help_lines(width):
    commands = (
        "Number: play", "E [number]: edit", "D [source] [target]: share target's voice",
        "G [number]: regenerate locally", "GG [number]: regenerate with Gemini",
        "G11 [number]: regenerate with ElevenLabs", "Q: quit",
    )
    lines, line = [], ""
    for command in commands:
        combined = line + "   " + command if line else command
        if _cell_width(combined) <= width:
            line = combined
        else:
            if line:
                lines.append(line)
            wrapped = _wrap_menu_text(command, width)
            lines.extend(wrapped[:-1])
            line = wrapped[-1]
    return lines + ([line] if line else [])


def _prepare_playback_samples(player, rows, output, *, include_candidates=False):
    """Warm reusable playback copies, deduplicating aliases and active takes."""
    prepare = getattr(player, "prepare", None)
    if not callable(prepare):
        return 0, 0
    samples, seen = [], set()
    for row in rows:
        voices = ([row["selected"]] if row["selected"] is not None else [])
        if include_candidates:
            voices.extend(row["candidates"])
        for voice in voices:
            source = str(Path(voice["reference_audio"]).resolve())
            if source not in seen:
                seen.add(source)
                samples.append((row, source))
    if samples:
        output(f"Preparing voice playback: {len(samples)} sample(s).")
    failed = 0
    for index, (row, source) in enumerate(samples, 1):
        output(f"[{index}/{len(samples)}] Preparing {_label(row)}…")
        try:
            prepare(source)
        except (ValueError, RuntimeError, OSError) as exc:
            failed += 1
            output(f"Could not prepare playback for {_label(row)}: {exc}")
    return len(samples) - failed, failed


def _warn_regenerate_all(rows, *, book_local=False):
    """Explain the replacement scope before loading any models."""
    groups = {}
    for row in rows:
        canonical = row.get("canonical_voice_id", row["voice_id"])
        groups.setdefault(canonical, []).append(row)
    lines = [
        "WARNING: regenerate-all replaces voices in the shared series voice library.",
        f"This run targets {len(groups)} distinct voices.",
        "Future renders of every book or role sharing these voice IDs will use successful "
        "replacements. Sharing may extend beyond this bundle.",
        "All targeted active voices are cleared before generation. Failed replacements have no active voice.",
        "Previous samples remain as history only; existing audiobook audio is not rewritten.",
    ]
    if book_local:
        lines = [
            "WARNING: regenerate-all creates voice overrides in this book's voices directory.",
            f"This run targets {len(groups)} distinct voices.",
            "Future renders of this book and descendants that inherit these voices will use "
            "successful replacements. Earlier books and sibling branches are unchanged.",
            "All targeted active voices are cleared before generation. Failed replacements have no active voice.",
            "Previous samples remain as history only; existing audiobook audio is not rewritten.",
        ]
    shared = {canonical: members for canonical, members in groups.items() if len(members) > 1}
    if shared:
        lines.append("Linked roles in this book (one replacement for each group):")
        for members in shared.values():
            lines.append("  " + "; ".join(_label(row) for row in members))
    # stderr remains visible when callers redirect progress output. Flush the
    # whole warning before generation can initialize a model or replace a voice.
    print(display("\n".join(lines)), file=sys.stderr, flush=True)


def run_casting(session, player, *, input_fn=None, output=print, playback_fn=None):
    """One menu for playback, editing, sharing, and choosing a voice designer."""
    if input_fn is None:
        from .terminal import enable_line_editing
        enable_line_editing()
        input_fn = input
    if playback_fn is None:
        from .terminal import play_sample
        playback_fn = play_sample

    def say(text=""):
        output(display(text))

    def ask(text):
        return input_fn(display(text)).strip()

    def choose(rows, value):
        if not value.isdecimal() or not 1 <= int(value) <= len(rows):
            raise ValueError("Enter a voice number from the menu")
        return rows[int(value) - 1]

    def play(row, voice):
        say(f"Playing {_label(row)}. Press any key to stop.")
        playback_fn(player, voice["reference_audio"])

    def selected_row(rows, value):
        if value:
            return choose(rows, value)
        if last_voice_id is not None:
            last = next((row for row in rows if row["voice_id"] == last_voice_id), None)
            if last is not None:
                return last
        value = ask("Voice number (empty cancels): ")
        return choose(rows, value) if value else None

    def linked_roles(rows, row):
        canonical = row.get("canonical_voice_id", row["voice_id"])
        return [other for other in rows
                if other["voice_id"] != row["voice_id"]
                and other.get("canonical_voice_id", other["voice_id"]) == canonical]

    def edit(row):
        say("Voice prompt: " + row["prompt"])
        prompt = ask("New prompt (Enter keeps current): ")
        say("Sample script: " + row["reference_text"])
        source = row.get("sample_source", {})
        if source.get("kind") == "book_excerpt":
            say("From: " + str(source.get("chapter_title") or "book text"))
        elif source.get("kind") == "fallback":
            say("Fallback script: no suitable passage long enough for a voice reference.")
        script = ask("New script (Enter keeps current): ")
        say("Vocal traits: " + (row.get("speech_traits") or "(none)"))
        traits = ask("New vocal traits (Enter keeps current; - clears): ")
        edits = {}
        if prompt:
            edits["prompt"] = prompt
        if script:
            edits["reference_text"] = script
        if traits:
            edits["speech_traits"] = "" if traits == "-" else traits
        if edits:
            session.update_prompt(row["voice_id"], **edits)
        return edits

    last_voice_id = None
    try:
        prepare = getattr(session, "prepare_samples", None)
        if callable(prepare):
            prepare()
        rows = session.rows()
        pending = [row for row in rows if row["selected"] is None and not row.get("regeneration")]
        missing = sum(not row["candidates"] for row in pending)
        if pending:
            say(f"Preparing voices: {len(pending) - missing} saved, {missing} to generate.")
        generated = 0
        for row in pending:
            if not row["candidates"]:
                generated += 1
                say(f"[{generated}/{missing}] Generating {_label(row)}…")
            try:
                session.ensure_ready(row["voice_id"])
            except ModelSetupError as exc:
                say(f"Voice preparation paused: {exc}")
                break
            except VerificationUnavailable as exc:
                say(f"Voice preparation paused: {exc}")
                break
            except (ValueError, RuntimeError, OSError) as exc:
                say(f"Could not prepare {_label(row)}: {exc}")
        _prepare_playback_samples(player, session.rows(), say)
        while True:
            rows = session.rows()
            ready = sum(row["selected"] is not None for row in rows)
            say(f"\nVoice casting — {ready}/{len(rows)} ready")
            width = shutil.get_terminal_size(fallback=(80, 24)).columns
            for line in _menu_lines(rows, width):
                say(line)
            for line in _menu_help_lines(max(8, width)):
                say(line)
            if last_voice_id is not None:
                last = next((row for row in rows if row["voice_id"] == last_voice_id), None)
                if last:
                    say(f"Without a number, E, D, G, GG, and G11 use {_label(last)}.")
            action = ask("> ").lower()
            try:
                if action == "q":
                    return
                if action.startswith("d"):
                    numbers = action[1:].split()
                    if len(numbers) > 2:
                        raise ValueError("Use D source-number target-number to share the target's voice")
                    row = selected_row(rows, numbers[0] if numbers else "")
                    if row is None:
                        continue
                    value = numbers[1] if len(numbers) == 2 else ask(
                        f"Use which voice for {_label(row)}? Target number (empty cancels): ")
                    if not value:
                        continue
                    target = choose(rows, value)
                    if row["voice_id"] == target["voice_id"]:
                        raise ValueError("Choose a different target voice")
                    shared = session.share_voice(row["voice_id"], target["voice_id"])
                    last_voice_id = row["voice_id"]
                    say(f"{_label(row)} now shares {_label(target)}'s voice; no new sample was generated.")
                    affected = (shared or {}).get("affected_voice_ids", [])
                    others = [_label(other) for other in rows
                              if other["voice_id"] in affected and other["voice_id"] != row["voice_id"]]
                    if others:
                        say("Also updated linked roles: " + "; ".join(others) + ".")
                elif action.startswith("e"):
                    row = selected_row(rows, action[1:].strip())
                    if row is None:
                        continue
                    say(f"\nEdit {_label(row)}")
                    linked = linked_roles(rows, row)
                    if linked or row.get("canonical_voice_id", row["voice_id"]) != row["voice_id"]:
                        say("Description and script edits affect the shared voice" +
                            (" used by " + "; ".join(_label(other) for other in linked) if linked else "") +
                            ". Vocal traits apply only to this character.")
                    edits = edit(row)
                    last_voice_id = row["voice_id"]
                    if edits:
                        say("Edits saved. The saved sample is unchanged; vocal traits apply to the next render. "
                            "Description and script edits apply when you regenerate.")
                    else:
                        say("No changes saved.")
                elif action.startswith("g"):
                    # G11 is a provider command, not the compact local G 11.
                    if action.startswith("g11"):
                        voice_designer, prefix, location = "elevenlabs", 3, "with ElevenLabs"
                    elif action.startswith("gg"):
                        voice_designer, prefix, location = "gemini", 2, "with Gemini"
                    else:
                        voice_designer, prefix, location = "local", 1, "locally"
                    row = selected_row(rows, action[prefix:].strip())
                    if row is None:
                        continue
                    say(f"\nRegenerate {_label(row)} {location}")
                    if voice_designer == "elevenlabs":
                        say("ElevenLabs receives this voice prompt and sample script; the verified preview is saved locally for cloning.")
                        say("ElevenLabs requires a 20–1000 character voice description and a 100–1000 character sample script.")
                    if (linked_roles(rows, row)
                            or row.get("canonical_voice_id", row["voice_id"]) != row["voice_id"]):
                        say("This voice is shared. Regeneration updates every linked role.")
                    edit(row)
                    say(f"Generating and verifying {_label(row)}…")
                    voice = session.regenerate(row["voice_id"], voice_designer=voice_designer)
                    last_voice_id = row["voice_id"]
                    say("New voice saved and selected.")
                    play(row, voice)
                else:
                    row = choose(rows, action)
                    if row["selected"] is None:
                        raise ValueError("No verified sample is available; use G, GG, or G11 followed by this voice number to retry")
                    last_voice_id = row["voice_id"]
                    play(row, row["selected"])
            except (ValueError, RuntimeError, OSError) as exc:
                say(f"Action stopped: {exc}")
    except EOFError:
        say("\nCasting closed; saved voices and edits are retained.")
    finally:
        player.stop()


def casting_command(args, bundle, config):
    from .casting import CastingSession
    import json

    action = args.action
    attempt_file = getattr(args, "attempt_file", None)
    keep_unselected = getattr(args, "keep_unselected", False)
    include_inherited = getattr(args, "include_inherited", False)
    speech_traits = getattr(args, "speech_traits", None)
    if include_inherited and action != "regenerate-all":
        raise ValueError("--include-inherited is used with --action regenerate-all")
    if action in {"generate", "edit", "play", "accept", "recover"} and not args.voice_id:
        raise ValueError(f"--voice-id is required for --action {action}")
    if args.candidate_id and action not in {"play", "accept"}:
        raise ValueError("--candidate-id is used with --action play or accept")
    if action == "accept" and not args.candidate_id:
        raise ValueError("--candidate-id is required for --action accept")
    if (args.prompt_file or args.text_file) and action not in {"generate", "edit"}:
        raise ValueError("--prompt-file and --text-file are used with --action generate or edit")
    if speech_traits is not None and action not in {"generate", "edit"}:
        raise ValueError("--speech-traits is used with --action generate or edit")
    if action == "edit" and not (args.prompt_file or args.text_file or speech_traits is not None):
        raise ValueError("Provide --prompt-file, --text-file, or --speech-traits to save casting edits")
    if args.voice_id and action in {"interactive", "generate-missing", "regenerate-all", "prepare-samples", "normalize-samples"}:
        raise ValueError("--voice-id is not used with this action")
    if action == "recover" and not attempt_file:
        raise ValueError("--attempt-file is required for --action recover")
    if attempt_file and action != "recover":
        raise ValueError("--attempt-file is used with --action recover")
    if keep_unselected and action != "recover":
        raise ValueError("--keep-unselected is used with --action recover")
    settings = config.get("casting", {})
    enabled = settings.get("normalize_samples", True)
    if action == "normalize-samples" and not enabled:
        raise ValueError("Sample normalization is disabled; set casting.normalize_samples = true")
    if args.voice_id and args.voice_id not in {voice["id"] for voice in bundle["cast"]["voices"]}:
        raise ValueError("Voice ID is not in the bundle cast")
    if action in {"interactive", "generate-missing", "regenerate-all", "prepare-samples", "generate"} and not args.text_file:
        from epub_tts_narrator.audition_scripts import ensure_bundle_scripts
        from .book_context import voice_store
        store = voice_store(bundle, config)
        voice_ids = [args.voice_id] if args.voice_id else None
        if action == "regenerate-all" and not include_inherited:
            # Ignored references do not need new scripts or any analyzer calls.
            voice_ids = [voice["id"] for voice in bundle["cast"]["voices"]
                         if not (store.load(voice["id"]) or store.regeneration_status(voice["id"])
                                 or {}).get("inherited", False)]
        if voice_ids is None or voice_ids:
            bundle = ensure_bundle_scripts(bundle, getattr(args, "config", None), canonical_id=store.canonical_id,
                                           voice_ids=voice_ids)
    normalizer = None
    if enabled and action in {"interactive", "play", "normalize-samples"}:
        from .loudness import SampleNormalizer
        render = config.get("render", {})
        work = render.get("work_dir") or Path(bundle["directory"]) / "render-work"
        normalizer = SampleNormalizer(
            Path(work) / "voice-previews",
            target_lufs=settings.get("sample_loudness_lufs", -20.0),
            true_peak_db=settings.get("sample_true_peak_db", -1.5),
            ffmpeg=render.get("ffmpeg", "ffmpeg"),
        )
    session = CastingSession(bundle, config, model_setup_factory=ModelSetup)
    player = AudioPlayer(settings.get("player", ["/usr/bin/afplay"]), normalizer=normalizer)
    try:
        rows = session.rows()
        row = next((r for r in rows if r["voice_id"] == args.voice_id), None)
        if args.voice_id and row is None:
            raise ValueError("Voice ID is not in the bundle cast")
        if args.prompt_file or args.text_file or speech_traits is not None:
            edits = {}
            if args.prompt_file:
                edits["prompt"] = args.prompt_file.read_text(encoding="utf-8")
            if args.text_file:
                edits["reference_text"] = args.text_file.read_text(encoding="utf-8")
            if speech_traits is not None:
                edits["speech_traits"] = speech_traits
            session.update_prompt(args.voice_id, **edits)
        if action == "interactive":
            run_casting(session, player)
        elif action == "list":
            print(json.dumps([row] if row else rows, ensure_ascii=False, indent=2))
        elif action == "generate":
            candidate = session.regenerate(args.voice_id)
            print(f"Voice saved and selected: {candidate['candidate_id']}\n{candidate['reference_audio']}")
        elif action == "generate-missing":
            print(f"Prepared {len(session.generate_missing())} voices for use.")
        elif action == "prepare-samples":
            print(json.dumps(session.prepare_samples(), ensure_ascii=False, indent=2))
        elif action == "refresh-prompts":
            result = session.refresh_prompts(
                voice_ids=[args.voice_id] if args.voice_id else None,
                config_path=getattr(args, "config", None))
            print(f"Refreshed voice descriptions and sample scripts for {len(result['voices'])} distinct voice(s).")
            print(f"Previous drafts and refresh details: {result['history']}")
            print("Recorded voices are unchanged. Use G/GG in casting or regenerate-all to apply the refreshed drafts.")
        elif action == "normalize-samples":
            prepared, failed = _prepare_playback_samples(
                player, rows, lambda text: print(display(text)), include_candidates=True)
            if failed:
                raise ValueError(f"Could not normalize {failed} sample(s); {prepared} playback copies are ready.")
            print(f"Normalized playback copies are ready for {prepared} sample(s).")
        elif action == "regenerate-all":
            if not include_inherited:
                skipped = {row.get("canonical_voice_id", row["voice_id"]) for row in rows
                           if (row["selected"] or row.get("regeneration") or {}).get("inherited", False)}
                rows = [row for row in rows if row.get("canonical_voice_id", row["voice_id"]) not in skipped]
                if skipped:
                    print(f"Skipped {len(skipped)} inherited voice(s). Use --include-inherited to replace them in this book.")
            if not rows:
                print("No voices to regenerate.")
                return 0
            _warn_regenerate_all(rows, book_local=getattr(session, "config", config).get("_book_local_voices", False))
            session.begin_regeneration([row["voice_id"] for row in rows])
            seen, failed = set(), []
            for row in rows:
                canonical = row.get("canonical_voice_id", row["voice_id"])
                if canonical in seen:
                    continue
                seen.add(canonical)
                print(f"Regenerating {display(_label(row))}…")
                try:
                    session.regenerate(row["voice_id"])
                except VerificationUnavailable:
                    raise
                except (ValueError, RuntimeError, OSError) as exc:
                    failed.append(display(_label(row)))
                    print(f"Could not regenerate {display(_label(row))}: {display(exc)}")
            if failed:
                raise ValueError(
                    f"{len(failed)} voices failed regeneration and have no active voice: {', '.join(failed)}. "
                    "Open casting and use G, GG, or G11 to retry them."
                )
            print(f"Regenerated {len(seen)} shared character voices.")
        elif action == "edit":
            print("Casting edits saved. Vocal traits apply to the next render; "
                  "the voice description and script apply to the next generation.")
        elif action == "accept":
            candidate = session.accept(args.voice_id, args.candidate_id)
            print(f"Accepted {candidate['candidate_id']} for {args.voice_id} in this series.")
        elif action == "recover":
            candidate = session.recover_attempt(args.voice_id, attempt_file, select=not keep_unselected)
            status = "saved without changing the active voice" if keep_unselected else "saved and selected"
            print(f"Recovered voice {status}: {candidate['candidate_id']}\n{candidate['reference_audio']}")
        elif action == "play":
            if row.get("regeneration") and not args.candidate_id:
                raise ValueError("No active voice after bulk regeneration; use G, GG, or G11 in casting to retry")
            candidate = (next((c for c in row["candidates"] if c["candidate_id"] == args.candidate_id), None)
                         if args.candidate_id else row["selected"] or
                         (row["candidates"][-1] if row["candidates"] else None))
            if candidate is None:
                raise ValueError("No matching voice sample; generate one first")
            print(f"Playing {args.voice_id}: {candidate['candidate_id']}")
            player.start(candidate["reference_audio"])
            player.wait()
        return 0
    finally:
        player.stop()
        session.close()
