"""Full-screen ``pronounce``: choose a word's reading by ear, then settle unsure uses.

The listening loop stays in ``name_pronunciation_tool.NameTool``, which runs
on a worker thread exactly as on the line interface; this app answers its
questions. Before each question the tool reports what it is about
(``NameTool.report``): the current reading, its possessive and plural, the
test sentence with every use marked, or one use to tag with its excerpt.
Every choice becomes the complete command the loop already reads
(``e M AY1 L``, ``i more like sal-vuh-TORE``, ``s proper noun, adjective``),
so the tool asks no follow-up questions. Audio plays here; Space stops it.
"""

from __future__ import annotations

import queue
import threading
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, OptionList, SelectionList, Static
from textual.widgets.option_list import Option

from ..tui_widgets import BASE_CSS, Dialog, MessageLog, TextEditScreen


DEFINE_ACTIONS = [
    ("p", "▶  Play the test sentence again"),
    ("b", "▶  Hear the engine's own reading (no tag)"),
    ("a", "✓  Accept this reading"),
    ("i", "✎  Describe how it should sound…"),
    ("n", "↻  Ask for a different guess"),
    ("e", "⌨  Edit the phonemes…"),
    ("r", "Aa Say it as another spelling…"),
    ("s", "☰  Choose the parts of speech…"),
    ("q", "✗  Quit (your draft is kept)"),
]
TAG_ACTIONS = [("y", "✓  Yes, this use gets the reading"), ("n", "✗  No, keep the default reading here"),
               ("", "→  Keep as labelled"), ("q", "■  Stop tagging")]
_EOF = object()


class PartsScreen(Dialog):
    """Which parts of speech get the reading; Space marks, Enter confirms."""

    DEFAULT_CSS = BASE_CSS + "PartsScreen .dialog { width: 60; height: auto; } PartsScreen SelectionList { height: 9; }"
    BINDINGS = [Binding("escape", "cancel", "Back", priority=True),
                Binding("enter", "confirm", "Confirm", priority=True)]

    def __init__(self, chosen):
        super().__init__()
        self.chosen = set(chosen)

    def compose(self) -> ComposeResult:
        from ..name_pronunciations import PART_LABELS, PARTS_OF_SPEECH
        with Vertical(classes="dialog"):
            yield Label("Which uses get this reading?", classes="dialog-title")
            yield SelectionList(*[(PART_LABELS[part], part, part in self.chosen) for part in PARTS_OF_SPEECH],
                                id="parts")
            yield Static(Text("Space marks · Enter confirms · Escape goes back"), classes="dialog-help")

    def on_mount(self):
        self.query_one("#parts", SelectionList).focus()

    def action_confirm(self):
        from ..name_pronunciations import PART_LABELS, PARTS_OF_SPEECH
        selected = set(self.query_one("#parts", SelectionList).selected)
        if not selected:
            self.notify("Mark at least one part of speech.", severity="warning")
            return
        self.dismiss(", ".join(PART_LABELS[part] for part in PARTS_OF_SPEECH if part in selected))

    def action_cancel(self):
        self.dismiss(None)


def sentence_text(sentence, spans):
    """The test sentence with every use that gets the reading highlighted."""
    text, cursor = Text(), 0
    for start, end, _ in sorted(spans):
        text.append(sentence[cursor:start])
        text.append(sentence[start:end], style="bold black on yellow")
        cursor = end
    text.append(sentence[cursor:])
    return text


class PronounceApp(App):
    """Runs ``command(input_fn, output, playback_fn, observer)`` on a worker; its exit code is the result."""

    CSS = BASE_CSS + """
    #pron-heading { text-style: bold; height: auto; padding: 0 1; }
    #pron-body { height: 1fr; }
    #pron-main { width: 3fr; padding: 0 1; }
    #pron-reading, #pron-sentence { border: round $primary-background; padding: 0 1; height: auto; margin-bottom: 1; }
    #pron-status { height: auto; color: $warning; padding: 0 1; }
    #pron-actions { width: 2fr; min-width: 40; }
    #pron-log { height: 9; border: round $primary-background; }
    #pron-help { color: $text-muted; height: auto; padding: 0 1; }
    """
    BINDINGS = [Binding("space", "stop_audio", "Stop audio"), Binding("escape", "quit_word", "Quit"),
                *[Binding(key, f"command('{key}')", show=False) for key in "pbainers"],
                Binding("y", "command('y')", show=False), Binding("q", "quit_word", show=False)]

    def __init__(self, command, *, title="pronounce"):
        super().__init__()
        self.command, self.title_text = command, title
        self.answers = queue.Queue()
        self.stop_playback = threading.Event()
        self.state, self.waiting, self.result, self.raised = None, False, 0, None

    def compose(self) -> ComposeResult:
        yield Static(Text(self.title_text), id="pron-heading")
        with Horizontal(id="pron-body"):
            with Vertical(id="pron-main"):
                yield Static(id="pron-reading")
                yield Static(id="pron-sentence")
                yield Static(id="pron-status")
            yield OptionList(id="pron-actions")
        yield MessageLog(id="pron-log")
        yield Static(id="pron-help")

    def on_mount(self):
        self.set_status("Starting…")
        self.query_one("#pron-actions", OptionList).focus()
        self.run_worker(self._run, thread=True, exclusive=True)

    # Worker side -------------------------------------------------------------
    def _call(self):
        return self.command(self._input, self._output, self._play, self._observe)

    def _run(self):
        try:
            self.result = self._call()
        except BaseException as error:  # Shown after the screen closes, as on the line interface.
            self.raised = error
        self.call_from_thread(self.exit, self.result)

    def _observe(self, state):
        self.call_from_thread(self.show_state, dict(state))

    def _output(self, text):
        self.call_from_thread(self.log_line, str(text))

    def _input(self, prompt=""):
        self.call_from_thread(self.set_waiting, True)
        answer = self.answers.get()
        self.call_from_thread(self.set_waiting, False)
        if answer is _EOF:
            raise EOFError
        return answer

    def _wait(self, seconds):
        """The pause after a play: a key ends it early (returned), else None."""
        if self.stop_playback.wait(seconds):
            self.stop_playback.clear()
            return "key"
        return None

    PLAYING = "▶ Playing… Space stops."

    def _play(self, player, path):
        """Play until done or Space; True only when it played to the end."""
        self.stop_playback.clear()
        self.call_from_thread(self.set_status, self.PLAYING)
        try:
            player.start(path)
            while player.poll():
                if self.stop_playback.is_set():
                    player.stop()
                    return False
                time.sleep(0.05)
            return True
        finally:
            player.stop()
            self.call_from_thread(self.set_status, "")

    # UI side -------------------------------------------------------------------
    def log_line(self, text):
        self.query_one("#pron-log", MessageLog).write(Text(text))
        if not self.waiting and text.strip():
            self.set_status(text if len(text) < 160 else text[:157] + "…")

    def set_status(self, text):
        self.query_one("#pron-status", Static).update(Text(text))

    def set_waiting(self, waiting):
        self.waiting = waiting
        if waiting:
            self.set_status("")
        self.query_one("#pron-actions", OptionList).disabled = not waiting
        if waiting:
            self.query_one("#pron-actions", OptionList).focus()

    def show_state(self, state):
        self.state = state
        actions = self.query_one("#pron-actions", OptionList)
        actions.clear_options()
        if state["kind"] == "define":
            self._show_define(state)
            actions.add_options([Option(Text(label), id=f"cmd:{key}") for key, label in DEFINE_ACTIONS])
            help_text = ("↑↓ choose · Enter does it · P play · B engine's reading · A accept · I describe · N another "
                         "guess · E phonemes · R spelling · S parts of speech · Space stops audio · Q quit")
        else:
            self._show_tag(state)
            actions.add_options([Option(Text(label), id=f"cmd:{key}") for key, label in TAG_ACTIONS])
            help_text = "↑↓ choose · Enter does it · Y yes · N no · Q stop tagging"
        actions.highlighted = 0
        self.query_one("#pron-help", Static).update(Text(help_text))

    def _show_define(self, state):
        forms = ", ".join(f"{count} {form}" for form, count in state["forms"].items())
        self.query_one("#pron-heading", Static).update(Text(
            f"{self.title_text} · {state['name']} · {state['uses']} use(s) in this book ({forms})"))
        reading = Text()
        reading.append("Reading  ", style="dim")
        if state.get("say_as"):
            reading.append("said as ", style="dim")
            reading.append(state["say_as"], style="bold")
        else:
            reading.append(state["phonemes"] or "", style="bold")
        if state.get("respelling"):
            reading.append(f"   say it like {state['respelling']}", style="italic")
        reading.append("\nApplies to  ", style="dim")
        reading.append(state["applies"])
        for label, (word, value) in (("Possessive", state["possessive"]), ("Plural", state["plural"])):
            reading.append(f"\n{label}  ", style="dim")
            reading.append(f"{word} → {value}")
        self.query_one("#pron-reading", Static).update(reading)
        sentence = Text("Test sentence\n", style="dim")
        sentence.append_text(sentence_text(state["sentence"], state["spans"]))
        self.query_one("#pron-sentence", Static).update(sentence)

    def _show_tag(self, state):
        self.query_one("#pron-heading", Static).update(Text(
            f"{self.title_text} · {state['name']} · use {state['index']} of {state['total']} to decide"))
        reading = Text()
        reading.append(f"Should this use of {state['name']} get its reading ({state['parts']})?\n", style="bold")
        reading.append(state["label"], style="dim")
        self.query_one("#pron-reading", Static).update(reading)
        excerpt = Text(state["before"])
        excerpt.append(state["word"], style="bold black on yellow")
        excerpt.append(state["after"])
        self.query_one("#pron-sentence", Static).update(excerpt)

    def answer(self, command):
        if not self.waiting:
            self.set_status("Still working; Space stops audio.")
            return
        self.answers.put(command)

    def on_option_list_option_selected(self, event):
        self.action_command(event.option.id.removeprefix("cmd:"))

    def action_command(self, key):
        if not self.waiting:
            self.set_status("Still working; Space stops audio.")
            return
        state = self.state or {}
        if state.get("kind") == "tag":
            if key in {"y", "n", "", "q"}:
                self.answer(key)
            return
        prompts = {"i": ("How should it sound?", "", "e.g. more like sal-vuh-TORE-ee · Enter asks the model · Escape cancels"),
                   "e": ("Phonemes (CMU Arpabet; | splits groups, e.g. M AY1 L | K IY1)", state.get("phonemes") or "",
                         "Enter saves and plays · Escape cancels"),
                   "r": (f"Say {state.get('name', 'it')} as this spelling", state.get("say_as") or "",
                         "e.g. mile-key · Enter saves and plays · Escape cancels")}
        if key in prompts:
            title, value, help_text = prompts[key]
            self.push_screen(TextEditScreen(title, value, single_line=True, help_text=help_text),
                             lambda text: self.answer(f"{key} {text}") if text else None)
        elif key == "s":
            self.push_screen(PartsScreen(state.get("parts", [])),
                             lambda parts: self.answer(f"s {parts}") if parts else None)
        elif key in {"p", "b", "a", "n", "q"}:
            self.answer(key)

    def check_action(self, action, parameters):
        # While a dialog is open its own keys apply (typing, Escape to cancel).
        if action in {"command", "quit_word", "stop_audio"} and len(self.screen_stack) > 1:
            return False
        return True

    def action_stop_audio(self):
        self.stop_playback.set()

    def action_quit_word(self):
        self.stop_playback.set()
        if self.waiting:
            self.answer("q")

    def on_unmount(self):
        # Closing the screen ends input, which the tool treats as quitting with its draft kept.
        self.stop_playback.set()
        self.answers.put(_EOF)


def run_pronounce_app(command, *, title):
    """Run ``command`` (see ``PronounceApp``) full-screen; print its messages afterwards and return its code."""
    app = PronounceApp(command, title=title)
    result = app.run()
    if app.raised is not None:
        raise app.raised
    return result if isinstance(result, int) else app.result


QUEUE_ACTIONS = [
    ("s", "▶  Start listening here (each word twice, then accepted)"),
    ("play", "▶  Play this word in its book sentence"),
    ("b", "▶  Hear the engine's own reading"),
    ("e", "Aa Say it as a spelling…"),
    ("p", "⌨  Edit the phonemes…"),
    ("c", "✓  Confirm the current choice"),
    ("i", "✗  Keep the engine's default reading"),
    ("r", "↺  Reset to the inherited choice"),
    ("o", "↻  Mark for listening again"),
    ("o all", "↻  Mark every entry for listening again"),
    ("q", "■  Quit (everything is saved)"),
]


class PronunciationQueueApp(PronounceApp):
    """``pronunciations BOOK``: the listening queue, its entries listed, any key interrupting a walk.

    ``command(input_fn, output, playback_fn, observer, wait_fn, redraw)``
    runs ``pronunciation_review_queue.run_pronunciation_queue``; every
    choice is the command its line menu reads (``e 3 mile-key``, ``s 7``).
    """

    CSS = PronounceApp.CSS + """
    #queue-entries { width: 3fr; border: round $primary-background; }
    #queue-side { width: 2fr; min-width: 44; }
    #queue-detail { height: auto; border: round $primary-background; padding: 0 1; margin-bottom: 1; }
    """
    PLAYING = "▶ Listening… any key interrupts and goes back five words."
    BINDINGS = [Binding("escape,q", "command('q')", "Quit"), Binding("s", "command('s')", "Listen"),
                *[Binding(key, f"command('{key}')", show=False) for key in "becirop"],
                Binding("O", "command('o all')", show=False)]

    def compose(self) -> ComposeResult:
        yield Static(Text(self.title_text), id="pron-heading")
        with Horizontal(id="pron-body"):
            yield OptionList(id="queue-entries")
            with Vertical(id="queue-side"):
                yield Static(id="queue-detail")
                yield Static(id="pron-status")
                yield OptionList(*[Option(Text(label), id=f"cmd:{key}") for key, label in QUEUE_ACTIONS],
                                 id="pron-actions")
        yield MessageLog(id="pron-log")
        yield Static(Text("↑↓ entries · Tab to actions · S listen from here · P phonemes · E spelling · B engine's "
                          "reading · C confirm · I keep default · R reset · O listen again (Shift+O all) · Q quit · "
                          "while listening, any key interrupts"), id="pron-help")

    def on_mount(self):
        self.set_status("Starting…")
        self.query_one("#queue-entries", OptionList).focus()
        self.run_worker(self._run, thread=True, exclusive=True)

    def _call(self):
        return self.command(self._input, self._output, self._play, self._observe, self._wait,
                            lambda text: self.call_from_thread(self.set_status, text.strip("\r\n").replace("\x1b[K", "")))

    def set_waiting(self, waiting):
        self.waiting = waiting
        if waiting:
            self.set_status("")
        self.query_one("#pron-actions", OptionList).disabled = not waiting

    def show_state(self, state):
        moved = self.state is None or self.state.get("resume") != state.get("resume")
        self.state = state
        entries = self.query_one("#queue-entries", OptionList)
        current = entries.highlighted
        entries.clear_options()
        for row in state["rows"]:
            label = Text()
            label.append("● " if row["queued"] else "  ", style="yellow")
            label.append(f"{row['number']:>3}. ")
            label.append(row["word"], style="bold")
            label.append(f" — {row['description']}", style="dim")
            if row["error"]:
                label.append("  audio unavailable", style="red")
            if row["conflict"]:
                label.append("  capitalization differs", style="magenta")
            entries.add_option(Option(label, id=str(row["number"])))
        if entries.option_count:
            wanted = next((row["number"] - 1 for row in state["rows"] if row["word"] == (state.get("resume") or state.get("last"))), 0)
            # A walk that stopped moves the selection to where listening resumes.
            entries.highlighted = (wanted if moved or current is None or current >= entries.option_count
                                   else current)
        rows = len(state["rows"])
        self.query_one("#pron-heading", Static).update(Text(
            f"{self.title_text} · {state['queued']} of {rows} entr{'y' if rows == 1 else 'ies'} await listening"))
        self._show_detail()

    def _selected(self):
        entries = self.query_one("#queue-entries", OptionList)
        if not self.state or entries.highlighted is None or not self.state["rows"]:
            return None
        return self.state["rows"][entries.highlighted]

    def _show_detail(self):
        row = self._selected()
        detail = Text()
        if row is None:
            detail.append("No pronunciation entries.", style="dim")
        else:
            detail.append(f"{row['number']}. {row['word']}\n", style="bold")
            detail.append(f"Now: {row['description']}\nStatus: {row['status']}")
            if row["queued"]:
                detail.append("\nAwaits listening", style="yellow")
            if row["proposal"]:
                detail.append("\nA guessed proposal: play it and confirm it yourself", style="dim")
            if row["error"]:
                detail.append(f"\nAudio unavailable: {row['error']}", style="red")
        self.query_one("#queue-detail", Static).update(detail)

    def on_option_list_option_highlighted(self, event):
        if event.option_list.id == "queue-entries":
            self._show_detail()

    def on_option_list_option_selected(self, event):
        if event.option_list.id == "queue-entries":
            self.action_command("play")
        else:
            self.action_command(event.option.id.removeprefix("cmd:"))

    def on_key(self, event):
        # While a word plays or in the pause after it, any key interrupts the walk.
        if not self.waiting and len(self.screen_stack) == 1:
            self.stop_playback.set()
            event.stop()

    COMMANDS = frozenset({"s", "play", "b", "e", "p", "c", "i", "r", "o", "o all", "q"})

    def action_command(self, key):
        if not self.waiting or key not in self.COMMANDS:
            return
        row = self._selected()
        number = str(row["number"]) if row else ""
        if key in {"q", "o all"}:
            return self.answer(key)
        if row is None:
            return
        if key == "play":
            return self.answer(number)
        if key in {"e", "p"}:
            title = ("Say {word} as this spelling" if key == "e" else "Phonemes for {word} (CMU Arpabet)").format(word=row["word"])
            self.push_screen(TextEditScreen(title, "", single_line=True,
                                            help_text="Enter saves and plays it · Escape cancels"),
                             lambda text: self.answer(f"{key} {number} {text}") if text else None)
            return
        self.answer(f"{key} {number}")


def run_queue_app(command, *, title):
    app = PronunciationQueueApp(command, title=title)
    result = app.run()
    if app.raised is not None:
        raise app.raised
    return result if isinstance(result, int) else app.result
