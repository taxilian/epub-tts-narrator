"""Full-screen questions the ``analyze BOOK`` rerun reviews ask inside the run shell.

Each screen draws a ``rerun_questions`` view and dismisses with the letter
the line prompt takes, so the answer goes through the review's own code:

* ``MergeScreen`` — the cast review's "are these the same character?":
  both entries side by side, what merging does; [M] merge / [K] keep apart /
  [L] decide later, ``d`` shows the full plan with its IDs.
* ``GenderScreen`` — the gender check's "fix the description?" for a
  character whose voice exists: [F] fix / [L] leave.
* ``NarratorScreen`` — the narrator check's sections and point-of-view
  readers: [A] apply / [L] leave.
* ``SplitScreen`` — the character split's "are some of these lines someone
  else's?": the lines that would move and where: [S] split / [K] keep / [L] later.
* ``JointLineScreen`` — the joint-line check's "do they say this together?"
  for a line a person gave to one speaker: [J] together / [K] keep / [L] later.

Escape answers like the end of input on a terminal: decide later / leave.
Nothing here changes the book.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, Static

from .tui_widgets import BASE_CSS, Dialog


QUESTION_CSS = BASE_CSS + """
QuestionScreen { align: center middle; }
.question { width: 90%; max-width: 150; height: auto; max-height: 95%; }
.question-body { height: auto; max-height: 60vh; }
.question-why { color: $text-muted; margin-bottom: 1; }
.question-summary { text-style: bold; margin-top: 1; }
.question-note { color: $text-muted; }
.question-buttons { height: auto; margin-top: 1; }
.question-buttons Button { margin-right: 2; }
"""


class QuestionScreen(Dialog):
    """A question with lettered answers: each letter (or its button) answers at once, Escape answers ``later``.

    Subclasses set ``CHOICES`` (``[(letter, label, variant)]``), bind each
    letter to ``choose``, and yield their content from ``body``.
    """

    DEFAULT_CSS = QUESTION_CSS
    CHOICES = ()
    LATER = "l"
    HELP = ""
    ESCAPE = "Escape decides later"

    def __init__(self, view):
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog question"):
            yield Label(Text(self.view["title"]), classes="dialog-title")
            with VerticalScroll(classes="question-body"):
                yield from self.body()
            with Horizontal(classes="question-buttons"):
                for letter, label, variant in self.CHOICES:
                    yield Button(Text(f"[{letter.upper()}] {label}"), id=f"choice-{letter}", variant=variant)
            yield Static(Text("Press a letter (or its button) to answer · " + (self.HELP + " · " if self.HELP else "")
                              + self.ESCAPE), classes="dialog-help")

    def body(self):  # pragma: no cover - overridden
        return []

    def on_mount(self):
        self.query_one(f"#choice-{self.CHOICES[0][0]}", Button).focus()

    def action_choose(self, letter):
        self.dismiss(letter)

    def action_later(self):
        self.dismiss(self.LATER)

    def on_button_pressed(self, event):
        if event.button.id and event.button.id.startswith("choice-"):
            self.dismiss(event.button.id.removeprefix("choice-"))


def _card(entry):
    text = Text()
    text.append(f" {entry['heading']} ", style="bold reverse")
    text.append("\n")
    text.append(entry["name"], style="bold")
    if entry["aliases"]:
        text.append("  also " + ", ".join(entry["aliases"]), style="dim")
    text.append("\n")
    if entry["description"]:
        text.append(entry["description"] + "\n")
    text.append("\nFirst line: ", style="dim")
    text.append((entry["first_line"] or "none in this book") + "\n", style="italic")
    count = entry["lines"]
    text.append(f"\n{count} line{'' if count == 1 else 's'} in this book · {entry['voice']}\n")
    text.append(entry["from"], style="dim")
    return text


class MergeScreen(QuestionScreen):
    """Two cast entries the review thinks are one person: merge, keep apart, or decide later."""

    DEFAULT_CSS = QUESTION_CSS + """
    MergeScreen .merge-cards { height: auto; }
    MergeScreen .merge-card { width: 1fr; height: auto; border: round $primary-background; padding: 0 1; }
    MergeScreen #merge-details { display: none; border: round $primary-background; padding: 0 1; margin-top: 1; }
    MergeScreen #merge-details.shown { display: block; }
    """
    CHOICES = (("m", "Merge them", "primary"), ("k", "Keep them apart", "default"), ("l", "Decide later", "default"))
    HELP = "D shows or hides the details"
    BINDINGS = [Binding("m", "choose('m')", "Merge", priority=True),
                Binding("k", "choose('k')", "Keep apart", priority=True),
                Binding("l,escape", "later", "Decide later", priority=True),
                Binding("d", "details", "Details", priority=True)]

    def body(self):
        view = self.view
        yield Static(Text(view["why"]), classes="question-why")
        with Horizontal(classes="merge-cards"):
            for entry in view["entries"]:
                yield Static(_card(entry), classes="merge-card")
        yield Static(Text(view["summary"]), classes="question-summary")
        yield Static(Text(view["afterwards"]), classes="question-note")
        for note in view["notes"]:
            yield Static(Text(note), classes="question-note")
        yield Static(Text(view["details"]), id="merge-details")

    def action_details(self):
        details = self.query_one("#merge-details", Static)
        details.toggle_class("shown")
        if details.has_class("shown"):
            self.call_after_refresh(details.scroll_visible, animate=False)


class GenderScreen(QuestionScreen):
    """A character whose voice description gives the wrong gender: fix the description, or leave it."""

    DEFAULT_CSS = QUESTION_CSS + """
    GenderScreen .gender-description { border: round $primary-background; padding: 0 1; height: auto; }
    """
    CHOICES = (("f", "Fix the description", "primary"), ("l", "Leave it", "default"))
    ESCAPE = "Escape leaves it"
    BINDINGS = [Binding("f", "choose('f')", "Fix", priority=True),
                Binding("l,escape", "later", "Leave it", priority=True)]

    def body(self):
        view = self.view
        heading = Text(view["name"], style="bold")
        if view["aliases"]:
            heading.append("  also " + ", ".join(view["aliases"]), style="dim")
        yield Static(heading)
        gender = Text("The voice description says ")
        gender.append(view["current"], style="bold")
        gender.append("; the book shows ")
        gender.append(view["wanted"], style="bold")
        gender.append(".")
        yield Static(gender)
        yield Static(Text(view["why"]), classes="question-why")
        for title, description in (("Description now", view["description"]),
                                   ("After the fix", view["fixed"])):
            yield Static(Text(title, style="dim"))
            yield Static(Text(description), classes="gender-description")
        yield Static(Text(view["voice"]), classes="question-summary")


class NarratorScreen(QuestionScreen):
    """Whose point of view each section follows: apply these narrators, or leave the narration as it is."""

    DEFAULT_CSS = QUESTION_CSS + """
    NarratorScreen .narrator-sections { border: round $primary-background; padding: 0 1; height: auto; }
    """
    CHOICES = (("a", "Apply these narrators", "primary"), ("l", "Leave the narration as it is", "default"))
    ESCAPE = "Escape leaves the narration as it is"
    BINDINGS = [Binding("a", "choose('a')", "Apply", priority=True),
                Binding("l,escape", "later", "Leave it", priority=True)]

    def body(self):
        view = self.view
        if view["notes"]:
            yield Static(Text("Your notes: " + view["notes"]), classes="question-why")
        yield Static(Text(view["summary"]), classes="question-summary")
        rows = Text()
        width = max((len(row["label"]) for row in view["sections"]), default=0)
        for index, row in enumerate(view["sections"]):
            if index:
                rows.append("\n")
            rows.append(row["label"].ljust(width + 2))
            rows.append(row["reader"], style="bold" if row["pov"] else "dim")
            if row["changed"]:
                rows.append(f"  {row['changed']} passage(s) change", style="yellow")
            if row["quote"]:
                rows.append(f"\n{' ' * (width + 2)}“{row['quote']}”", style="dim italic")
        yield Static(Text("Sections and who reads their narration", style="dim"))
        yield Static(rows, classes="narrator-sections")


class JointLineScreen(QuestionScreen):
    """A line a person gave to one speaker that the book says several say together."""

    DEFAULT_CSS = QUESTION_CSS + """
    JointLineScreen .joint-line { border: round $primary-background; padding: 0 1; height: auto; }
    """
    CHOICES = (("j", "Say it together", "primary"), ("k", "Keep it as it is", "default"),
               ("l", "Decide later", "default"))
    BINDINGS = [Binding("j", "choose('j')", "Together", priority=True),
                Binding("k", "choose('k')", "Keep", priority=True),
                Binding("l,escape", "later", "Decide later", priority=True)]

    def body(self):
        view = self.view
        yield Static(Text(view["line"], style="bold italic"), classes="joint-line")
        yield Static(Text(view["paragraph"]), classes="question-why")
        now = Text("Now: ", style="dim")
        now.append(view["now"])
        yield Static(now)
        together = Text("Together: ", style="dim")
        together.append(view["together"], style="bold")
        yield Static(together)
        yield Static(Text(view["why"]), classes="question-why")
        yield Static(Text(view["afterwards"]), classes="question-note")


class SplitScreen(QuestionScreen):
    """Some of one cast entry's lines belong to another character: split them off, keep them, or decide later."""

    DEFAULT_CSS = QUESTION_CSS + """
    SplitScreen .split-lines { border: round $primary-background; padding: 0 1; height: auto; }
    """
    CHOICES = (("s", "Split them", "primary"), ("k", "Keep them together", "default"),
               ("l", "Decide later", "default"))
    BINDINGS = [Binding("s", "choose('s')", "Split", priority=True),
                Binding("k", "choose('k')", "Keep", priority=True),
                Binding("l,escape", "later", "Decide later", priority=True)]

    def body(self):
        view = self.view
        yield Static(Text(view["summary"], style="bold"))
        yield Static(Text(view["why"]), classes="question-why")
        lines = Text()
        for line in view["lines"]:
            lines.append(line + "\n", style="italic")
        yield Static(lines, classes="split-lines")
        for note in view["notes"]:
            yield Static(Text(note), classes="question-note")
