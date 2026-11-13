"""Full-screen speaker chooser for passages automatic assignment could not settle.

Opened by ``speaker_decisions.prompt_speaker_decisions`` inside the run shell
(``tui_support.active_screen_host``). It shows a ``passage_view``: the story
around the passage scrolls on the left, every voice in its own colour (the
colours ``review-ui`` uses), and the passage itself highlighted in the middle;
the menu on the right lists the likely speakers with their reasons, then
every voice, then the other choices. It dismisses with the answer
``prompt_speaker_decisions`` expects; nothing here changes the book.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, SelectionList, Static
from textual.widgets.option_list import Option

from .tui_widgets import BASE_CSS


NARRATION_STYLE = "italic"
UNASSIGNED_STYLE = "bold red"
TARGET_STYLE = "bold black on yellow"


def speaker_styles(view):
    """Rich styles by role ID: cast order through ``review_tui_model.palette``; narrators read plainly."""
    from .review_tui_model import palette
    roles = view["roles"]
    colors = palette(len(roles))
    return {role["role_id"]: (NARRATION_STYLE if role["kind"] == "narration" else f"color({colors[index]})")
            for index, role in enumerate(roles)}


def paragraph_text(paragraph, view, styles):
    """One paragraph as styled text: a character's short name where their speech starts, the passage highlighted.

    Narration is italic and unlabelled; unassigned text is marked as such.
    """
    names, kinds = view.get("names") or view["labels"], {role["role_id"]: role["kind"] for role in view["roles"]}
    text, previous = Text(), None
    for run in paragraph["runs"]:
        speakers = tuple(run["speakers"])
        body = run["text"]
        lead = body[:len(body) - len(body.lstrip())]
        if run["target"]:
            text.append(lead)
            text.append(" ▶ ", style="reverse bold yellow")
            text.append(body.strip(), style=TARGET_STYLE)
            text.append(body[len(body.rstrip()):])
            previous = None
            continue
        style = (styles.get(speakers[0], "") if len(speakers) == 1 else
                 "bold " + styles.get(speakers[0], "") if speakers else UNASSIGNED_STYLE)
        text.append(lead)
        narration = len(speakers) == 1 and kinds.get(speakers[0]) == "narration"
        if speakers != previous and not narration:
            name = " & ".join(names.get(role_id, role_id) for role_id in speakers) or "Unassigned"
            text.append(f"{name} ", style=f"{style} reverse" if style else "reverse")
            text.append(" ")
        previous = speakers
        text.append(body.lstrip(), style=style)
    return text


class NewVoiceScreen(ModalScreen):
    """Name and voice description for a character the cast lacks; Escape goes back."""

    DEFAULT_CSS = BASE_CSS + """
    NewVoiceScreen .dialog { width: 80%; height: auto; }
    NewVoiceScreen Input { margin-bottom: 1; }
    """
    BINDINGS = [Binding("escape", "cancel", "Back", priority=True)]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("New character", classes="dialog-title")
            yield Static("It joins this book's cast and carries into later books in the series.")
            yield Input(placeholder="Name", id="new-name")
            yield Input(placeholder="How the voice should sound", id="new-description")
            yield Static(Text("Enter moves on, then saves · Escape goes back"), classes="dialog-help")

    def on_mount(self):
        self.query_one("#new-name", Input).focus()

    def on_input_submitted(self, event):
        name = self.query_one("#new-name", Input).value.strip()
        description = self.query_one("#new-description", Input).value.strip()
        if event.input.id == "new-name" or not description:
            self.query_one("#new-description" if name else "#new-name", Input).focus()
            return
        if name:
            self.dismiss({"action": "new", "name": name, "description": description})

    def action_cancel(self):
        self.dismiss(None)


class TogetherScreen(ModalScreen):
    """Several characters saying the same words at once; Space marks, Enter confirms."""

    DEFAULT_CSS = BASE_CSS + """
    TogetherScreen .dialog { width: 70%; height: 80%; }
    TogetherScreen SelectionList { height: 1fr; }
    """
    BINDINGS = [Binding("escape", "cancel", "Back", priority=True), Binding("enter", "confirm", "Confirm", priority=True)]

    def __init__(self, view):
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Who says these words together?", classes="dialog-title")
            yield SelectionList(*[(role["label"], role["role_id"]) for role in self.view["roles"]
                                  if role["kind"] == "dialogue"], id="together")
            yield Static(Text("Space marks a speaker · Enter confirms two or more · Escape goes back"),
                         classes="dialog-help")

    def on_mount(self):
        self.query_one("#together", SelectionList).focus()

    def action_confirm(self):
        chosen = list(self.query_one("#together", SelectionList).selected)
        if len(chosen) >= 2:
            self.dismiss({"action": "assign", "roles": chosen})
        else:
            self.notify("Mark at least two speakers, or go back and choose one.", severity="warning")

    def action_cancel(self):
        self.dismiss(None)


ACTIONS = {"together": "Several speak these words together…", "new": "New character…",
           "unresolved": "Leave the speaker unresolved", "skip": "Skip this passage for now",
           "stop": "Save and stop"}


class SpeakerScreen(ModalScreen):
    """Choose who speaks one passage.

    ``actions`` lists the other choices offered (default: together, new,
    skip, stop). With ``back=True`` (choosing a speaker inside another
    review) Escape returns ``None`` instead of stopping, and ``unresolved``
    dismisses with ``{"action": "unresolved"}``.
    """

    DEFAULT_CSS = BASE_CSS + """
    SpeakerScreen { align: center middle; }
    SpeakerScreen #speaker-frame { width: 100%; height: 100%; background: $surface; padding: 0 1; }
    SpeakerScreen #speaker-heading { text-style: bold; height: auto; }
    SpeakerScreen #speaker-body { height: 1fr; }
    SpeakerScreen #speaker-story { width: 3fr; border: round $primary-background; padding: 0 1; }
    SpeakerScreen #speaker-story Static { margin-bottom: 1; }
    SpeakerScreen #speaker-side { width: 2fr; min-width: 36; padding-left: 1; }
    SpeakerScreen #speaker-filter { margin-bottom: 1; }
    SpeakerScreen #speaker-menu { height: 1fr; }
    SpeakerScreen #speaker-error { color: $error; height: auto; }
    SpeakerScreen #speaker-help { color: $text-muted; height: auto; }
    """
    BINDINGS = [
        Binding("escape", "leave", "Save and stop", priority=True),
        Binding("slash", "filter", "Filter voices"),
        Binding("pageup", "scroll_story(-1)", "Scroll story up", show=False),
        Binding("pagedown", "scroll_story(1)", "Scroll story down", show=False),
        Binding("c", "center", "Back to the passage", show=False),
        Binding("t", "together", "Together"), Binding("n", "new", "New character"),
        Binding("s", "skip", "Skip"), Binding("q", "stop", "Save and stop"), Binding("u", "unresolved", "Unresolved"),
        *[Binding(str(number), f"suggestion({number})", show=False) for number in range(1, 10)],
    ]

    def __init__(self, view, *, actions=("together", "new", "skip", "stop"), back=False, title=None):
        super().__init__()
        self.view = view
        self.voice_styles = speaker_styles(view)
        self.actions, self.back, self.title_text = tuple(actions), back, title

    def compose(self) -> ComposeResult:
        view = self.view
        with Vertical(id="speaker-frame"):
            yield Static(Text(self.title_text or f"Passage {view['number']} of {view['total']} · {view['book']} · "
                              f"{view['chapter']} · {view['characters']} characters"), id="speaker-heading")
            with Horizontal(id="speaker-body"):
                with VerticalScroll(id="speaker-story"):
                    for index, paragraph in enumerate(view["paragraphs"]):
                        yield Static(paragraph_text(paragraph, view, self.voice_styles),
                                     id="speaker-target" if index == view["target"] else None)
                with Vertical(id="speaker-side"):
                    yield Input(placeholder="/ to filter voices", id="speaker-filter")
                    yield OptionList(*self._options(""), id="speaker-menu")
                    yield Static(Text(view.get("error") or ""), id="speaker-error")
            keys = {"together": "T together", "new": "N new character", "skip": "S skip",
                    "stop": "Q or Escape save and stop", "unresolved": "U unresolved"}
            yield Static(Text(" · ".join(["↑↓ choose", "Enter assigns", "1–9 a likely speaker", "/ filter",
                                          *(keys[action] for action in self.actions),
                                          *(["Escape goes back"] if self.back else []),
                                          "PgUp/PgDn scroll", "C back to the passage"])), id="speaker-help")

    def _options(self, text):
        """Menu entries: likely speakers (numbered), every voice, then the other choices."""
        words = text.casefold().split()

        def matches(label):
            return all(word in label.casefold() for word in words)

        options, suggested = [], set()
        likely = [row for row in self.view["suggestions"] if matches(row["label"])]
        if likely and not words:
            options.append(Option(Text("Likely speakers", style="bold"), disabled=True))
        for number, row in enumerate(self.view["suggestions"], 1):
            suggested.add(row["role_id"])
            if not matches(row["label"]):
                continue
            label = Text(f"{number}. ")
            label.append_text(self._voice_label(row["role_id"], row["label"]))
            if number == 1:
                label.append("  best guess", style="dim")
            if row["reason"]:
                label.append("\n   " + row["reason"], style="dim")
            options.append(Option(label, id=f"role:{row['role_id']}"))
        others = [role for role in self.view["roles"] if role["role_id"] not in suggested and matches(role["label"])]
        if others and not words:
            options.append(Option(Text("All voices", style="bold"), disabled=True))
        for role in others:
            options.append(Option(self._voice_label(role["role_id"], role["label"]), id=f"role:{role['role_id']}"))
        if not words:
            options.append(Option(Text("Other choices", style="bold"), disabled=True))
            for key in self.actions:
                options.append(Option(Text(ACTIONS[key]), id=f"action:{key}"))
        return options

    def _voice_label(self, role_id, label):
        """The short name in the voice's colour; the rest of a disambiguated label dimmed after it."""
        name = (self.view.get("names") or {}).get(role_id, label)
        text = Text()
        text.append(name, style="bold " + self.voice_styles.get(role_id, ""))  # A span: the list's own style keeps it.
        rest = label[len(name):].lstrip(" —") if label.startswith(name) else ""
        if rest:
            text.append(" — " + rest, style="dim")
        return text

    def on_mount(self):
        self.call_after_refresh(self.action_center)
        menu = self.query_one("#speaker-menu", OptionList)
        menu.focus()
        first = next((index for index in range(menu.option_count)
                      if not menu.get_option_at_index(index).disabled), None)
        if first is not None:
            menu.highlighted = first

    def action_center(self):
        story = self.query_one("#speaker-story", VerticalScroll)
        target = self.query("#speaker-target")
        if target:
            story.scroll_to_center(target.first(), animate=False)

    def action_scroll_story(self, direction):
        story = self.query_one("#speaker-story", VerticalScroll)
        story.scroll_page_down(animate=False) if direction > 0 else story.scroll_page_up(animate=False)

    def action_filter(self):
        self.query_one("#speaker-filter", Input).focus()

    def on_input_changed(self, event):
        menu = self.query_one("#speaker-menu", OptionList)
        menu.clear_options()
        menu.add_options(self._options(event.value))
        first = next((index for index in range(menu.option_count)
                      if not menu.get_option_at_index(index).disabled), None)
        if first is not None:
            menu.highlighted = first

    def on_input_submitted(self, event):
        menu = self.query_one("#speaker-menu", OptionList)
        if menu.highlighted is not None:
            self._choose(menu.get_option_at_index(menu.highlighted).id)
        else:
            menu.focus()

    def on_key(self, event):
        # Up and Down move through the menu while typing in the filter box.
        if isinstance(self.focused, Input) and event.key in ("up", "down"):
            event.stop()
            menu = self.query_one("#speaker-menu", OptionList)
            menu.action_cursor_down() if event.key == "down" else menu.action_cursor_up()

    def check_action(self, action, parameters):
        # Letters and digits are shortcuts only while the menu, not the filter box, has focus.
        if isinstance(self.focused, Input) and action in {"together", "new", "skip", "stop", "suggestion", "filter",
                                                           "center", "unresolved"}:
            return False
        if action in ACTIONS and action not in self.actions:
            return False
        return True

    def on_option_list_option_selected(self, event):
        self._choose(event.option.id)

    def _choose(self, identifier):
        if not identifier:
            return
        kind, _, value = identifier.partition(":")
        if kind == "role":
            self.dismiss({"action": "assign", "roles": [value]})
        else:
            getattr(self, f"action_{value}")()

    def action_suggestion(self, number):
        suggestions = self.view["suggestions"]
        if 1 <= number <= len(suggestions):
            self.dismiss({"action": "assign", "roles": [suggestions[number - 1]["role_id"]]})

    def action_together(self):
        self.app.push_screen(TogetherScreen(self.view), self._answered)

    def action_new(self):
        self.app.push_screen(NewVoiceScreen(), self._answered)

    def _answered(self, answer):
        if answer is not None:
            self.dismiss(answer)

    def action_skip(self):
        self.dismiss({"action": "skip"})

    def action_stop(self):
        self.dismiss({"action": "stop"})

    def action_unresolved(self):
        self.dismiss({"action": "unresolved"})

    def action_leave(self):
        """Escape: out of the filter box first, else back (inside a review) or save and stop."""
        if isinstance(self.focused, Input):
            self.query_one("#speaker-menu", OptionList).focus()
        elif self.back:
            self.dismiss(None)
        else:
            self.action_stop()
