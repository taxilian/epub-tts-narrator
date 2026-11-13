"""Full-screen review of ``verify-attribution`` findings.

Opened by ``attribution_verification_command`` inside the run shell
(``tui_support.active_screen_host``) over an
``attribution_manual_review.FindingReview``. The findings list on the left
shows each finding's decision; the story around the selected one scrolls on
the right in each voice's colour with the passage highlighted, above what
the verifier proposes, its evidence and what will be saved. Accept, leave,
choose another speaker (an existing voice, several together, unresolved or
a new character), change the direction or the review flag, then save.
Dismisses with ``"save"`` or ``"leave"``; nothing is published here.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .speaker_tui import NewVoiceScreen, SpeakerScreen, paragraph_text, speaker_styles
from .tui_widgets import BASE_CSS, ConfirmScreen, TextEditScreen


MARKS = {"undecided": ("·", "dim"), "accept": ("✓", "green"), "leave": ("✗", "red"),
         "edit": ("✎", "yellow"), "new": ("+", "magenta")}
STATUS = {"undecided": "undecided", "accept": "accept the proposal", "leave": "leave as saved",
          "edit": "your edit", "new": "new character"}


def _speaker_names(segment, review):
    if segment.get("simultaneous_speakers"):
        role_ids = [speaker["role_id"] for speaker in segment["simultaneous_speakers"]]
    else:
        role_ids = [segment["role_id"]] if segment.get("role_id") else []
    return role_ids, " & ".join(review.labels[role_id] for role_id in role_ids) or "unresolved"


class FindingsScreen(ModalScreen):
    """Decide each attribution finding, then save or leave."""

    DEFAULT_CSS = BASE_CSS + """
    FindingsScreen #review-frame { width: 100%; height: 100%; background: $surface; padding: 0 1; }
    FindingsScreen #review-heading { text-style: bold; height: auto; }
    FindingsScreen #review-body { height: 1fr; }
    FindingsScreen #review-list { width: 2fr; min-width: 34; }
    FindingsScreen #review-right { width: 3fr; padding-left: 1; }
    FindingsScreen #review-story { height: 1fr; border: round $primary-background; padding: 0 1; }
    FindingsScreen #review-story Static { margin-bottom: 1; }
    FindingsScreen #review-detail { height: auto; max-height: 14; border: round $primary-background; padding: 0 1; }
    FindingsScreen #review-help { color: $text-muted; height: auto; }
    """
    BINDINGS = [
        Binding("a,enter", "accept", "Accept", priority=True), Binding("l", "leave_finding", "Leave"),
        Binding("s", "speaker", "Speaker…"), Binding("n", "new", "New character…"),
        Binding("d", "direction", "Direction…"), Binding("v", "review_flag", "Review flag"),
        Binding("u", "undo", "Undo"), Binding("y", "accept_rest", "Accept the rest"),
        Binding("w", "save", "Save"), Binding("q,escape", "quit", "Quit"),
        Binding("pageup", "scroll_story(-1)", show=False), Binding("pagedown", "scroll_story(1)", show=False),
    ]

    def __init__(self, review, *, book=""):
        super().__init__()
        self.review, self.book = review, book
        self.error = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="review-frame"):
            yield Static(id="review-heading")
            with Horizontal(id="review-body"):
                yield OptionList(*self._options(), id="review-list")
                with Vertical(id="review-right"):
                    yield VerticalScroll(id="review-story")
                    yield Static(id="review-detail")
            yield Static(Text("↑↓ finding · Enter/A accept · L leave as saved · S another speaker · N new character · "
                              "D direction · V review flag · U undo · Y accept the rest · W save · Q quit · "
                              "PgUp/PgDn scroll"), id="review-help")

    def _options(self):
        review, options = self.review, []
        styles = speaker_styles({"roles": [{"role_id": role["id"], "kind": role["kind"]} for role in review.roles]})
        for index, finding in enumerate(review.found):
            mark, style = MARKS[review.status(index)]
            label = Text()
            label.append(f"{mark} ", style=f"bold {style}")
            label.append(f"{index + 1}. ")
            if finding["review_only"]:
                label.append("needs review", style="yellow")
            else:
                segment_id = review.segment_id(index)
                label.append_text(self._voices(review.originals[segment_id], styles))
                label.append(" → ")
                label.append_text(self._voices(review.proposals[segment_id], styles))
            label.append("\n   " + finding["text"][:70], style="dim")
            options.append(Option(label, id=str(index)))
        return options

    @property
    def index(self):
        menu = self.query_one("#review-list", OptionList)
        return menu.highlighted if menu.highlighted is not None else 0

    def on_mount(self):
        menu = self.query_one("#review-list", OptionList)
        menu.focus()
        if menu.option_count:
            menu.highlighted = 0
        self.refresh_finding()

    def on_option_list_option_highlighted(self, event):
        self.error = ""
        self.refresh_finding()

    def view(self, index):
        from .speaker_decisions import passage_view
        review = self.review
        segment = review.originals[review.segment_id(index)]
        proposal = review.proposals[review.segment_id(index)]
        suggestions = []
        for source, reason in ((proposal, "Proposed by verification."), (segment, "As saved.")):
            for role_id in _speaker_names(source, review)[0]:
                if role_id not in {row["role_id"] for row in suggestions}:
                    suggestions.append({"role_id": role_id, "reason": reason})
        return passage_view(review.document, segment, number=index + 1, total=len(review.found),
                            labels=review.labels, suggestions=suggestions, roles=review.roles, paragraphs=6)

    def refresh_finding(self):
        review = self.review
        counts = review.counts()
        self.query_one("#review-heading", Static).update(Text(
            f"Attribution review{' · ' + self.book if self.book else ''} · {len(review.found)} finding(s) · "
            f"{counts['accept']} accepted · {counts['edit'] + counts['new']} edited · {counts['leave']} left · "
            f"{counts['undecided']} undecided"))
        if not review.found:
            return
        index = self.index
        view = self.view(index)
        styles = speaker_styles(view)
        story = self.query_one("#review-story", VerticalScroll)
        story.remove_children()
        widgets = [Static(paragraph_text(paragraph, view, styles), classes="review-target" if number == view["target"]
                          else None) for number, paragraph in enumerate(view["paragraphs"])]
        story.mount_all(widgets)
        target = widgets[view["target"]] if widgets else None
        if target is not None:
            self.call_after_refresh(lambda: story.scroll_to_center(target, animate=False))
        finding = review.found[index]
        saved = review.originals[review.segment_id(index)]
        staged = review.staged(index)
        status = review.status(index)
        detail = Text()
        detail.append(f"Finding {index + 1} of {len(review.found)} · chapter {finding['chapter_id']}\n", style="bold")
        detail.append("Saved: ")
        detail.append_text(self._voices(saved, styles))
        detail.append("   Verification proposes: ")
        detail.append("a review flag" if finding["review_only"] else "", style="yellow")
        if not finding["review_only"]:
            detail.append_text(self._voices(review.proposals[review.segment_id(index)], styles))
        detail.append("\n")
        for item in finding["evidence"]:
            detail.append(f"Evidence: “{item['quote']}”\n", style="dim")
        mark, style = MARKS[status]
        detail.append(f"{mark} Decision: {STATUS[status]}", style=f"bold {style}")
        if status != "undecided":
            detail.append(" — will save ")
            if status == "new":
                decision = review.decisions[review.segment_id(index)]
                detail.append(f"{decision['name']} (new: {decision['description']})", style="bold magenta")
            else:
                detail.append_text(self._voices(staged, styles))
        direction = staged["delivery"]["direction"] or "(none)"
        detail.append(f"\nDirection: {direction}   Review flag: "
                      f"{'yes' if staged['attribution']['review_required'] else 'no'}")
        if self.error:
            detail.append(f"\n{self.error}", style="bold red")
        self.query_one("#review-detail", Static).update(detail)

    def _voices(self, segment, styles):
        role_ids, label = _speaker_names(segment, self.review)
        text = Text()
        if not role_ids:
            text.append("unresolved", style="bold red")
        for number, role_id in enumerate(role_ids):
            if number:
                text.append(" & ")
            text.append(self.review.names[role_id], style="bold " + styles.get(role_id, ""))
        return text

    def _refresh_list(self, advance=False):
        menu = self.query_one("#review-list", OptionList)
        index = self.index
        menu.clear_options()
        menu.add_options(self._options())
        menu.highlighted = min(index + (1 if advance else 0), menu.option_count - 1)
        self.refresh_finding()

    def _attempt(self, change, advance=False):
        try:
            change()
        except ValueError as error:
            self.error = str(error)
            self.refresh_finding()
            return
        self.error = ""
        self._refresh_list(advance)

    def action_accept(self):
        self._attempt(lambda: self.review.accept(self.index), advance=True)

    def action_leave_finding(self):
        self._attempt(lambda: self.review.leave(self.index), advance=True)

    def action_undo(self):
        self._attempt(lambda: self.review.undo(self.index))

    def action_accept_rest(self):
        self._attempt(self.review.accept_undecided)

    def action_review_flag(self):
        staged = self.review.staged(self.index)
        self._attempt(lambda: self.review.set_review(self.index, not staged["attribution"]["review_required"]))

    def action_speaker(self):
        index = self.index
        view = self.view(index)
        for name in self.review.new_names():
            # Reusing a character made earlier in this review: offered like any voice.
            view["roles"].append({"role_id": f"new:{name}", "label": f"{name} (new)", "kind": "dialogue",
                                  "voice_id": f"new:{name}"})
            view["labels"][f"new:{name}"] = view["names"][f"new:{name}"] = f"{name} (new)"

        def chosen(answer):
            if not answer:
                return
            if answer["action"] == "new":
                self._attempt(lambda: self.review.new_character(index, answer["name"], answer["description"]))
            elif answer["action"] == "unresolved":
                self._attempt(lambda: self.review.set_speaker(index, []))
            elif answer["action"] == "assign":
                roles = answer["roles"]
                fresh = [role_id for role_id in roles if role_id.startswith("new:")]
                if fresh and len(roles) == 1:
                    name = fresh[0].removeprefix("new:")
                    self._attempt(lambda: self.review.new_character(index, name, "-"))
                elif fresh:
                    self.error = "A new character speaks alone here; choose them without others."
                    self.refresh_finding()
                else:
                    self._attempt(lambda: self.review.set_speaker(index, roles))
        self.app.push_screen(SpeakerScreen(view, actions=("together", "new", "unresolved"), back=True,
                                           title=f"Who speaks finding {index + 1}?"), chosen)

    def action_new(self):
        index = self.index

        def chosen(answer):
            if answer:
                self._attempt(lambda: self.review.new_character(index, answer["name"], answer["description"]))
        self.app.push_screen(NewVoiceScreen(), chosen)

    def action_direction(self):
        index = self.index

        def chosen(text):
            if text is not None:
                self._attempt(lambda: self.review.set_direction(index, text))

        def validate(text):
            if len(text) > 240:
                raise ValueError(f"Delivery direction exceeds 240 characters (received {len(text)})")
        self.app.push_screen(TextEditScreen("How is this line spoken?", self.review.staged(index)["delivery"]["direction"],
                                            single_line=True, validate=validate,
                                            help_text="Enter saves · an empty direction clears it · Escape cancels"),
                             chosen)

    def action_scroll_story(self, direction):
        story = self.query_one("#review-story", VerticalScroll)
        story.scroll_page_down(animate=False) if direction > 0 else story.scroll_page_up(animate=False)

    def action_save(self):
        counts = self.review.counts()
        changes = counts["accept"] + counts["edit"] + counts["new"]
        if not changes:
            self.error = "Nothing to save yet: accept or edit a finding first, or Q to quit."
            self.refresh_finding()
            return
        message = (f"Save {changes} change(s)? {counts['leave'] + counts['undecided']} finding(s) stay as saved."
                   + (f" {len(self.review.new_names())} new character(s) join the cast." if counts["new"] else ""))
        self.app.push_screen(ConfirmScreen(message, title="Save these decisions?", yes="Save"),
                             lambda yes: self.dismiss("save") if yes else None)

    def action_quit(self):
        counts = self.review.counts()
        if counts["accept"] + counts["edit"] + counts["new"] == 0:
            self.dismiss("leave")
            return
        self.app.push_screen(ConfirmScreen("Quit without saving? Your decisions here are discarded and the book "
                                           "stays as it is.", title="Leave the book unchanged?",
                                           yes="Quit without saving", danger=True),
                             lambda yes: self.dismiss("leave") if yes else None)
