"""``pronounce-name``: choose a name's reading by ear, then tag its real uses.

Flow for ``pronounce-name BOOK NAME``:

1. Start from your unconfirmed draft, this book's or an inherited confirmed
   choice, ``--phonemes``, or a model guess (``fish-phoneme-tags`` rules).
2. Render and play a short test sentence that uses the name several times,
   including its possessive (and plural when the book uses one).
3. Accept, ask the model to revise from your description, ask for a different
   guess, or edit the phonemes yourself; repeat until accepted. Each step is
   checkpointed as a draft, so quitting and rerunning resumes it. On accept,
   the configured recognizer transcribes the accepted test sentence once, and
   how it spells the name there is saved as a trusted heard variant
   (``heard_variants.py``) so renders do not reject that spelling.
4. Tag every occurrence that really is the name (homonyms stay untagged), with
   per-batch checkpoints; you decide any the model is unsure about.

``pronounce-name BOOK`` with no name tags every defined (including inherited)
name still untagged in this book; ``--list`` shows status.
"""

from pathlib import Path
import sys

from ..name_pronunciations import (book_occurrences, effective_definitions, form_phonemes, local_definition,
                                   local_draft, load_local,
                                   name_key, normalize_phonemes, occurrences, plural_spelling, save_decisions,
                                   save_definition, tagging_state, validate_name_phonemes)
from ..pronunciations import validate_word
from .audition import AudioPlayer, display


MENU = ("A: accept   P: play again   B: play the default (untagged) reading   "
        "I [text]: tell the model how it should sound   N: different model guess   "
        "E [phonemes]: edit phonemes yourself   Q: quit (your draft is kept)")


def test_sentence(name, source_occurrences, source_text_sample=""):
    """Several uses of the name, including the book's possessive style."""
    possessives = [item["word"] for item in source_occurrences if item["form"] == "possessive"]
    apostrophe = ("’" if "’" in possessives[0] else "'") if possessives else (
        "’" if source_text_sample.count("’") >= source_text_sample.count("'") else "'")
    sentence = f"The name is {name}. It is {name}{apostrophe}s name. Give it to {name}."
    if any(item["form"] == "plural" for item in source_occurrences):
        sentence += f" The {plural_spelling(name)} are here."
    return sentence


def sentence_controls(name, sentence, phonemes):
    forms = form_phonemes(name, phonemes)
    return [{"kind": "pronunciation", "start": item["start"], "end": item["end"], "text": item["word"],
             "alphabet": "cmu_arpabet",
             "phonemes": forms[item["word"].casefold()][1] if item["word"].casefold() in forms else phonemes}
            for item in occurrences(sentence, name)]


def _excerpt(source, item, width=160):
    text = next(block["text"] for block in source["blocks"] if block["id"] == item["block_id"])
    left, right = max(0, item["start"] - width), min(len(text), item["end"] + width)
    return ("…" if left else "") + text[left:item["start"]] + "»" + text[item["start"]:item["end"]] + "«" + \
        text[item["end"]:right] + ("…" if right < len(text) else "")


class NameTool:
    def __init__(self, bundle, config, *, analysis_config, input_fn, output, engine_factory=None, store=None,
                 player=None, playback_fn=None, requests=None, interactive=True, voice_id=None,
                 verifier_factory=None):
        from .pronunciation_review import PronunciationReviewSession
        self.bundle, self.config = bundle, config
        self.directory = Path(bundle["directory"])
        self.source = bundle["source"]
        self._analysis_config = analysis_config
        self.input_fn, self.output = input_fn, output
        self.session = PronunciationReviewSession(bundle, config, engine_factory=engine_factory, store=store,
                                                  notify=self.say)
        self.player, self.playback_fn = player, playback_fn
        self._requests = requests
        self.interactive = interactive
        self.voice_id = voice_id
        # The configured speech recognizer hears each confirmed test sentence
        # once, so renders accept how it spells the name (heard_variants.py).
        self._verifier_factory, self._verifier = verifier_factory, None
        self._previews = {}

    def say(self, value):
        self.output(display(value))

    @property
    def requests(self):
        if self._requests is None:
            from ..name_pronunciation_model import make_requests
            self._requests = make_requests(self.directory, self._analysis_config(), progress=self.say)
        return self._requests

    def _workers(self):
        try:
            return max(1, min(4, int(self._analysis_config()["analysis"]["attribution_workers"])))
        except (KeyError, TypeError, ValueError):
            return 1

    # Listening -------------------------------------------------------------
    def play(self, name, phonemes, *, default=False):
        sentence = test_sentence(name, book_occurrences(self.source, name),
                                 " ".join(block["text"] for block in self.source["blocks"][:200]))
        controls = [] if default else sentence_controls(name, sentence, phonemes)
        pronunciation = None if default else {"alphabet": "cmu_arpabet", "value": phonemes}
        preview = self.session.audition_sentence(name, sentence, controls, pronunciation=pronunciation,
                                                 voice_id=self.voice_id)
        label = "default reading (no tags)" if default else phonemes
        self.say(f"Test sentence ({label}), voice {preview['context']['voice_id']}: {sentence}")
        self.say(f"Audio: {preview['path']}")
        self.say("Press any key to stop.")
        self.playback_fn(self.player, preview["path"])
        self.session.mark_played(preview)
        if not default:
            self._previews[phonemes] = preview

    def define(self, name, *, phonemes=None):
        """Listening loop; returns the accepted phonemes, or None on quit."""
        from ..name_pronunciation_model import guess
        key = name_key(name)
        found = book_occurrences(self.source, name)
        if not found:
            raise ValueError(f"{name} does not occur in this book's source text")
        effective = effective_definitions(self.directory).get(key)
        draft = local_draft(self.directory, name)
        history = [step["phonemes"] for step in (draft or local_definition(self.directory, name) or {}).get("history", [])]
        respelling = None
        if phonemes is not None:
            current = validate_name_phonemes(normalize_phonemes(phonemes))
            save_definition(self.directory, name, current, status="draft", source="command_line")
            last_source = "command_line"
        elif draft is not None:
            current, respelling = draft["phonemes"], draft["respelling"]
            last_source = draft["history"][-1]["source"]
            self.say(f"Resuming your unconfirmed choice for {name}.")
        elif effective is not None:
            current, respelling = effective["phonemes"], effective["respelling"]
            last_source = "inherited" if effective["inherited"] else effective["history"][-1]["source"]
            where = f"inherited from {effective['origin_directory']}" if effective["inherited"] else "in this book"
            self.say(f"{name} is already defined ({where}). Accept it as is, or modify it.")
        else:
            self.say(f"Asking the model how to pronounce {name}…")
            result = guess(self.bundle, name, self.requests, found)
            current, respelling = result["phonemes"], result["respelling"]
            save_definition(self.directory, name, current, status="draft", source="model", respelling=respelling)
            last_source = "model"
        counts = {}
        for item in found:
            counts[item["form"]] = counts.get(item["form"], 0) + 1
        self.say(f"{name}: {len(found)} possible uses in this book ("
                 + ", ".join(f"{count} {form}" for form, count in sorted(counts.items())) + ").")
        play_next = True
        while True:
            hint = f" (say it like {respelling})" if respelling else ""
            self.say(f"Current reading of {name}: {current}{hint}")
            forms = form_phonemes(name, current)
            self.say(f"Possessive/plural derive from it: {name}'s → {forms[(name + chr(39) + 's').casefold()][1]}")
            if play_next:
                play_next = False
                try:
                    self.play(name, current)
                except (ValueError, RuntimeError, OSError) as exc:
                    from .diagnostics import configured_secrets, safe_error_text
                    self.say("Playback stopped: " + safe_error_text(exc, secrets=configured_secrets(self.config))
                             + ". Use P to try again.")
            self.say(MENU)
            command = self.input_fn("> ").strip()
            verb, _, rest = command.partition(" ")
            verb, rest = verb.casefold(), rest.strip()
            try:
                if verb in {"a", "accept"}:
                    self._accept(name, current, respelling, last_source, effective)
                    self._learn_heard(name, current)
                    return current
                if verb in {"q", "quit"}:
                    self.say(f"No change confirmed for {name}; your draft is kept for the next run.")
                    return None
                if verb in {"p", "play"}:
                    self.play(name, current)
                elif verb == "b":
                    self.play(name, current, default=True)
                elif verb in {"i", "n"}:
                    instruction = rest if verb == "i" else "Give a different plausible reading than the ones already heard."
                    if not instruction:
                        instruction = self.input_fn("How should it sound? (e.g. more like sal-vuh-TORE-ee): ").strip()
                    if not instruction:
                        continue
                    history.append(current)
                    self.say("Asking the model to revise…" if verb == "i" else "Asking the model for a different guess…")
                    result = guess(self.bundle, name, self.requests, found, current=current,
                                   instruction=instruction, rejected=history)
                    current, respelling = result["phonemes"], result["respelling"]
                    save_definition(self.directory, name, current, status="draft", source="model_revision",
                                    respelling=respelling, instruction=instruction)
                    last_source = "model_revision"
                    play_next = True
                elif verb == "e":
                    value = rest or self.input_fn(f"Phonemes (now {current}; empty cancels): ").strip()
                    if not value:
                        continue
                    proposed = validate_name_phonemes(normalize_phonemes(value))
                    history.append(current)
                    current, respelling = proposed, None
                    save_definition(self.directory, name, current, status="draft", source="manual")
                    last_source = "manual"
                    play_next = True
                elif verb:
                    raise ValueError("Use A, P, B, I, N, E or Q")
            except (ValueError, RuntimeError, OSError) as exc:
                from .diagnostics import configured_secrets, safe_error_text
                self.say("Action stopped: " + safe_error_text(exc, secrets=configured_secrets(self.config)))

    def _accept(self, name, current, respelling, source, effective):
        """Confirm locally unless this exact reading is already the effective one."""
        from ..name_pronunciations import discard_draft
        local = local_definition(self.directory, name)
        if local is not None and local["phonemes"] == current:
            discard_draft(self.directory, name)
            self.say(f"Keeping {name}: {current}.")
            return
        if local is None and effective is not None and effective["inherited"] and effective["phonemes"] == current:
            discard_draft(self.directory, name)
            self.say(f"Keeping the inherited reading of {name}: {current}.")
            return
        save_definition(self.directory, name, current, status="confirmed", source=source, respelling=respelling)
        self.say(f"Confirmed {name}: {current}. Existing audio is unchanged.")

    def _learn_heard(self, name, phonemes):
        """Record how the recognizer hears the accepted test sentence's name.

        The person just accepted this audio, so a recognizer spelling of the
        name in it is trusted ("listened") for renders with the same
        recognizer. Any other difference in the sentence means the preview was
        not heard cleanly, and nothing is kept.
        """
        from ..name_pronunciations import form_spellings
        from .common import file_hash, regular_file
        from .heard_variants import learn_from_listening
        from .quality import normalize_words
        if self.config.get("verification", {}).get("heard_variants", "learn") != "learn":
            return
        preview = self._previews.get(phonemes)
        if preview is None:
            return
        try:
            if self._verifier is None:
                if self._verifier_factory is not None:
                    self._verifier = self._verifier_factory()
                else:
                    from .verification import create_verifier
                    self._verifier = create_verifier(self.config["verification"])
            result = self._verifier.transcribe(Path(preview["path"]))
            transcript = result.get("text") if isinstance(result, dict) else None
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("the recognizer returned no transcript")
            words = {word for form in form_spellings(name) for word in normalize_words(form)}
            learned = learn_from_listening(self.directory, self._verifier, preview["request"]["text"], transcript,
                                           words, audio_sha256=file_hash(regular_file(Path(preview["path"]))))
        except Exception as exc:  # Optional: never undoes the confirmed choice.
            from .diagnostics import configured_secrets, safe_error_text
            self.say(f"Skipped learning how the recognizer hears {name}: "
                     + safe_error_text(exc, secrets=configured_secrets(self.config)))
            return
        if learned:
            self.say(f"The recognizer hears {name} as: " + "; ".join(f"{heard} (for {expected})"
                                                                    for expected, heard in learned)
                     + ". Renders accept that spelling for this name.")
        else:
            self.say(f"The recognizer heard the test sentence for {name} as written, or not cleanly enough to learn from.")

    # Tagging ---------------------------------------------------------------
    def tag(self, name, *, review=False):
        """Tag real uses; returns True when every current occurrence is decided."""
        from ..name_pronunciation_model import tag
        state = tagging_state(self.directory, self.source, name)
        if state["pending"]:
            self.say(f"Tagging {len(state['pending'])} occurrence(s) of {name} (homonyms stay untagged)…")
            tag(self.bundle, name, self.requests, state["pending"], workers=self._workers(), progress=self.say,
                on_batch=lambda decisions: save_decisions(self.directory, self.source, name, decisions))
            state = tagging_state(self.directory, self.source, name)
        asked = list(state["unsure"])
        if review:
            asked += [item for item in state["decided"] if item["basis"] == "model"]
        for index, item in enumerate(asked, 1):
            if not self.interactive:
                break
            label = "unsure" if item["decision"] == "unsure" else f"model said {item['decision']}"
            self.say(f"[{index}/{len(asked)}] {label}: {_excerpt(self.source, item)}")
            answer = self.input_fn(f"Is »{item['word']}« the name {name}? y/n (Enter keeps, Q stops): ").strip().casefold()
            if answer == "q":
                break
            if answer in {"y", "n"}:
                decision = {key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
                save_decisions(self.directory, self.source, name,
                               [dict(decision, decision="name" if answer == "y" else "other", basis="user")])
        state = tagging_state(self.directory, self.source, name)
        names = [item for item in state["decided"] if item["decision"] == "name"]
        others = [item for item in state["decided"] if item["decision"] == "other"]
        self.say(f"{name}: {len(names)} use(s) tagged as the name; {len(others)} other use(s) keep the default "
                 f"reading; {len(state['unsure'])} unsure; {len(state['pending'])} untagged.")
        for item in others[:10]:
            self.say("  not the name: " + _excerpt(self.source, item, 70))
        if others and not review:
            self.say("Use --review-tags to check or change the model's decisions.")
        if not state["ready"]:
            self.say(f"Rendering stays blocked until every use of {name} is decided; rerun this command.")
        return state["ready"]

    def tag_all(self, *, review=False):
        definitions = effective_definitions(self.directory)
        if not definitions:
            self.say("No name pronunciations are defined for this book or its series ancestors.")
            return True
        ready = True
        for _, definition in sorted(definitions.items()):
            state = tagging_state(self.directory, self.source, definition["name"])
            if state["ready"] and not review:
                continue
            ready = self.tag(definition["name"], review=review) and ready
        if ready:
            self.say("Every defined name is tagged in this book.")
        return ready

    def show(self):
        definitions = effective_definitions(self.directory)
        drafts = load_local(self.directory)["drafts"]
        if not definitions and not drafts:
            self.say("No name pronunciations are defined for this book or its series ancestors.")
        for key in sorted({*definitions, *drafts}):
            definition = definitions.get(key)
            if definition is not None:
                state = tagging_state(self.directory, self.source, definition["name"])
                origin = f"inherited from {definition['origin_directory']}" if definition["inherited"] else "this book"
                status = "tagged" if state["ready"] else f"{len(state['pending'])} untagged, {len(state['unsure'])} unsure"
                tagged = sum(item["decision"] == "name" for item in state["decided"])
                self.say(f"{definition['name']}: {definition['phonemes']} ({origin}); {tagged} tagged uses; {status}")
            if key in drafts:
                self.say(f"{drafts[key]['name']}: unconfirmed draft {drafts[key]['phonemes']}")

    def close(self):
        if self.player is not None:
            self.player.stop()
        close = getattr(self._verifier, "close", None)
        if callable(close):
            close()
        self.session.close()


def run_pronounce_name(bundle, config, args, *, analysis_config, input_fn=None, output=print, engine_factory=None,
                       store=None, player=None, playback_fn=None, requests=None, interactive=None,
                       verifier_factory=None):
    name = getattr(args, "name", None)
    if name is not None:
        validate_word(name)
    if input_fn is None:
        from .terminal import enable_line_editing
        enable_line_editing()
        input_fn = input
    if playback_fn is None:
        from .terminal import play_sample
        playback_fn = play_sample
    if interactive is None:
        interactive = sys.stdin.isatty()
    player = player or AudioPlayer(config.get("casting", {}).get("player", ["/usr/bin/afplay"]))
    tool = NameTool(bundle, config, analysis_config=analysis_config, input_fn=input_fn, output=output,
                    engine_factory=engine_factory, store=store, player=player, playback_fn=playback_fn,
                    requests=requests, interactive=interactive, voice_id=getattr(args, "voice_id", None),
                    verifier_factory=verifier_factory)
    review = bool(getattr(args, "review_tags", False))
    try:
        if getattr(args, "list", False):
            tool.show()
            return 0
        if name is None:
            return 0 if tool.tag_all(review=review) else 2
        if not getattr(args, "tag_only", False):
            if tool.define(name, phonemes=getattr(args, "phonemes", None)) is None:
                return 0
        elif effective_definitions(tool.directory).get(name_key(name)) is None:
            raise ValueError(f"{name} has no confirmed pronunciation here or in a series ancestor")
        ready = tool.tag(name, review=review)
        tool.say("Rendering uses the tagged reading for every tagged use; re-render chapters to hear it.")
        return 0 if ready else 2
    except EOFError:
        tool.say("Closed; your draft and tagging progress are saved.")
        return 0
    finally:
        tool.close()
