"""``pronounce`` (alias ``pronounce-name``): choose a word's reading by ear, then tag its uses.

Flow for ``pronounce BOOK WORD``:

1. Start from your unconfirmed draft, this book's or an inherited confirmed
   choice, ``--phonemes``, a description of the sound, or a model guess
   (``fish-phoneme-tags`` rules). ``--as`` (or ``S`` in the menu) chooses the
   parts of speech that get the reading; the default is every part of speech
   for an invented spelling and proper nouns for an ordinary English word.
2. When an ordinary word could share the spelling, or only some parts of
   speech apply, the model labels each use's part of speech first, in
   checkpointed batches, and the counts are shown.
3. Render and play up to three short sentences from the book with uses that get
   the reading (a built-in sentence when none is short enough), always ending
   with the derived plural so a later series book can use it even when this
   book never does.
4. Accept, ask the model to revise from your description, ask for a different
   guess, edit the phonemes yourself, or change the parts of speech; repeat
   until accepted. Each step is checkpointed as a draft. On accept, the
   configured recognizer transcribes the accepted test sentence once, and how
   it spells the word there is saved as a trusted heard variant
   (``heard_variants.py``) so renders do not reject that spelling.
5. You decide any use the model was unsure about (``--review-tags``: every
   model-labelled use).

``pronounce BOOK`` with no word labels and tags every defined (including
inherited) word still untagged in this book; ``--list`` shows status.
"""

from pathlib import Path
import re
import sys

from ..name_pronunciations import (PART_LABELS, book_occurrences, default_parts, describe_parts,
                                   effective_definitions, form_phonemes, in_dictionary, local_definition,
                                   local_draft, load_local, mention_key, name_key, normalize_phonemes, occurrences,
                                   parse_parts, parts_for, plural_spelling, save_decisions, save_definition,
                                   tagging_state, validate_name_phonemes, describe_reading, normalize_say_as,
                                   reading_of, validate_reading, word_pronunciation)
from ..pronunciations import validate_word
from .audition import AudioPlayer, display


MENU = ("A: accept   P: play again   B: play the default (untagged) reading   "
        "I [text]: tell the model how it should sound   N: different model guess   "
        "E [phonemes]: edit phonemes yourself (| splits groups: M AY1 L | K IY1)   "
        "R [spelling]: say it as another spelling (e.g. mile-key)   "
        "S [parts]: choose the parts of speech that get it   Q: quit (your draft is kept)")
MAX_SAMPLE_SENTENCE = 200
MAX_SAMPLE_SENTENCES = 3
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")


def test_sentence(name, source_occurrences, source_text_sample=""):
    """Several uses of the name, including the book's possessive style and the plural.

    The plural is always auditioned: a later series book may need it even
    when this book never uses it.
    """
    possessives = [item["word"] for item in source_occurrences if item["form"] == "possessive"]
    apostrophe = ("’" if "’" in possessives[0] else "'") if possessives else (
        "’" if source_text_sample.count("’") >= source_text_sample.count("'") else "'")
    return (f"The name is {name}. It is {name}{apostrophe}s name. Give it to {name}."
            f" The {plural_spelling(name)} are here.")


def with_plural_audition(name, sentence, spans):
    """An audition sentence that ends with the derived plural when it lacks one.

    A later series book may need the plural even when this book never uses
    it, so every audition lets the listener hear the derived plural reading
    (and correct the base reading) before accepting the definition. ``spans``
    are the sentence's tagged (start, end, word) uses that get the reading,
    or None to let ``sentence_controls`` find them.
    """
    plural = plural_spelling(name)
    covered = (any(word.casefold() == plural.casefold() for _, _, word in spans) if spans is not None
               else any(item["form"] == "plural" for item in occurrences(sentence, name)))
    if covered:
        return sentence, spans
    start = len(sentence) + 5  # after " The "
    sentence += f" The {plural} are here."
    return sentence, (None if spans is None else [*spans, (start, start + len(plural), plural)])


def _sentence_span(text, start, end):
    left = 0
    for match in _SENTENCE_END.finditer(text, 0, start):
        left = match.end()
    match = _SENTENCE_END.search(text, end)
    right = match.end() if match else len(text)
    while left < right and text[left].isspace():
        left += 1
    return left, right


def book_sentences(source, name, state):
    """Up to three short book sentences whose every use of the word gets the reading.

    Returns (text, spans) with spans relative to the joined text, preferring
    different forms and parts of speech; (None, []) when no sentence fits.
    """
    used = {mention_key(item) for item in state["decided"] if item["use"]}
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    by_block = {}
    for item in state["occurrences"]:
        by_block.setdefault(item["block_id"], []).append(item)
    candidates, seen = [], set()
    for item in state["decided"]:
        if not item["use"]:
            continue
        text = blocks[item["block_id"]]
        left, right = _sentence_span(text, item["start"], item["end"])
        inside = [other for other in by_block[item["block_id"]] if left <= other["start"] and other["end"] <= right]
        sentence = text[left:right]
        if (len(sentence) > MAX_SAMPLE_SENTENCE or sentence in seen
                or any(mention_key(other) not in used for other in inside)):
            continue
        seen.add(sentence)
        candidates.append((item["form"], item.get("label"), sentence,
                           [(other["start"] - left, other["end"] - left, other["word"]) for other in inside]))
    chosen, kinds = [], set()
    for candidate in sorted(candidates, key=lambda value: len(value[2])):
        if (candidate[0], candidate[1]) not in kinds and len(chosen) < MAX_SAMPLE_SENTENCES:
            kinds.add((candidate[0], candidate[1]))
            chosen.append(candidate)
    for candidate in sorted(candidates, key=lambda value: len(value[2])):
        if len(chosen) >= MAX_SAMPLE_SENTENCES:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    if not chosen:
        return None, []
    text, spans = "", []
    for _, _, sentence, inside in chosen:
        offset = len(text) + (1 if text else 0)
        text = (text + " " + sentence) if text else sentence
        spans += [(start + offset, end + offset, word) for start, end, word in inside]
    return text, spans


def sentence_controls(name, sentence, reading, spans=None):
    """Pronunciation controls for each use in a test sentence.

    ``reading`` is ``{"phonemes": …}``, ``{"say_as": …}``, or (older callers) a
    phoneme string.
    """
    if isinstance(reading, str):
        reading = {"phonemes": reading}
    if spans is None:
        spans = [(item["start"], item["end"], item["word"]) for item in occurrences(sentence, name)]
    controls = []
    for start, end, word in spans:
        pronunciation = word_pronunciation(name, reading, word)
        field = "phonemes" if pronunciation["alphabet"] == "cmu_arpabet" else "replacement"
        controls.append({"kind": "pronunciation", "start": start, "end": end, "text": word,
                         "alphabet": pronunciation["alphabet"], field: pronunciation["value"]})
    return controls


def _reading_key(reading):
    return tuple(sorted(reading.items()))


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
        self._samples = {}

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
    def _sample(self, name, parts):
        key = (name_key(name), tuple(parts))
        if key not in self._samples:
            state = tagging_state(self.directory, self.source, name, parts)
            sentence, spans = book_sentences(self.source, name, state)
            if sentence is None and "proper_noun" not in parts:
                sentence, spans = f"The word is {name}. Say {name} again: {name}.", None
            elif sentence is None:
                sentence = test_sentence(name, book_occurrences(self.source, name),
                                         " ".join(block["text"] for block in self.source["blocks"][:200]))
                spans = None
            sentence, spans = with_plural_audition(name, sentence, spans)
            self._samples[key] = (sentence, spans)
        return self._samples[key]

    def play(self, name, reading, *, default=False, parts=None):
        sentence, spans = self._sample(name, parts if parts is not None else parts_for(self.directory, name))
        controls = [] if default else sentence_controls(name, sentence, reading, spans)
        pronunciation = None if default else word_pronunciation(name, reading, name)
        preview = self.session.audition_sentence(name, sentence, controls, pronunciation=pronunciation,
                                                 voice_id=self.voice_id)
        label = "default reading (no tags)" if default else describe_reading(reading)
        self.say(f"Test sentence ({label}), voice {preview['context']['voice_id']}: {sentence}")
        self.say(f"Audio: {preview['path']}")
        self.say("Press any key to stop.")
        self.playback_fn(self.player, preview["path"])
        self.session.mark_played(preview)
        if not default:
            self._previews[_reading_key(reading)] = preview

    def ensure_labels(self, name, parts, *, announce=True):
        """Label unlabelled uses when the chosen parts need it; returns the state."""
        from ..name_pronunciation_model import describe_state, label
        state = tagging_state(self.directory, self.source, name, parts)
        if state["pending"]:
            self.say(f"Labelling the part of speech of {len(state['pending'])} use(s) of {name}…")
            label(self.bundle, name, self.requests, state["pending"], workers=self._workers(), progress=self.say,
                  on_batch=lambda decisions: save_decisions(self.directory, self.source, name, decisions))
            state = tagging_state(self.directory, self.source, name, parts)
        if announce:
            if state["labels_needed"]:
                self.say(f"Uses of {name} in this book: {describe_state(state) or 'none labelled'}"
                         + (f", {len(state['unsure'])} unsure" if state["unsure"] else "")
                         + f". Getting this reading: {describe_parts(parts)}"
                         f" ({sum(bool(item['use']) for item in state['decided'])} use(s)).")
            else:
                self.say(f"{name} is not an ordinary English word, so all {len(state['occurrences'])} use(s) get "
                         "this reading (no labelling needed).")
        return state

    def _save(self, name, reading, *, status="draft", source, respelling=None, instruction=None, parts=None):
        save_definition(self.directory, name, reading.get("phonemes"), say_as=reading.get("say_as"), status=status,
                        source=source, respelling=respelling if "phonemes" in reading else None,
                        instruction=instruction, parts_of_speech=parts)

    def define(self, name, *, phonemes=None, sound=None, parts=None, say_as=None):
        """Listening loop; returns the accepted reading, or None on quit.

        A reading is ``{"phonemes": …}`` (CMU Arpabet, optionally in ``|``
        groups) or ``{"say_as": …}`` (a spelling the engine reads instead).
        """
        from ..name_pronunciation_model import guess
        key = name_key(name)
        found = book_occurrences(self.source, name)
        if not found:
            raise ValueError(f"{name} does not occur in this book's source text")
        effective = effective_definitions(self.directory).get(key)
        draft = local_draft(self.directory, name)
        chosen_parts = parts
        parts = (parts or (draft or {}).get("parts_of_speech") or (effective or {}).get("parts_of_speech")
                 or default_parts(name))
        history = [step["phonemes"] for step in (draft or local_definition(self.directory, name) or {}).get("history", [])
                   if "phonemes" in step]
        respelling = None
        if phonemes is not None or say_as is not None:
            current = validate_reading({"phonemes": normalize_phonemes(phonemes)} if phonemes is not None
                                       else {"say_as": normalize_say_as(say_as)})
            self._save(name, current, source="command_line", parts=parts)
            last_source = "command_line"
        elif sound is not None:
            # The listener's own description wins over any saved draft or inherited reading.
            self.say(f"Asking the model for phonemes that sound like: {sound}")
            result = guess(self.bundle, name, self.requests, found, instruction=sound, parts=parts)
            current, respelling = {"phonemes": result["phonemes"]}, result["respelling"]
            self._save(name, current, source="model_revision", respelling=respelling, instruction=sound, parts=parts)
            last_source = "model_revision"
        elif draft is not None:
            current, respelling = reading_of(draft), draft["respelling"]
            last_source = draft["history"][-1]["source"]
            self.say(f"Resuming your unconfirmed choice for {name}.")
        elif effective is not None:
            current, respelling = reading_of(effective), effective["respelling"]
            last_source = "inherited" if effective["inherited"] else effective["history"][-1]["source"]
            where = f"inherited from {effective['origin_directory']}" if effective["inherited"] else "in this book"
            self.say(f"{name} is already defined ({where}). Accept it as is, or modify it.")
        else:
            self.say(f"Asking the model how to pronounce {name}…")
            result = guess(self.bundle, name, self.requests, found, parts=parts)
            current, respelling = {"phonemes": result["phonemes"]}, result["respelling"]
            self._save(name, current, source="model", respelling=respelling, parts=parts)
            last_source = "model"
        if (chosen_parts is not None and (draft or effective) is not None and phonemes is None and say_as is None
                and sound is None):
            self._save(name, current, source=last_source, respelling=respelling, parts=parts)
        counts = {}
        for item in found:
            counts[item["form"]] = counts.get(item["form"], 0) + 1
        self.say(f"{name}: {len(found)} possible uses in this book ("
                 + ", ".join(f"{count} {form}" for form, count in sorted(counts.items())) + ").")
        if in_dictionary(name) and parts == ["proper_noun"] and chosen_parts is None:
            self.say(f"{name} is also an ordinary English word, so this reading applies only where it is used as a "
                     "proper noun. Use S to choose other parts of speech.")
        self.ensure_labels(name, parts)
        play_next = True
        while True:
            hint = f" (say it like {respelling})" if respelling and "phonemes" in current else ""
            self.say(f"Current reading of {name}: {describe_reading(current)}{hint}; applies to {describe_parts(parts)}")
            possessive, plural = name + chr(39) + "s", plural_spelling(name)
            possessive_reading = word_pronunciation(name, current, possessive)["value"]
            plural_reading = word_pronunciation(name, current, plural)["value"]
            prefix = "" if "phonemes" in current else "said as "
            self.say(f"Possessive/plural derive from it: {possessive} → {prefix}{possessive_reading}; "
                     f"{plural} → {prefix}{plural_reading}")
            if play_next:
                play_next = False
                try:
                    self.play(name, current, parts=parts)
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
                    self._accept(name, current, respelling, last_source, effective, parts)
                    self._learn_heard(name, current)
                    return current
                if verb in {"q", "quit"}:
                    self.say(f"No change confirmed for {name}; your draft is kept for the next run.")
                    return None
                if verb in {"p", "play"}:
                    self.play(name, current, parts=parts)
                elif verb == "b":
                    self.play(name, current, default=True, parts=parts)
                elif verb == "s":
                    value = rest or self.input_fn(
                        f"Parts of speech for {name} (now {describe_parts(parts)}; e.g. proper noun, adjective, "
                        "or all; empty cancels): ").strip()
                    if not value:
                        continue
                    parts = parse_parts(value)
                    self._save(name, current, source=last_source, respelling=respelling, parts=parts)
                    self.ensure_labels(name, parts)
                    play_next = True
                elif verb in {"i", "n"}:
                    instruction = rest if verb == "i" else "Give a different plausible reading than the ones already heard."
                    if not instruction:
                        instruction = self.input_fn("How should it sound? (e.g. more like sal-vuh-TORE-ee): ").strip()
                    if not instruction:
                        continue
                    if "say_as" in current:
                        # The model writes phonemes; tell it what the say-as spelling sounded like.
                        instruction = f"Currently said as the spelling {current['say_as']}. {instruction}"
                    else:
                        history.append(current["phonemes"])
                    self.say("Asking the model to revise…" if verb == "i" else "Asking the model for a different guess…")
                    result = guess(self.bundle, name, self.requests, found, current=current.get("phonemes"),
                                   instruction=instruction, rejected=history, parts=parts)
                    current, respelling = {"phonemes": result["phonemes"]}, result["respelling"]
                    self._save(name, current, source="model_revision", respelling=respelling, instruction=instruction,
                               parts=parts)
                    last_source = "model_revision"
                    play_next = True
                elif verb == "e":
                    now = current["phonemes"] if "phonemes" in current else "none"
                    value = rest or self.input_fn(f"Phonemes (now {now}; | splits groups, e.g. M AY1 L | K IY1; "
                                                  "empty cancels): ").strip()
                    if not value:
                        continue
                    proposed = validate_reading({"phonemes": normalize_phonemes(value)})
                    if "phonemes" in current:
                        history.append(current["phonemes"])
                    current, respelling = proposed, None
                    self._save(name, current, source="manual")
                    last_source = "manual"
                    play_next = True
                elif verb == "r":
                    now = current["say_as"] if "say_as" in current else "none"
                    value = rest or self.input_fn(f"Say {name} as this spelling instead (now {now}; e.g. mile-key; "
                                                  "empty cancels): ").strip()
                    if not value:
                        continue
                    proposed = validate_reading({"say_as": normalize_say_as(value)})
                    if "phonemes" in current:
                        history.append(current["phonemes"])
                    current, respelling = proposed, None
                    self._save(name, current, source="manual")
                    last_source = "manual"
                    play_next = True
                elif verb:
                    raise ValueError("Use A, P, B, I, N, E, R, S or Q")
            except (ValueError, RuntimeError, OSError) as exc:
                from .diagnostics import configured_secrets, safe_error_text
                self.say("Action stopped: " + safe_error_text(exc, secrets=configured_secrets(self.config)))

    def _accept(self, name, current, respelling, source, effective, parts):
        """Confirm locally unless this exact reading and parts are already the effective ones."""
        from ..name_pronunciations import discard_draft
        local = local_definition(self.directory, name)
        same = lambda value: reading_of(value) == current and value["parts_of_speech"] == parts
        shown = describe_reading(current)
        if local is not None and same(local):
            discard_draft(self.directory, name)
            self.say(f"Keeping {name}: {shown} ({describe_parts(parts)}).")
            return
        if local is None and effective is not None and effective["inherited"] and same(effective):
            discard_draft(self.directory, name)
            self.say(f"Keeping the inherited reading of {name}: {shown} ({describe_parts(parts)}).")
            return
        self._save(name, current, status="confirmed", source=source, respelling=respelling, parts=parts)
        self.say(f"Confirmed {name}: {shown} for {describe_parts(parts)}. Existing audio is unchanged.")

    def _learn_heard(self, name, reading):
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
        preview = self._previews.get(_reading_key(reading))
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
        """Label uses and settle unsure ones; returns True when every current use is decided."""
        from ..name_pronunciation_model import describe_state
        parts = parts_for(self.directory, name)
        state = self.ensure_labels(name, parts, announce=False)
        asked = list(state["unsure"])
        if review:
            asked += [item for item in state["decided"] if item["basis"] == "model"]
        for index, item in enumerate(asked, 1):
            if not self.interactive:
                break
            label = ("unsure" if item["label"] == "unsure"
                     else f"labelled {PART_LABELS.get(item['label'], item['label'])}, "
                          + ("gets the reading" if item["use"] else "keeps the default reading"))
            self.say(f"[{index}/{len(asked)}] {label}: {_excerpt(self.source, item)}")
            answer = self.input_fn(f"Should »{item['word']}« here get the reading of {name}? y/n "
                                   "(Enter keeps, Q stops): ").strip().casefold()
            if answer == "q":
                break
            if answer in {"y", "n"}:
                decision = {key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
                save_decisions(self.directory, self.source, name,
                               [dict(decision, basis="user", use=answer == "y")])
        state = tagging_state(self.directory, self.source, name, parts)
        used = [item for item in state["decided"] if item["use"]]
        others = [item for item in state["decided"] if not item["use"]]
        self.say(f"{name} ({describe_parts(parts)}): {len(used)} use(s) get the reading; {len(others)} keep the "
                 f"default reading; {len(state['unsure'])} unsure; {len(state['pending'])} unlabelled."
                 + (f" Labels: {describe_state(state)}." if state["label_counts"] else ""))
        for item in others[:10]:
            how = PART_LABELS.get(item["label"], "your choice") if item["basis"] != "user" else "your choice"
            self.say(f"  default reading ({how}): " + _excerpt(self.source, item, 70))
        if others and not review:
            self.say("Use --review-tags to check or change the model's labels.")
        if not state["ready"]:
            self.say(f"Rendering stays blocked until every use of {name} is decided; rerun this command.")
        return state["ready"]

    def tag_all(self, *, review=False):
        definitions = effective_definitions(self.directory)
        if not definitions:
            self.say("No pronunciations are defined for this book or its series ancestors.")
            return True
        ready = True
        for _, definition in sorted(definitions.items()):
            state = tagging_state(self.directory, self.source, definition["name"], definition["parts_of_speech"])
            if state["ready"] and not review:
                continue
            ready = self.tag(definition["name"], review=review) and ready
        if ready:
            self.say("Every defined word is tagged in this book.")
        return ready

    def show(self):
        definitions = effective_definitions(self.directory)
        drafts = load_local(self.directory)["drafts"]
        if not definitions and not drafts:
            self.say("No pronunciations are defined for this book or its series ancestors.")
        for key in sorted({*definitions, *drafts}):
            definition = definitions.get(key)
            if definition is not None:
                state = tagging_state(self.directory, self.source, definition["name"], definition["parts_of_speech"])
                origin = f"inherited from {definition['origin_directory']}" if definition["inherited"] else "this book"
                status = "tagged" if state["ready"] else f"{len(state['pending'])} untagged, {len(state['unsure'])} unsure"
                tagged = sum(bool(item["use"]) for item in state["decided"])
                self.say(f"{definition['name']}: {describe_reading(reading_of(definition))} for {describe_parts(definition['parts_of_speech'])} "
                         f"({origin}); {tagged} tagged uses; {status}")
            if key in drafts:
                self.say(f"{drafts[key]['name']}: unconfirmed draft {describe_reading(reading_of(drafts[key]))} "
                         f"for {describe_parts(drafts[key]['parts_of_speech'])}")

    def close(self):
        if self.player is not None:
            self.player.stop()
        close = getattr(self._verifier, "close", None)
        if callable(close):
            close()
        self.session.close()


def _remove(directory, name, output):
    from ..name_pronunciations import remove_definition
    removed = remove_definition(directory, name)
    if removed is None:
        output(display(f"Removed the unconfirmed draft for {name}; nothing was confirmed in this book."))
    else:
        reading = reading_of(removed)
        option = (f"--say-as {reading['say_as']}" if "say_as" in reading else f"--phonemes \"{reading['phonemes']}\"")
        output(display(f"Removed {removed['name']} ({describe_reading(reading)}) from {directory}. To define it again: "
                       f"pronounce BOOK {removed['name']} {option} --as \"{describe_parts(removed['parts_of_speech'])}\""))
    now = effective_definitions(directory).get(name_key(name))
    if now is not None:
        output(display(f"This book now inherits {now['name']} ({describe_reading(reading_of(now))}) from {now['origin_directory']}."))
    else:
        output(display(f"{name} now uses the engine's default reading here and in books continued from this one, "
                       "unless one of them defines it; re-render affected chapters to hear the change."))
    return 0


def run_pronounce_name(bundle, config, args, *, analysis_config, input_fn=None, output=print, engine_factory=None,
                       store=None, player=None, playback_fn=None, requests=None, interactive=None,
                       verifier_factory=None):
    name = getattr(args, "name", None)
    if name is not None:
        validate_word(name)
    if getattr(args, "remove", False):
        if name is None:
            raise ValueError("Name the pronunciation to remove, e.g. pronounce-name BOOK Salvatore --remove")
        if any(getattr(args, field, None) for field in ("sound", "phonemes", "say_as", "tag_only", "review_tags", "list",
                                                         "parts_of_speech")):
            raise ValueError("--remove cannot be combined with other pronounce choices")
        return _remove(Path(bundle["directory"]), name, output)
    sound = getattr(args, "sound", None)
    if sound is not None:
        sound = " ".join(sound.split())
        if name is None or not sound or len(sound) > 1000:
            raise ValueError("Give the name first, then how it should sound in up to 1000 characters, e.g. pronounce-name BOOK Salvatore 'sal-vuh-TORE'")
        if (getattr(args, "phonemes", None) is not None or getattr(args, "say_as", None) is not None
                or getattr(args, "tag_only", False)):
            raise ValueError("Choose how it sounds, --phonemes or --say-as, not more than one, and not with --tag-only")
    if getattr(args, "say_as", None) is not None and (name is None or getattr(args, "phonemes", None) is not None
                                                      or getattr(args, "tag_only", False)):
        raise ValueError("--say-as needs a word, and cannot be combined with --phonemes or --tag-only")
    parts = getattr(args, "parts_of_speech", None)
    if parts is not None:
        if name is None or getattr(args, "tag_only", False) or getattr(args, "list", False):
            raise ValueError("--as needs a word to define and cannot be combined with --tag-only or --list")
        parts = parse_parts(parts)
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
            if tool.define(name, phonemes=getattr(args, "phonemes", None), sound=sound, parts=parts,
                           say_as=getattr(args, "say_as", None)) is None:
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
