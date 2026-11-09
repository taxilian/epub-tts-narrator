"""Experimental "narrator read" analysis: one sequential prep read, then markup.

A human narrator reads the book in order keeping a running cast list, then marks
up the script. This module does the same with few, terse requests:

1. prep: sequential chapter-aligned windows; the model returns only ledger
   deltas (new characters, aliases, voice quotes, merges, questions).
2. merge check: one batched yes/no check of merges made during prep.
3. markup: parallel chapter groups; the model labels locally detected
   quotation IDs with ledger IDs (plus rare explicit-anchor spans).
4. verify: one batched check of unresolved labels and labels that disagree
   with an explicit speech tag, with wider context.
5. voices: batched terse voice design for non-inherited voices.

Code owns source text, offsets, IDs, segments and validation; every response is
cached by CachedRequests, so rerunning replays completed calls without model
requests. Book text is untrusted data in every prompt. Remaining uncertainty
stays unresolved for the human speaker menu; nothing falls back to the narrator.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import re
import threading
import time

from .narrator_ledger import (EFFECT_PRESETS, IDENTITY_CHANGES, apply_delta, character, contextual_series_evidence, exact_quote, ledger_id, merge,
                              name_index, new_ledger, recheck_aliases, render, tag_character, undo_merge, _clean, _norm)
from .narrator_prompts import (ACCENTS, HINTS, MARKUP, MERGE_CHECK, PREP, SERIES_CHECK, VERIFY, VERIFY_TIEBREAK, VOICE,
                               VERSION as PROMPT_VERSION)
from .performance_plan import DELIVERY_TAGS, VOCAL_SOUNDS
from .provider import OutputTruncatedError
from .quote_spans import _NOT_NAMES, quote_spans, spans_by_block
from .speech_traits import validate_speech_traits

VERSION = 1
STRATEGY = "narrator_read"
PREP_WINDOW_CHARS = 48000
MARKUP_CHARS = 30000
VERIFY_BATCH = 25
VOICE_BATCH = 40
CONTEXT_BEFORE = 800
CONTEXT_AFTER = 1500
MARKER_OPEN, MARKER_CLOSE = "‹", "›"
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
STAGES = ("prep", "identity_check", "markup", "verify", "voices", "performance")
# Which operator notes each request mode receives: cast notes steer the ledger
# and identity checks, attribution notes steer markup and verification.
CAST_HINT_MODES = frozenset({"narrator_prep", "narrator_merge_check", "narrator_alias_check",
                             "narrator_duplicate_check", "narrator_series_check"})
ATTRIBUTION_HINT_MODES = frozenset({"narrator_markup", "narrator_verify", "narrator_verify_tiebreak"})


def settings_notes(settings):
    """The narrator read's notes from ``[analysis]`` settings, combined as the
    consensus pipeline combines them: cast review plus reconciliation notes for
    the cast, attribution review notes (else cast review notes) for attribution."""
    cast_review = settings.get("cast_review_notes") or ""
    cast = "\n".join(note for note in [cast_review, *(settings.get("cast_reconciliation_notes") or [])] if note)
    return {"cast_notes": cast, "attribution_notes": settings.get("attribution_review_notes") or cast_review}


def hint_modes(cast_notes="", attribution_notes=""):
    """Request mode -> operator notes, for the modes whose notes are non-empty."""
    hints = {}
    for modes, notes in ((CAST_HINT_MODES, cast_notes), (ATTRIBUTION_HINT_MODES, attribution_notes)):
        if not isinstance(notes, str):
            raise ValueError("Narrator-read notes must be strings")
        if notes.strip():
            hints.update(dict.fromkeys(modes, notes.strip()))
    return hints


class Meter:
    """The narrator read's request gate: adds operator notes, and accounts
    requests per stage from provider usage reports (numbers only)."""

    def __init__(self, client, hints=None):
        self.hints = dict(hints or {})
        self.lock = threading.Lock()
        self.local = threading.local()
        self.stats = {stage: {"logical_requests": 0, "provider_calls": 0, "prompt_tokens": 0,
                              "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0,
                              "usage_missing": 0, "provider_seconds": 0.0, "request_bytes": 0,
                              "response_bytes": 0, "wall_seconds": 0.0} for stage in STAGES}
        self.client = client
        # Calls made outside meter.run (the shared inline-performance review runs in
        # its own worker threads) are attributed to this stage while it is set.
        self.default_stage = None
        self.previous = getattr(client, "usage_observer", None)
        if hasattr(client, "usage_observer"):
            client.usage_observer = self.observe
        if hasattr(client, "include_usage"):
            client.include_usage = True

    def close(self):
        if hasattr(self.client, "usage_observer"):
            self.client.usage_observer = self.previous

    def observe(self, record):
        stage = getattr(self.local, "stage", None) or self.default_stage
        if self.previous is not None:
            try:
                self.previous(record)
            except Exception:
                pass
        if stage not in self.stats:
            return
        with self.lock:
            row = self.stats[stage]
            row["provider_calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
                if type(record.get(key)) is int:
                    row[key] += record[key]
            if record.get("prompt_tokens") is None:
                row["usage_missing"] += 1
            row["provider_seconds"] += float(record.get("elapsed_seconds") or 0)
            row["request_bytes"] += int(record.get("request_bytes") or 0)
            row["response_bytes"] += int(record.get("response_bytes") or 0)

    def run(self, requests, stage, system, payload, validate):
        self.local.stage = stage
        try:
            with self.lock:
                self.stats[stage]["logical_requests"] += 1
            hint = self.hints.get(payload.get("mode"))
            if hint:
                system, payload = system + HINTS, {**payload, "hints": hint}
            return requests.run(system, payload, validate)
        finally:
            self.local.stage = None

    def timed(self, stage, seconds):
        with self.lock:
            self.stats[stage]["wall_seconds"] += seconds

    def summary(self):
        total = {key: sum(row[key] for row in self.stats.values()) for key in next(iter(self.stats.values()))}
        return {"stages": deepcopy(self.stats), "total": total}


def _numbers(source):
    return {block["id"]: index for index, block in enumerate(source["blocks"], 1)}


def _render_blocks(blocks, numbers, spans=None):
    lines = []
    for block in blocks:
        text = block["text"]
        for span in reversed((spans or {}).get(block["id"], [])):
            text = text[:span["start"]] + f"{MARKER_OPEN}{span['id']}{MARKER_CLOSE}" + text[span["start"]:]
        lines.append(f"{numbers[block['id']]}|{text}")
    return "\n".join(lines)


def _context(blocks, limit, *, tail):
    text, size = [], 0
    for block in (reversed(blocks) if tail else blocks):
        text.append(block["text"])
        size += len(block["text"])
        if size >= limit:
            break
    joined = "\n".join(reversed(text) if tail else text)
    return joined[-limit:] if tail else joined[:limit]


def chapter_groups(source, limit):
    """Group whole chapters up to ``limit`` characters; split only oversized chapters."""
    chapters = []
    for block in source["blocks"]:
        if not chapters or chapters[-1][0]["chapter_id"] != block["chapter_id"]:
            chapters.append([])
        chapters[-1].append(block)
    groups, current, size = [], [], 0
    for chapter in chapters:
        length = sum(len(block["text"]) for block in chapter)
        if length > limit:
            if current:
                groups.append(current)
                current, size = [], 0
            part, part_size = [], 0
            for block in chapter:
                if part and part_size + len(block["text"]) > limit:
                    groups.append(part)
                    part, part_size = [], 0
                part.append(block)
                part_size += len(block["text"])
            if part:
                groups.append(part)
            continue
        if current and size + length > limit:
            groups.append(current)
            current, size = [], 0
        current.extend(chapter)
        size += length
    if current:
        groups.append(current)
    return groups


def _parallel(function, items, workers):
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
        futures = [executor.submit(function, item) for item in items]
        return [future.result() for future in futures]


def _is(value, options):
    """Type-safe membership: unexpected lists or objects are simply not members."""
    return isinstance(value, str) and value in options


def _unwrap(rows):
    """Accept one redundant nesting level: [[[...], [...]]] -> [[...], [...]]."""
    if (isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list) and rows[0]
            and all(isinstance(item, list) for item in rows[0])):
        return rows[0]
    return rows


def _item_row(row, prefix):
    """Normalize a batched answer row's item ID: 3 or "3" means prefix + "3"."""
    if isinstance(row, list) and row and (type(row[0]) is int or (isinstance(row[0], str) and row[0].strip().isdigit())):
        return [f"{prefix}{int(row[0])}", *row[1:]]
    return row


_COMMON = frozenset("""about after also back been from have into just like more only over says said that their
them then they this very what when where which while with would your character book series""".split())


def _words(text):
    return re.findall(r"\w+", (text or "").casefold())


def _cast_line(row):
    return "|".join([row["id"], row["name"], ";".join(row["aliases"]), row["sex"], row["note"]])


# ---------------------------------------------------------------------------
# Pass 1: sequential prep read

def prep_read(source, requests, meter, ledger, *, window_chars=PREP_WINDOW_CHARS, progress=lambda message: None):
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    numbers = _numbers(source)
    windows = chapter_groups(source, window_chars)
    progress(f"Narrator read: preparing the cast in {len(windows)} sequential window(s)")

    def read(ledger, window):
        cast_text, prior_text = render(ledger)
        payload = {"mode": "narrator_prep", "v": PROMPT_VERSION, "cast": cast_text,
                   "text": _render_blocks(window, numbers)}
        if prior_text:
            payload["prior"] = prior_text
        window_numbers = {numbers[block["id"]]: block["id"] for block in window}

        def validate(response, ledger=ledger, window_numbers=window_numbers):
            return apply_delta(ledger, response, blocks, window_numbers)

        try:
            return meter.run(requests, "prep", PREP, payload, validate)
        except OutputTruncatedError:
            if len(window) < 2:
                raise
            # A runaway reply: read the two halves in order instead, as markup does.
            middle = len(window) // 2
            progress(f"Narrator read: prep output was cut off; reading {len(window)} paragraphs in two halves")
            return read(read(ledger, window[:middle]), window[middle:])

    for index, window in enumerate(windows, 1):
        ledger = read(ledger, window)
        seen = sum(1 for row in ledger["characters"] if row["series"] is None or row["seen"])
        progress(f"Narrator read: prep window {index}/{len(windows)} complete; {seen} character(s) in the ledger")
    return recheck_aliases(ledger), len(windows)


def check_merges(source, requests, meter, ledger, *, workers=1, progress=lambda message: None):
    """One batched yes/no check of prep merges; rejected or uncertain merges are split back."""
    applied = [row for row in ledger["merges"] if row["status"] == "applied"]
    # A merge into an entry that was itself merged later (A into B, then B into C)
    # is covered by the later merge's check; only merges whose result still
    # exists as an entry are checked.
    superseded = [row for row in applied if character(ledger, row["into"]) is None
                  or not any(item["id"] == row["from"] for item in character(ledger, row["into"])["merged"])]
    applied = [row for row in applied if row not in superseded]
    if not applied:
        return ledger, [{"from": row["from"], "into": row["into"], "verdict": "superseded"} for row in superseded]
    numbers = _numbers(source)
    positions = {block["id"]: index for index, block in enumerate(source["blocks"])}
    items = []
    for index, row in enumerate(applied, 1):
        target = character(ledger, row["into"])
        snapshot = next(item for item in target["merged"] if item["id"] == row["from"])
        target_view = dict(target, aliases=[alias for alias in target["aliases"]
                                            if alias not in {snapshot["name"], *snapshot["aliases"]}])
        wanted = {row["evidence"]["block_id"], *(value for value in (snapshot["first"], target["first"]) if value)}
        selected = set()
        for block_id in wanted:
            position = positions[block_id]
            selected.update(range(max(0, position - 1), min(len(source["blocks"]), position + 2)))
        context = [source["blocks"][position] for position in sorted(selected)]
        items.append({"id": f"m{index}", "a": _cast_line(snapshot), "b": _cast_line(target_view),
                      "evidence": [numbers[row["evidence"]["block_id"]], row["evidence"]["quote"]],
                      "text": _render_blocks(context, numbers), "_merge": row})
    progress(f"Narrator read: checking {len(items)} merge(s) found during prep")
    verdicts = {}
    batches = [items[start:start + VERIFY_BATCH] for start in range(0, len(items), VERIFY_BATCH)]

    def run(batch):
        payload = {"mode": "narrator_merge_check", "v": PROMPT_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        expected = {item["id"] for item in batch}

        def validate(response):
            rows = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"r":[[item_id,"y"|"n"|"?"]]}')
            result = {}
            for row in rows:
                row = _item_row(row, "m")
                if isinstance(row, list) and len(row) >= 2 and _is(row[0], expected) and _is(row[1], {"y", "n", "?"}):
                    result[row[0]] = row[1]
            missing = sorted(expected - set(result))
            if missing:
                raise ValueError(f"missing or invalid verdicts for items: {', '.join(missing[:20])}")
            return result

        return meter.run(requests, "identity_check", MERGE_CHECK, payload, validate)

    for result in _parallel(run, batches, workers):
        verdicts.update(result)
    outcomes = []
    for item in items:
        row, verdict = item["_merge"], verdicts[item["id"]]
        outcomes.append({"from": row["from"], "into": row["into"], "verdict": verdict})
        if verdict != "y":
            ledger = undo_merge(ledger, row["from"], row["into"])
    outcomes += [{"from": row["from"], "into": row["into"], "verdict": "superseded"} for row in superseded]
    return ledger, outcomes


def alias_conflicts(source, ledger, *, limit=2):
    """Aliases that may name a different person: a separate capitalized name (not a
    part or shortening of the main name) that appears in a paragraph together with
    the main name. Returns [(entry_id, alias, blocks)]."""
    found = []
    for row in ledger["characters"]:
        if row["series"] is not None and not row["seen"]:
            continue
        main_words = _norm(row["name"]).split()
        if not main_words:
            continue
        for alias in row["aliases"]:
            words = alias.split()
            if not words or not all(word[:1].isupper() for word in words):
                continue
            key = _norm(alias)
            if not key or any(word.startswith(part) or part.startswith(word)
                              for word in main_words for part in key.split()):
                continue
            name_pattern = re.compile(r"\b" + re.escape(row["name"].split()[0]) + r"\b")
            alias_pattern = re.compile(r"\b" + re.escape(alias) + r"\b")
            blocks = [block for block in source["blocks"]
                      if name_pattern.search(block["text"]) and alias_pattern.search(block["text"])][:limit]
            if blocks:
                found.append((row["id"], alias, blocks))
    return found


def check_aliases(source, requests, meter, ledger, *, workers=1, progress=lambda message: None):
    """Batched same-person check for suspicious aliases; split an alias off on "n"."""
    conflicts = alias_conflicts(source, ledger)
    if not conflicts:
        return ledger, []
    numbers = _numbers(source)
    items = []
    for index, (entry_id, alias, blocks) in enumerate(conflicts, 1):
        row = character(ledger, entry_id)
        items.append({"id": f"m{index}", "a": f"{alias}|(given as an alias)", "b": _cast_line(row),
                      "text": _render_blocks(blocks, numbers), "_conflict": (entry_id, alias)})
    progress(f"Narrator read: checking {len(items)} alias(es) that may name a different person")
    batches = [items[start:start + VERIFY_BATCH] for start in range(0, len(items), VERIFY_BATCH)]

    def run(batch):
        payload = {"mode": "narrator_alias_check", "v": PROMPT_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        expected = {item["id"] for item in batch}

        def validate(response):
            rows = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"r":[[item_id,"y"|"n"|"?"]]}')
            result = {}
            for row in rows:
                row = _item_row(row, "m")
                if isinstance(row, list) and len(row) >= 2 and _is(row[0], expected) and _is(row[1], {"y", "n", "?"}):
                    result[row[0]] = row[1]
            missing = sorted(expected - set(result))
            if missing:
                raise ValueError(f"missing or invalid verdicts for items: {', '.join(missing[:20])}")
            return result

        return meter.run(requests, "identity_check", MERGE_CHECK, payload, validate)

    verdicts = {}
    for result in _parallel(run, batches, workers):
        verdicts.update(result)
    ledger = deepcopy(ledger)
    outcomes = []
    for item in items:
        entry_id, alias = item["_conflict"]
        verdict = verdicts[item["id"]]
        outcomes.append({"entry": entry_id, "alias": alias, "verdict": verdict})
        if verdict != "n":
            continue
        row = character(ledger, entry_id)
        row["aliases"] = [value for value in row["aliases"] if value != alias]
        prior = [other for other in ledger["characters"] if other["series"] is not None and not other["seen"]
                 and _norm(alias) in {_norm(other["name"]), *map(_norm, other["aliases"])}]
        if len(prior) == 1:
            prior[0]["seen"] = True        # provisional; the series check confirms it
        else:
            ledger["characters"].append({"id": str(ledger["next"]), "name": alias, "aliases": [], "sex": "?",
                                         "note": f"Distinct from {row['name']}.", "first": None,
                                         "voice": [], "series": None, "series_evidence": [], "seen": True,
                                         "merged": [], "changes": []})
            ledger["next"] += 1
    return ledger, outcomes


def duplicate_candidates(ledger):
    """Pairs of active entries whose names overlap (shared name, shared first
    name, or one name contained in the other): a name only nominates a check.

    Returns (from_id, into_id) with ``into`` preferring the prior-book entry, then
    the earlier entry. Two different prior-book characters are never paired.
    """
    active = [row for row in ledger["characters"] if row["series"] is None or row["seen"]]
    pairs = []
    for left_index, left in enumerate(active):
        for right in active[left_index + 1:]:
            if left["series"] and right["series"] and left["series"] != right["series"]:
                continue
            left_names = {_norm(name) for name in [left["name"], *left["aliases"]]} - {""}
            right_names = {_norm(name) for name in [right["name"], *right["aliases"]]} - {""}
            first_word = ({name.split()[0] for name in left_names if " " in name} & right_names
                          or {name.split()[0] for name in right_names if " " in name} & left_names)
            # One name's words all inside the other's ("Andurs" / "Captain Andurs").
            contained = any(set(a.split()) < set(b.split()) or set(b.split()) < set(a.split())
                            for a in left_names for b in right_names)
            if not (left_names & right_names or first_word or contained):
                continue
            into, source = ((left, right) if left["series"] or not right["series"] and int(left["id"]) < int(right["id"])
                            else (right, left))
            pairs.append((source["id"], into["id"]))
    return pairs


def check_duplicates(source, requests, meter, ledger, labels, extras, spans=(), *, workers=1,
                     progress=lambda message: None):
    """Batched same-person check for name-overlapping entries; merge only on "y"."""
    pairs = duplicate_candidates(ledger)
    if not pairs:
        return ledger, labels, extras, []
    numbers = _numbers(source)
    by_id = {block["id"]: block for block in source["blocks"]}
    spoken = {}
    for span in spans:
        who = labels.get(span["id"], {}).get("who", ("",))
        if who[0] == "char":
            spoken.setdefault(who[1], [])
            if span["block_id"] not in spoken[who[1]]:
                spoken[who[1]].append(span["block_id"])
    items = []
    for index, (from_id, into_id) in enumerate(pairs, 1):
        context = []
        for identifier in (from_id, into_id):
            row = character(ledger, identifier)
            names = [name for name in [row["name"], *row["aliases"]] if name]
            pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b")
            hits = [block for block in source["blocks"] if pattern.search(block["text"])][:2]
            hits += [by_id[block_id] for block_id in spoken.get(identifier, [])[:2]]
            context.extend(block for block in hits if block not in context)
        # Paragraphs naming both entries: two names in one scene usually mean two people.
        first_names = []
        for identifier in (from_id, into_id):
            row = character(ledger, identifier)
            first_names.append(re.compile(r"\b" + re.escape(row["name"]) + r"\b"))
        together = [block for block in source["blocks"]
                    if all(pattern.search(block["text"]) for pattern in first_names)][:2]
        context.extend(block for block in together if block not in context)
        context.sort(key=lambda block: numbers[block["id"]])
        items.append({"id": f"m{index}", "a": _cast_line(character(ledger, from_id)),
                      "b": _cast_line(character(ledger, into_id)), "text": _render_blocks(context, numbers),
                      "_pair": (from_id, into_id)})
    progress(f"Narrator read: checking {len(items)} possible duplicate character entr(y/ies)")
    batches = [items[start:start + VERIFY_BATCH] for start in range(0, len(items), VERIFY_BATCH)]

    def run(batch):
        payload = {"mode": "narrator_duplicate_check", "v": PROMPT_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        expected = {item["id"] for item in batch}

        def validate(response):
            rows = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"r":[[item_id,"y"|"n"|"?"]]}')
            result = {}
            for row in rows:
                row = _item_row(row, "m")
                if isinstance(row, list) and len(row) >= 2 and _is(row[0], expected) and _is(row[1], {"y", "n", "?"}):
                    result[row[0]] = row[1]
            missing = sorted(expected - set(result))
            if missing:
                raise ValueError(f"missing or invalid verdicts for items: {', '.join(missing[:20])}")
            return result

        return meter.run(requests, "identity_check", MERGE_CHECK, payload, validate)

    verdicts = {}
    for result in _parallel(run, batches, workers):
        verdicts.update(result)
    ledger, labels, extras = deepcopy(ledger), deepcopy(labels), deepcopy(extras)
    outcomes, renamed = [], {}
    for item in items:
        from_id, into_id = item["_pair"]
        from_id, into_id = renamed.get(from_id, from_id), renamed.get(into_id, into_id)
        verdict = verdicts[item["id"]]
        outcomes.append({"from": from_id, "into": into_id, "verdict": verdict})
        if verdict == "n" and character(ledger, from_id) is not None and character(ledger, into_id) is not None:
            # Two people: drop any alias that is the other person's own name.
            left, right = character(ledger, from_id), character(ledger, into_id)
            for owner, other in ((left, right), (right, left)):
                owner["aliases"] = [alias for alias in owner["aliases"] if _norm(alias) != _norm(other["name"])]
        if verdict != "y" or from_id == into_id or character(ledger, from_id) is None or character(ledger, into_id) is None:
            continue
        merge(ledger, from_id, into_id, None)
        ledger["merges"][-1]["status"] = "applied_after_duplicate_check"
        renamed = {key: (into_id if value == from_id else value) for key, value in renamed.items()}
        renamed[from_id] = into_id
    if renamed:
        def remap(who):
            if who[0] == "char":
                return ("char", renamed.get(who[1], who[1]))
            if who[0] in {"unresolved", "multi"}:
                values = list(dict.fromkeys(renamed.get(value, value) for value in who[1]))
                if who[0] == "multi" and len(values) < 2:
                    return ("char", values[0])
                return (who[0], values)
            return who
        for item in [*labels.values(), *extras]:
            item["who"] = remap(item["who"])
    return ledger, labels, extras, outcomes


def check_series(source, requests, meter, ledger, labels=None, spans=(), *, workers=1,
                 progress=lambda message: None):
    """Confirm every returning series character seen in this book.

    One batched check per 25 characters replaces the consensus jury: "y" needs an
    exact quote beyond the bare name and authorizes voice reuse; "n" makes the
    entry a new character of this book (and an advisory issue); "?" leaves the
    series link to the conservative name rule and human review.
    """
    if not any(row["series"] is not None and row["seen"] for row in ledger["characters"]):
        return ledger, []
    ledger = deepcopy(ledger)
    targets = [row for row in ledger["characters"] if row["series"] is not None and row["seen"]]
    numbers = _numbers(source)
    by_id = {block["id"]: block for block in source["blocks"]}
    spoken = {}
    for span in spans:
        who = (labels or {}).get(span["id"], {}).get("who", ("",))
        for identifier in ([who[1]] if who[0] == "char" else who[1] if who[0] == "multi" else []):
            spoken.setdefault(identifier, [])
            if span["block_id"] not in spoken[identifier]:
                spoken[identifier].append(span["block_id"])
    def evidence_blocks(row, *, mentions, lines):
        """Paragraphs naming the character and lines it speaks, those that share the
        most words with the earlier description first (then in book order)."""
        names = [name for name in [row["name"], *row["aliases"]] if name]
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b")
        name_words = {word for name in names for word in _words(name)}
        described = {word for word in _words(row.get("prior_note", row["note"]))
                     if len(word) > 3 and word not in name_words and word not in _COMMON}

        def ranked(blocks):
            return sorted(blocks, key=lambda block: (-len(described & set(_words(block["text"]))),
                                                     numbers[block["id"]]))

        hits = ranked([block for block in source["blocks"] if pattern.search(block["text"])])[:mentions]
        hits += [block for block in ranked([by_id[block_id] for block_id in spoken.get(row["id"], [])])
                 if block not in hits][:lines]
        if not hits and row["first"]:
            hits = [by_id[row["first"]]]
        return sorted(hits, key=lambda block: numbers[block["id"]])

    def items_for(rows, *, mentions, lines):
        items = []
        for index, row in enumerate(rows, 1):
            hits = evidence_blocks(row, mentions=mentions, lines=lines)
            prior = "|".join([row["name"], ";".join(row["aliases"]), row.get("prior_note", row["note"])])
            items.append({"id": f"p{index}", "prior": prior, "text": _render_blocks(hits, numbers),
                          "_row": row["id"], "_numbers": {numbers[block["id"]]: block["id"] for block in hits}})
        return items

    def run(batch):
        payload = {"mode": "narrator_series_check", "v": PROMPT_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        expected = {item["id"]: item for item in batch}

        def validate(response):
            rows = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"r":[[item_id,"y",N,quote],[item_id,"n"],[item_id,"?"]]}')
            result = {}
            for row in rows:
                row = _item_row(row, "p")
                if not isinstance(row, list) or len(row) < 2 or not _is(row[0], expected) or not _is(row[1], {"y", "n", "?"}):
                    continue
                verdict, evidence = row[1], None
                if verdict == "y":
                    shown = expected[row[0]]["_numbers"]
                    named = shown.get(row[2]) if len(row) > 3 and type(row[2]) is int else None
                    quote = row[3] if len(row) > 3 else None
                    # The named paragraph first, then any other paragraph shown in the item.
                    for block_id in dict.fromkeys([named, *shown.values()]) if quote is not None else ():
                        exact = exact_quote(quote, by_id[block_id]["text"]) if block_id else None
                        if exact:
                            evidence = {"block_id": block_id, "quote": exact}
                            break
                    else:
                        verdict = "?"
                result[row[0]] = (verdict, evidence)
            missing = sorted(set(expected) - set(result))
            if missing:
                raise ValueError(f"missing or invalid verdicts for items: {', '.join(missing[:20])}")
            return result

        return meter.run(requests, "identity_check", SERIES_CHECK, payload, validate)

    def ask(rows, *, mentions, lines):
        items = items_for(rows, mentions=mentions, lines=lines)
        batches = [items[start:start + VERIFY_BATCH] for start in range(0, len(items), VERIFY_BATCH)]
        verdicts = {}
        for result in _parallel(run, batches, workers):
            verdicts.update(result)
        answers = {}
        for item in items:
            verdict, evidence = verdicts[item["id"]]
            row = character(ledger, item["_row"])
            if verdict == "y" and evidence is not None:
                if evidence not in row["series_evidence"]:
                    row["series_evidence"].insert(0, evidence)
                verdict = "y" if contextual_series_evidence(row) else "?"
            answers[row["id"]] = verdict
        return answers

    progress(f"Narrator read: confirming {len(targets)} returning series character(s)")
    answers = ask(targets, mentions=2, lines=2)
    unsure = [row for row in targets if answers[row["id"]] == "?"]
    if unsure:
        # One wider look: more of the paragraphs that best match the earlier description.
        progress(f"Narrator read: a wider look at {len(unsure)} unconfirmed returning character(s)")
        answers.update(ask(unsure, mentions=4, lines=4))
    outcomes = []
    for row in targets:
        verdict = answers[row["id"]]
        outcomes.append({"id": row["id"], "series": row["series"], "verdict": verdict})
        if verdict == "y":
            row["series_confirmed"] = True
        elif verdict == "n":
            # A different person who shares the name: a new character in this book.
            row["series"], row["series_evidence"], row["note"] = None, [], ""
            row.pop("prior_note", None)
    return ledger, outcomes


# ---------------------------------------------------------------------------
# Pass 2: parallel markup

def _parse_who(who, ledger, index):
    if type(who) is int:
        who = str(who)
    if not isinstance(who, str) or not who.strip():
        raise ValueError("who must be a cast id, '-', '?id,id' or '+Name'")
    who = who.strip()
    if who == "-":
        return ("narrator",)
    if "&" in who and not who.startswith(("+", "?")):
        # "3&7": these characters say the same words together.
        performers = []
        for token in who.split("&"):
            token = token.strip()
            row = character(ledger, token)
            resolved = row["id"] if row is not None else index.get(_norm(token)) if token else None
            if resolved and resolved not in performers:
                performers.append(resolved)
        if len(performers) >= 2:
            return ("multi", performers)
        if performers:
            return ("char", performers[0])
        raise ValueError(f"unknown speakers in {who[:60]!r}; use cast ids like '3&7'")
    if who.startswith("?") or ("," in who and not who.startswith("+")):
        # "?3,7" (or a bare "3,7") lists candidates; it never picks one.
        candidates = []
        for token in re.split(r"[,;/ ]+", who.lstrip("?")):
            token = token.strip()
            row = character(ledger, token)
            resolved = row["id"] if row is not None else index.get(_norm(token)) if token else None
            if resolved and resolved not in candidates:
                candidates.append(resolved)
        return ("unresolved", candidates)
    if who.startswith("+"):
        name = _clean(who[1:], 80)
        if not re.search(r"[^\W\d_]", name) or name.casefold() == "name":
            # "+?", "+" alone, or the prompt's own placeholder: no name to give a new speaker.
            return ("unresolved", [])
        # Same resolution as an explicit speech tag: exact name, or a first name
        # unique in the cast ("+Barry" with one "Barry Francis").
        existing = tag_character(index, name)
        return ("char", existing) if existing else ("new", name)
    row = character(ledger, who)
    if row is not None:
        return ("char", row["id"])
    resolved = index.get(_norm(who))
    if resolved:
        return ("char", resolved)
    raise ValueError(f"unknown speaker {who[:60]!r}; use a cast id from the cast list, '-', '?id,id' or '+Name'")


def _markup_validator(ledger, index, targets, blocks_by_number, block_spans):
    expected = [span["id"] for span in targets]
    expected_set = set(expected)

    def validate(response):
        if not isinstance(response, dict):
            raise ValueError("response must be an object")
        rows = _unwrap(response.get("q", []))
        if not isinstance(rows, list):
            raise ValueError('"q" must be an array of [qid, who, kind?, delivery?] arrays')
        labels, errors = {}, []
        for row in rows:
            if not isinstance(row, list) or len(row) < 2 or type(row[0]) not in {str, int}:
                errors.append("each q entry must be [qid, who, ...]")
                continue
            # Bare numbers ("16" or 16) unambiguously name quotation q16.
            qid = str(row[0]).strip().strip(MARKER_OPEN + MARKER_CLOSE)
            if qid.isdigit():
                qid = "q" + qid
            if not _is(qid, expected_set):
                errors.append(f"{qid[:20]!r} is not a quotation id in text")
                continue
            if qid in labels:
                errors.append(f"{qid} labeled twice")
                continue
            try:
                who = _parse_who(row[1], ledger, index)
            except ValueError as error:
                errors.append(f"{qid}: {error}")
                continue
            kind = row[2] if len(row) > 2 and _is(row[2], {"t", "l"}) else None
            delivery = _clean(row[3], 40) if len(row) > 3 and isinstance(row[3], str) else ""
            quote = row[4] if len(row) > 4 and isinstance(row[4], str) and 0 < len(row[4]) <= 240 else ""
            labels[qid] = {"who": who, "kind": kind, "delivery": delivery, "quote": quote}
        missing = [qid for qid in expected if qid not in labels]
        if missing:
            errors.append(f"missing labels for {len(missing)} quotation id(s): {', '.join(missing[:30])}")
        if errors:
            raise ValueError("; ".join(errors[:12]))
        extras, dropped = [], 0
        for row in response.get("x", []) if isinstance(response.get("x", []), list) else []:
            try:
                number, start_text, end_text, who = row[0], row[1], row[2], row[3]
                block = blocks_by_number[number]
                text = block["text"]
                if text.count(start_text) != 1 or not end_text:
                    raise ValueError
                start = text.index(start_text)
                end = text.index(end_text, start) + len(end_text)
                if any(span["start"] < end and span["end"] > start for span in block_spans.get(block["id"], [])):
                    raise ValueError
                parsed = _parse_who(who, ledger, index)
                kind = row[4] if len(row) > 4 and _is(row[4], {"t", "l"}) else None
                extras.append({"block_id": block["id"], "start": start, "end": end,
                               "who": parsed, "kind": kind, "delivery": ""})
            except (ValueError, TypeError, KeyError, IndexError):
                dropped += 1
        notes, dropped_notes = [], 0
        raw_notes = _unwrap(response.get("p", []))
        for row in raw_notes if isinstance(raw_notes, list) else []:
            note = _parse_note(row, expected_set, blocks_by_number)
            if note is None:
                dropped_notes += 1
            else:
                notes.append(note)
        return {"labels": labels, "extras": extras, "dropped_extras": dropped, "notes": notes,
                "dropped_notes": dropped_notes}

    return validate


NOTE_REASON = "The source text explicitly supports this performance note."


def _parse_note(row, qids, blocks_by_number):
    """One terse performance note -> {"target", "kind", "raw"} in the inline-proposal shape, or None."""
    if not isinstance(row, list) or len(row) < 3 or not isinstance(row[1], str) or not isinstance(row[2], str):
        return None
    target, op, original = row[0], row[1].strip().lower(), row[2]
    if isinstance(target, str) and target.strip().lower().startswith("q"):
        qid = target.strip().strip(MARKER_OPEN + MARKER_CLOSE)
        if qid not in qids:
            return None
        target = ("q", qid)
    else:
        try:
            number = int(target)
        except (TypeError, ValueError):
            return None
        if number not in blocks_by_number:
            return None
        target = ("n", blocks_by_number[number]["id"])
    if not original:
        return None
    if op == "as":
        # A per-line exception to the voice changes in effect: "normal", or one effect preset.
        value = original.strip().casefold()
        if target[0] != "q" or value not in {"normal", *EFFECT_PRESETS}:
            return None
        return {"target": target, "kind": "voice_as", "raw": {"as": value}}
    if op == "sp":
        if len(row) < 4 or not isinstance(row[3], str) or not row[3] or row[3] == original:
            return None
        note = row[4] if len(row) > 4 and isinstance(row[4], str) and row[4].strip() else "Say the normal spelling naturally."
        return {"target": target, "kind": "edit",
                "raw": {"original": original, "replacement": row[3], "reason": NOTE_REASON, "direction": note[:240]}}
    if op == "tag":
        op = "style"
    if op in DELIVERY_TAGS:
        note = row[3] if len(row) > 3 and isinstance(row[3], str) else ""
        if op == "style" and not note.strip():
            return None
        return {"target": target, "kind": "control",
                "raw": {"kind": "delivery", "tag": op, "original": original, "reason": NOTE_REASON,
                        "direction": note[:240]}}
    if op == "pause":
        if len(row) < 5 or type(row[3]) is not int or not isinstance(row[4], str):
            return None
        return {"target": target, "kind": "control",
                "raw": {"kind": "pause", "duration_ms": row[3], "position": row[4], "original": original,
                        "reason": NOTE_REASON}}
    if op in VOCAL_SOUNDS:
        if len(row) < 4 or not isinstance(row[3], str):
            return None
        return {"target": target, "kind": "control",
                "raw": {"kind": "vocalization", "sound": op, "position": row[3], "original": original,
                        "reason": NOTE_REASON}}
    return None


def markup(source, requests, meter, ledger, spans, *, policy, workers=1, group_chars=MARKUP_CHARS,
           progress=lambda message: None):
    numbers = _numbers(source)
    block_spans = spans_by_block(spans)
    groups = chapter_groups(source, group_chars)
    # The full cast, including prior-book characters not yet seen: a returning
    # character can be labeled by ID; the series check still confirms identity.
    index = name_index(ledger, include_unseen=True)
    cast_text, prior_text = render(ledger, include_questions=False)
    cast_text = "\n".join(text for text in (cast_text, prior_text) if text)
    compact_policy = {key: value for key, value in policy.items() if key in {"thoughts", "letters", "nested_quotations"}}
    positions = {block["id"]: position for position, block in enumerate(source["blocks"])}
    progress(f"Narrator read: labeling {len(spans)} quotation(s) in {len(groups)} chapter group(s)")
    done = [0]
    lock = threading.Lock()

    empty = {"labels": {}, "extras": [], "dropped_extras": 0, "notes": [], "dropped_notes": 0}

    def request(group):
        targets = [span for block in group for span in block_spans.get(block["id"], [])]
        if not targets:
            return dict(empty)
        first, last = positions[group[0]["id"]], positions[group[-1]["id"]]
        payload = {"mode": "narrator_markup", "v": PROMPT_VERSION, "cast": cast_text, "policy": compact_policy,
                   "before": _context(source["blocks"][max(0, first - 20):first], CONTEXT_BEFORE, tail=True),
                   "text": _render_blocks(group, numbers, block_spans),
                   "after": _context(source["blocks"][last + 1:last + 21], CONTEXT_AFTER, tail=False)}
        blocks_by_number = {numbers[block["id"]]: block for block in group}
        validate = _markup_validator(ledger, index, targets, blocks_by_number, block_spans)
        try:
            return meter.run(requests, "markup", MARKUP, payload, validate)
        except OutputTruncatedError:
            # An explicit output-capacity signal: split at a paragraph boundary
            # and label each half with its own neighbouring context.
            if len(group) < 2:
                raise
            with lock:
                subdivided[0] += 1
            middle = len(group) // 2
            halves = [request(group[:middle]), request(group[middle:])]
            return {"labels": {**halves[0]["labels"], **halves[1]["labels"]},
                    "extras": halves[0]["extras"] + halves[1]["extras"],
                    "dropped_extras": halves[0]["dropped_extras"] + halves[1]["dropped_extras"],
                    "notes": halves[0]["notes"] + halves[1]["notes"],
                    "dropped_notes": halves[0]["dropped_notes"] + halves[1]["dropped_notes"]}

    subdivided = [0]

    def run(group):
        result = request(group)
        with lock:
            done[0] += 1
            progress(f"Narrator read: markup {done[0]}/{len(groups)} complete")
        return result

    labels, extras, notes, dropped = {}, [], [], {"extras": 0, "notes": 0}
    for result in _parallel(run, groups, workers):
        labels.update(result["labels"])
        extras.extend(result["extras"])
        notes.extend(result["notes"])
        dropped["extras"] += result["dropped_extras"]
        dropped["notes"] += result["dropped_notes"]
    dropped["subdivided_groups"] = subdivided[0]
    return labels, extras, notes, dropped, len(groups)


def add_new_speakers(ledger, labels, extras):
    """Turn markup '+Name' speakers into ledger entries (deduplicated by name).

    Prior-book characters labeled during markup become seen (without evidence:
    the series check must still confirm them).
    """
    ledger = deepcopy(ledger)
    created = {}
    for item in [*labels.values(), *extras]:
        who = item["who"]
        for identifier in ([who[1]] if who[0] == "char" else who[1] if who[0] in {"unresolved", "multi"} else []):
            row = character(ledger, identifier)
            if row is not None and who[0] in {"char", "multi"}:
                row["seen"] = True
        if who[0] != "new":
            continue
        key = _norm(item["who"][1])
        if key not in created:
            identifier = str(ledger["next"])
            ledger["next"] += 1
            ledger["characters"].append({"id": identifier, "name": item["who"][1], "aliases": [], "sex": "?",
                                         "note": "Minor speaker added during markup.", "first": None,
                                         "voice": [], "series": None, "series_evidence": [], "seen": True,
                                         "merged": [], "changes": []})
            created[key] = identifier
        item["who"] = ("char", created[key])
    return ledger, len(created)


# ---------------------------------------------------------------------------
# Pass 3: targeted verification

SCENE_RADIUS = 30
SELF_CHECK_REASONS = frozenset({"absent_speaker", "self_address"})


def _proper_tokens(names):
    """Capitalized name words (3+ letters) that identify a character in running text."""
    return {word for name in names for word in re.findall(r"\b[A-Z][\w'-]{2,}", name)
            if word.split("'")[0] not in _NOT_NAMES}


def scene_checks(source, ledger, spans, labels, *, radius=SCENE_RADIUS):
    """Self-checks for errors that leave no local tag disagreement.

    * absent_speaker: a character is credited with a line although no name of
      theirs appears within ``radius`` paragraphs of the same chapter, while the
      text there names other speaking characters (the likely alternatives). For a
      generic speaker ("spaceport official") its name words count as its names.
    * self_address: a line calls out its own speaker's one-word name ("Killa,
      wait"; "..., Killashandra?"). Full-name introductions are not flagged.
    Returns {span_id: (reason, alternative character ids)}.
    """
    blocks = source["blocks"]
    position = {block["id"]: index for index, block in enumerate(blocks)}
    words = [set(re.findall(r"[\w'-]+", block["text"])) for block in blocks]
    words = [found | {word.casefold() for word in found} for found in words]
    speaking = {who[1] for who in (item["who"] for item in labels.values()) if who[0] == "char"}
    tokens, generic = {}, set()
    for row in ledger["characters"]:
        if row["id"] in speaking:
            tokens[row["id"]] = _proper_tokens([row["name"], *row["aliases"]])
            if not tokens[row["id"]]:
                # A generic speaker: its own descriptive words, matched in any case.
                tokens[row["id"]] = {word.casefold() for word in re.findall(r"[^\W\d_]{4,}", row["name"])}
                generic.add(row["id"])
    flagged = {}
    for span in spans:
        who = labels[span["id"]]["who"]
        if who[0] != "char" or not tokens.get(who[1]):
            continue
        label, own = who[1], tokens[who[1]]
        inner = span["text"].strip("\"“” ")
        one_word = {word for word in own if " " not in word} if label not in generic else set()
        vocative = any(re.match(rf"{re.escape(word)},\s+\S+\s+\S+", inner)
                       or re.search(rf",\s+{re.escape(word)}[.!?…]*$", inner) for word in one_word)
        index = position[span["block_id"]]
        chapter = blocks[index]["chapter_id"]
        nearby = set()
        for other in range(max(0, index - radius), min(len(blocks), index + radius + 1)):
            if blocks[other]["chapter_id"] == chapter:
                nearby |= words[other]
        named = [identifier for identifier, values in tokens.items()
                 if identifier != label and identifier not in generic and values & nearby]
        if vocative:
            flagged[span["id"]] = ("self_address", named[:4])
        elif not own & nearby and named:
            flagged[span["id"]] = ("absent_speaker", named[:4])
    return flagged


def cast_audit(cast):
    """Local cast warnings: a name that reads like a description or a copied
    field, and a name or alias shared by two characters. Advisory only."""
    warnings = []
    owners = {}
    for person in cast["characters"]:
        name = person["name"]
        if len(name.split()) > 6 or re.search(r"[|;/,]|\d", name) or name.casefold() in {"m", "f", "n", "?"}:
            warnings.append({"character_id": person["id"], "kind": "name_shape",
                             "reason": f"The character name {name[:80]!r} looks like a description or a copied field."})
        for value in {_norm(item) for item in [name, *person.get("aliases", [])]} - {""}:
            owners.setdefault(value, []).append(person)
    for value, people in owners.items():
        if len({person["id"] for person in people}) > 1:
            names = ", ".join(sorted({person["name"] for person in people}))
            warnings.append({"character_id": people[0]["id"], "kind": "shared_name",
                             "reason": f"{names} share the name or alias {value!r}; check that they are different people."})
    return warnings


def self_check_report(verify_stats, duplicate_outcomes, warnings, unresolved):
    """Numbers summarising what the narrator read checked and changed."""
    reasons, changed = Counter(verify_stats.get("reasons", {})), Counter(verify_stats.get("changed_by_reason", {}))
    followup = verify_stats.get("scene_followup") or {}
    reasons.update(followup.get("reasons", {}))
    changed.update(followup.get("changed_by_reason", {}))
    return {"lines_rechecked": dict(reasons), "lines_changed": dict(changed),
            "duplicate_checks": dict(Counter(outcome["verdict"] for outcome in duplicate_outcomes)),
            "cast_warnings": dict(Counter(warning["kind"] for warning in warnings)),
            "unresolved_lines": unresolved}


def verification_items(source, ledger, spans, labels):
    """Lines worth a second look: unresolved; disagreeing with an explicit tag or an
    adjacent action beat; in a paragraph whose quotations got different speakers;
    or flagged by the scene self-checks (``scene_checks``)."""
    index = name_index(ledger)
    scene = scene_checks(source, ledger, spans, labels)
    by_block = {}
    for span in spans:
        by_block.setdefault(span["block_id"], []).append(span)
    mixed = {}
    for block_id, group in by_block.items():
        speakers = [labels[span["id"]]["who"][1] for span in group if labels[span["id"]]["who"][0] == "char"]
        if len(set(speakers)) > 1:
            for span in group:
                mixed[span["id"]] = list(dict.fromkeys(speakers))
    items = []
    for span in spans:
        who = labels[span["id"]]["who"]
        if who[0] == "multi":
            continue
        label = who[1] if who[0] == "char" else None
        tagged = tag_character(index, span["tag"]["name"]) if span["tag"] and span["tag"]["name"] else None
        beat = tag_character(index, span["beat"]) if span.get("beat") else None
        if who[0] == "unresolved":
            options = list(dict.fromkeys([*who[1], *(value for value in (tagged, beat) if value)]))
            items.append({"span": span, "reason": "unresolved", "options": options})
        elif tagged and label != tagged:
            items.append({"span": span, "reason": "tag_disagreement",
                          "options": list(dict.fromkeys([tagged, *([label] if label else [])]))})
        elif beat and label and label != beat:
            items.append({"span": span, "reason": "beat_disagreement", "options": [label, beat]})
        elif span["id"] in mixed and label:
            options = list(dict.fromkeys([label, *mixed[span["id"]], *([beat] if beat else [])]))
            items.append({"span": span, "reason": "mixed_paragraph", "options": options})
        elif span["id"] in scene and label:
            reason, alternatives = scene[span["id"]]
            items.append({"span": span, "reason": reason, "options": list(dict.fromkeys([label, *alternatives]))})
    return items


def scene_followups(source, spans, labels, corrections, checked, *, radius=15):
    """After a correction X -> Y, re-check other X lines in the same scene (same chapter,
    within ``radius`` paragraphs): one confusion usually affects a whole exchange."""
    positions = {block["id"]: position for position, block in enumerate(source["blocks"])}
    chapter_of = {block["id"]: block["chapter_id"] for block in source["blocks"]}
    items, queued = [], set()
    for span_id, before, after in corrections:
        anchor = next(span for span in spans if span["id"] == span_id)
        center = positions[anchor["block_id"]]
        for span in spans:
            if span["id"] in checked or span["id"] in queued:
                continue
            if (chapter_of[span["block_id"]] == chapter_of[anchor["block_id"]]
                    and abs(positions[span["block_id"]] - center) <= radius
                    and labels[span["id"]]["who"] == ("char", before)):
                items.append({"span": span, "reason": "scene_followup", "options": [before, after]})
                queued.add(span["id"])
    return items


def verify(source, requests, meter, ledger, spans, labels, *, workers=1, progress=lambda message: None):
    items = verification_items(source, ledger, spans, labels)
    labels, stats = _verify_items(source, requests, meter, ledger, spans, labels, items,
                                  workers=workers, progress=progress)
    corrections = stats.pop("corrections")
    # Lines the verifier was unsure about stay eligible for the scene follow-up.
    decided = {item["span"]["id"] for item in items if labels[item["span"]["id"]].get("decided")}
    followups = scene_followups(source, spans, labels, corrections, decided)
    if followups:
        labels, second = _verify_items(source, requests, meter, ledger, spans, labels, followups,
                                       workers=workers, progress=progress)
        second.pop("corrections")
        stats["scene_followup"] = second
    return labels, stats


def _verify_items(source, requests, meter, ledger, spans, labels, items, *, workers=1, progress=lambda message: None):
    if not items:
        return labels, {"items": 0, "resolved": 0, "changed": 0, "unresolved": 0, "tie_breaks": 0, "corrections": []}
    numbers = _numbers(source)
    positions = {block["id"]: position for position, block in enumerate(source["blocks"])}
    block_spans = spans_by_block(spans)
    progress(f"Narrator read: verifying {len(items)} uncertain or tag-disagreeing label(s)")
    chapter_of = {block["id"]: block["chapter_id"] for block in source["blocks"]}
    chapter_speakers = {}
    for span in spans:
        who = labels[span["id"]]["who"]
        if who[0] == "char":
            chapter_speakers.setdefault(chapter_of[span["block_id"]], Counter())[who[1]] += 1
    payload_items = []
    for number, item in enumerate(items, 1):
        span = item["span"]
        position = positions[span["block_id"]]
        span_chapter = chapter_of[span["block_id"]]
        context = [block for block in source["blocks"][max(0, position - 2):position + 3]
                   if block["chapter_id"] == span_chapter]
        local_spans = {block["id"]: block_spans.get(block["id"], []) for block in context}
        if not item["options"]:
            # No candidates offered: the chapter's most frequent speakers.
            item["options"] = [identifier for identifier, _ in
                               chapter_speakers.get(span_chapter, Counter()).most_common(8)]
        options = item["options"]
        entry = {"id": f"i{number}", "target": span["id"],
                 "text": _render_blocks(context, numbers, local_spans),
                 "options": options,
                 "cast": "\n".join(_cast_line(character(ledger, identifier)) for identifier in options)}
        named = [value for value in ((span["tag"] or {}).get("name"), span.get("beat")) if value]
        if not named:
            # One paragraph, one speaker: an untagged quotation in a paragraph whose
            # other quotations name a single speaker is shown that name.
            beside = {value for other in block_spans.get(span["block_id"], []) if other is not span
                      for value in ((other["tag"] or {}).get("name"), other.get("beat")) if value}
            if len(beside) == 1:
                named = list(beside)
        if named:
            # The name the text itself places beside the line (speech tag or action beat).
            entry["named_beside"] = list(dict.fromkeys(named))
        payload_items.append(entry)
        item["item_id"] = f"i{number}"
    batches = [payload_items[start:start + VERIFY_BATCH] for start in range(0, len(payload_items), VERIFY_BATCH)]
    known = {row["id"] for row in ledger["characters"]}

    def outcome(item, before, verdict):
        if verdict == "-":
            return ("narrator",), True
        if verdict in item["options"]:
            return ("char", verdict), True
        if item["reason"] in SELF_CHECK_REASONS and before[0] == "char":
            # A self-check only nominates a doubt; an unsure answer keeps the
            # markup label, which was made with the whole chapter in view.
            return before, False
        return ("unresolved", item["options"]), False

    def run(batch):
        payload = {"mode": "narrator_verify", "v": PROMPT_VERSION, "items": batch}
        expected = {item["id"] for item in batch}

        def validate(response):
            rows = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"r":[[item_id,who]]}')
            result = {}
            for row in rows:
                row = _item_row(row, "i")
                if isinstance(row, list) and len(row) >= 2 and _is(row[0], expected) and type(row[1]) in {str, int}:
                    value = str(row[1]).strip()
                    result[row[0]] = (value if value in {"-", "?"} else
                                      ledger_id(value) if ledger_id(value) in known else "?")
            missing = sorted(expected - set(result))
            if missing:
                raise ValueError(f"missing verdicts for items: {', '.join(missing[:20])}")
            return result

        return meter.run(requests, "verify", VERIFY, payload, validate)

    verdicts = {}
    for result in _parallel(run, batches, workers):
        verdicts.update(result)

    entries = {entry["id"]: entry for entry in payload_items}
    tie_items, tie_outcomes = [], {}
    for item in items:
        before = labels[item["span"]["id"]]["who"]
        proposed, _ = outcome(item, before, verdicts[item["item_id"]])
        if before[0] not in {"char", "narrator"} or proposed[0] not in {"char", "narrator"} or proposed == before:
            continue
        entry = entries[item["item_id"]]
        outcomes = {"a": before, "b": proposed}
        tie_outcomes[item["item_id"]] = outcomes
        tie_item = {"id": item["item_id"], "target": entry["target"], "text": entry["text"],
                    "choices": [{"id": option, "who": "-" if who[0] == "narrator" else who[1]}
                                for option, who in outcomes.items()],
                    "cast": "\n".join(_cast_line(character(ledger, identifier)) for identifier in
                                      dict.fromkeys(who[1] for who in outcomes.values() if who[0] == "char"))}
        if "named_beside" in entry:
            tie_item["named_beside"] = entry["named_beside"]
        tie_items.append(tie_item)

    def choose(batch):
        payload = {"mode": "narrator_verify_tiebreak", "v": PROMPT_VERSION, "items": batch}
        expected = {item["id"]: {choice["id"] for choice in item["choices"]} for item in batch}

        def validate(response):
            rows = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"r":[[item_id,choice_id]]}')
            result = {}
            for row in rows:
                row = _item_row(row, "i")
                if not (isinstance(row, list) and len(row) == 2 and _is(row[0], expected)
                        and isinstance(row[1], str) and row[1] in expected[row[0]]):
                    raise ValueError("tie-break rows must contain an expected item id and one of its two choice ids")
                if row[0] in result:
                    raise ValueError(f"duplicate tie-break choice for item: {row[0]}")
                result[row[0]] = row[1]
            missing = sorted(expected.keys() - set(result))
            if missing:
                raise ValueError(f"missing tie-break choices for items: {', '.join(missing[:20])}")
            return result

        return meter.run(requests, "verify", VERIFY_TIEBREAK, payload, validate)

    selections = {}
    tie_batches = [tie_items[start:start + VERIFY_BATCH] for start in range(0, len(tie_items), VERIFY_BATCH)]
    for result in _parallel(choose, tie_batches, workers):
        for item_id, choice in result.items():
            selections[item_id] = tie_outcomes[item_id][choice]

    labels = deepcopy(labels)
    stats = {"items": len(items), "resolved": 0, "changed": 0, "unresolved": 0, "kept": 0,
             "tie_breaks": len(tie_items), "reasons": dict(Counter(item["reason"] for item in items)),
             "changed_by_reason": {}, "corrections": []}
    for item in items:
        span, verdict = item["span"], verdicts[item["item_id"]]
        label = labels[span["id"]]
        before = label["who"]
        after, decided = outcome(item, before, verdict)
        if item["item_id"] in selections:
            after, decided = selections[item["item_id"]], True
        if after != before:
            stats["changed_by_reason"][item["reason"]] = stats["changed_by_reason"].get(item["reason"], 0) + 1
        label["who"] = after
        label["verified"] = item["reason"]
        label["decided"] = decided
        if after[0] == "unresolved":
            stats["unresolved"] += 1
        elif before[0] == "unresolved":
            stats["resolved"] += 1
        elif after != before:
            stats["changed"] += 1
            if before[0] == "char" and after[0] == "char":
                stats["corrections"].append((span["id"], before[1], after[1]))
        else:
            stats["kept"] += 1
    return labels, stats


# ---------------------------------------------------------------------------
# Cast, voices, and the analysis document

def _speaking_ids(labels, extras):
    ids = Counter()
    for item in [*labels.values(), *extras]:
        who = item["who"]
        if who[0] == "char":
            ids[who[1]] += 1
        elif who[0] == "multi":
            for identifier in who[1]:
                ids[identifier] += 1
        elif who[0] == "unresolved":
            for identifier in who[1]:
                ids[identifier] += 0
    return ids


def build_cast(source, ledger, speaking):
    from .pipeline import seed_cast
    cast = seed_cast()
    narration = ledger["narration"]
    include = [row for row in ledger["characters"] if row["id"] in speaking
               or (narration["mode"] == "first" and narration["narrator"] == row["id"])]
    for row in include:
        evidence = [dict(item) for item in row["voice"]][:4]
        record = {"id": f"char_{row['id']}", "name": row["name"], "aliases": list(row["aliases"]),
                  "description": row["note"] or "Character in the book.", "evidence": evidence,
                  "voice_id": f"voice_{row['id']}"}
        if row["series"] is not None:
            contextual = contextual_series_evidence(row)
            record["series_identity"] = (
                {"character_id": row["series"], "basis": "context", "review_required": False,
                 "reason": "Recognized as a returning character during the narrator read.", "evidence": contextual[:2]}
                if contextual else
                {"character_id": row["series"], "basis": "uncertain", "review_required": True,
                 "reason": "Possibly a returning character; no identifying quote was recorded.", "evidence": []})
        cast["characters"].append(record)
        cast["roles"].append({"id": f"role_{row['id']}", "kind": "dialogue", "character_id": record["id"],
                              "voice_id": record["voice_id"], "description": f"Dialogue role for {row['name']}."})
        cast["voices"].append({"id": record["voice_id"], "description": "Voice pending design.", "traits": []})
    if narration["mode"] == "first" and narration["narrator"] and any(row["id"] == narration["narrator"] for row in include):
        role = cast["roles"][0]
        role["character_id"] = f"char_{narration['narrator']}"
        role["voice_id"] = f"voice_{narration['narrator']}"
        role["description"] = "First-person narration by the narrating character."
        cast["voices"] = [voice for voice in cast["voices"] if voice["id"] != "voice_narrator_main"]
    return cast


def _partners(source, labels, extras, spans):
    by_chapter = {}
    chapter_of = {block["id"]: block["chapter_id"] for block in source["blocks"]}
    span_block = {span["id"]: span["block_id"] for span in spans}
    for qid, label in labels.items():
        if label["who"][0] == "char":
            by_chapter.setdefault(chapter_of[span_block[qid]], Counter())[label["who"][1]] += 1
    for item in extras:
        if item["who"][0] == "char":
            by_chapter.setdefault(chapter_of[item["block_id"]], Counter())[item["who"][1]] += 1
    pairs = {}
    for counts in by_chapter.values():
        for left in counts:
            for right in counts:
                if left != right:
                    pairs.setdefault(left, Counter())[right] += min(counts[left], counts[right])
    return pairs


# A design without a gender gets men with women's voices and the reverse.
_GENDERED = re.compile(r"\b(?:male|female|man|woman|men|women|masculine|feminine|androgynous|non-binary|nonbinary"
                       r"|gender-?neutral)\b", re.I)
_SEX_WORD = {"m": "Male", "f": "Female", "n": "Androgynous"}


def gendered_description(description, sex):
    """``description`` stating the voice's gender, or None when neither it nor ``sex`` gives one."""
    if _GENDERED.search(description):
        return description
    word = _SEX_WORD.get(sex)
    if word is None:
        return None
    return f"{word} voice; {description[:1].lower()}{description[1:]}"


# Real-world words a voice designer understands, even when a book uses them as names.
_REAL_WORLD = frozenset("""
american british english scottish irish welsh australian canadian southern northern midwestern texan texas
cockney london received pronunciation rp yorkshire geordie scouse brummie lancashire west country cornish
boston bostonian brooklyn bronx york new zealand kiwi south african african nigerian kenyan jamaican caribbean
indian pakistani french german italian spanish portuguese russian polish dutch swedish norwegian danish finnish
greek turkish arabic persian chinese japanese korean mexican latin brazilian argentine appalachian cajun
california californian mid-atlantic transatlantic general standard neutral european asian scandinavian slavic
east west north south eastern western central midlands highland lowland gaelic hispanic
""".split())


def book_terms(source, ledger):
    """Proper nouns of the book a voice designer could not know: words capitalized
    mid-sentence at least twice (and every character name or alias word), minus
    real-world accent, language and place words."""
    counts, lower = Counter(), Counter()
    for block in source["blocks"]:
        for match in re.finditer(r"(?<=[a-z,;] )([A-Z][a-z]{2,}(?:[-'][A-Za-z]+)?)", block["text"]):
            counts[match.group(1).casefold()] += 1
        lower.update(re.findall(r"\b[a-z][a-z'-]{2,}\b", block["text"]))
    # A book's own term is capitalized more often than not ("Sendaria"); everyday
    # words capitalized in titles ("His Majesty") appear lowercase far more often.
    terms = {word for word, count in counts.items() if count >= 2 and lower[word] < count}
    for row in ledger["characters"]:
        for name in [row["name"], *row["aliases"]]:
            terms.update(word.casefold() for word in re.findall(r"[A-Z][\w'-]{2,}", name))
    return terms - _REAL_WORLD


def assign_accents(source, requests, meter, rows, terms, voices=None, *, progress=lambda message: None):
    """One request mapping the story's origins to distinct real-world accents.

    Returns {ledger id: accent}. Voices are designed in batches; a book-wide map
    keeps characters from the same place on one accent and gives different
    backgrounds different ones. An accent naming a book term is dropped.
    """
    if not rows:
        return {}
    by_name = {row["name"]: row["id"] for row in rows}
    characters = []
    for row in rows:
        note = row["note"]
        if voices and row["id"] in voices:
            note = f"{note}; already voiced: {voices[row['id']][:120]}"
        characters.append([row["name"], note])
    payload = {"mode": "narrator_accents", "v": PROMPT_VERSION, "book": source["book"]["title"],
               "characters": characters}

    def validate(response):
        groups = _unwrap(response.get("a")) if isinstance(response, dict) else None
        if not isinstance(groups, list):
            raise ValueError('response must be {"a":[[origin,accent,[name,name]]]}')
        result = {}
        for group in groups:
            if not isinstance(group, list) or len(group) < 3 or not isinstance(group[1], str):
                continue
            accent = " ".join(re.sub(r"[\[\]<>{}]", " ", group[1]).split())[:40]
            if not accent or any(word.casefold() in terms for word in re.findall(r"[A-Za-z][\w'-]{2,}", accent)):
                continue
            for name in group[2] if isinstance(group[2], list) else []:
                if isinstance(name, str) and name in by_name:
                    result.setdefault(by_name[name], accent)
        return result

    progress(f"Narrator read: assigning accents across {len(rows)} character(s)")
    return meter.run(requests, "voices", ACCENTS, payload, validate)


def design_voices(source, requests, meter, ledger, cast, partners, *, workers=1, progress=lambda message: None):
    """Fill descriptions/traits for voices that are not inherited series voices.

    A character with a lasting voice change is designed once per state in the
    same request (at the start of the book, and after each change), so the model
    contrasts them; ``add_voice_variants`` decides which state is the main voice.
    Returns (cast, designed count, {state key: profile}) with keys
    ``"start:<ledger id>"`` and change IDs.
    """
    # Series reconciliation renames IDs; series_origin keeps the local character ID.
    char_local = {f"char_{row['id']}": row for row in ledger["characters"]}
    owners = {}
    for item in cast["characters"]:
        local_id = item.get("series_origin", {}).get("local_character_id", item["id"])
        if local_id in char_local:
            owners.setdefault(item["voice_id"], char_local[local_id])
    narrator_voice = next(role["voice_id"] for role in cast["roles"] if role["id"] == "narrator_main")
    targets, inherited_changes = [], []
    for voice in cast["voices"]:
        row = owners.get(voice["id"])
        if "series_origin" in voice:
            # An inherited voice stays as it is; only its later states are designed.
            if row is not None and any(change["kind"] in IDENTITY_CHANGES for change in row.get("changes", [])):
                inherited_changes.append((voice, row))
            continue
        if row is None and voice["id"] != narrator_voice:
            continue
        targets.append((voice, row))
    if not targets and not inherited_changes:
        return cast, 0, {}
    narration = ledger["narration"]
    entries, changes_by_short = [], {}
    terms = book_terms(source, ledger)
    speaking_rows = [row for _voice, row in [*targets, *inherited_changes] if row is not None]
    accents = assign_accents(source, requests, meter, speaking_rows, terms,
                             {row["id"]: voice["description"] for voice, row in inherited_changes},
                             progress=progress)

    def with_accent(row, note):
        return f"{note}; accent: {accents[row['id']]}" if row["id"] in accents else note
    for number, (voice, row) in enumerate(targets, 1):
        if row is None:
            entries.append([f"v{number}", "Narrator", "?", f"Main narrator; narration mode {narration['mode'] or 'unknown'}.", "", ""])
            continue
        partner_names = [character(ledger, identifier)["name"] for identifier, _ in partners.get(row["id"], Counter()).most_common(3)
                         if character(ledger, identifier) is not None]
        later = [change for change in row.get("changes", []) if change["kind"] in IDENTITY_CHANGES]
        name = f"{row['name']} (at the start of the book, before any change)" if later else row["name"]
        entries.append([f"v{number}", name, row["sex"], with_accent(row, row["note"]),
                        " / ".join(item["quote"] for item in row["voice"][:4]), ", ".join(partner_names)])
        if later:
            changes_by_short[f"v{number}"] = f"start:{row['id']}"
        for change in later:
            # Designed beside the earlier voice, so the model contrasts the two.
            short = f"v{len(targets) + len(changes_by_short) + 1}"
            changes_by_short[short] = change["id"]
            entries.append([short, f"{row['name']} (later: {change['instruction']})", row["sex"],
                            with_accent(row, f"The same person after a {change['kind']} change: {change['instruction']}. "
                                             "Keep their accent and manner; change only what this describes."),
                            "", ", ".join(partner_names)])
    for voice, row in inherited_changes:
        for change in (item for item in row["changes"] if item["kind"] in IDENTITY_CHANGES):
            short = f"v{len(targets) + len(changes_by_short) + 1}"
            changes_by_short[short] = change["id"]
            entries.append([short, f"{row['name']} (later: {change['instruction']})", row["sex"],
                            f"Their voice so far: {voice['description'][:300]} Now the same person after a "
                            f"{change['kind']} change: {change['instruction']}. Keep their accent and manner.",
                            "", ""])
    by_short = {f"v{number}": voice for number, (voice, _row) in enumerate(targets, 1)
                if f"v{number}" not in changes_by_short}
    progress(f"Narrator read: designing {len(entries)} voice(s)")
    batches = [entries[start:start + VOICE_BATCH] for start in range(0, len(entries), VOICE_BATCH)]
    book = {"title": source["book"]["title"], "narration": narration["mode"] or "unknown"}

    def run(batch):
        payload = {"mode": "narrator_voices", "v": PROMPT_VERSION, "book": book, "voices": batch}
        expected = {entry[0] for entry in batch}
        sexes = {entry[0]: entry[2] for entry in batch}

        def validate(response):
            rows = _unwrap(response.get("v")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"v":[[id,description,speech_traits,traits]]}')
            result, errors = {}, []
            for row in rows:
                if not isinstance(row, list) or len(row) < 3:
                    continue
                row = _item_row(row, "v")
                if not _is(row[0], expected):
                    continue
                description = " ".join(str(row[1]).split())[:600] if isinstance(row[1], str) else ""
                try:
                    speech = validate_speech_traits(" ".join(str(row[2]).split())[:200])
                except ValueError as error:
                    errors.append(f"{row[0]}: {error}")
                    continue
                if not description:
                    errors.append(f"{row[0]}: description is empty")
                    continue
                gendered = gendered_description(description, sexes.get(row[0]))
                if gendered is None:
                    errors.append(f'{row[0]}: description must begin with "Male", "Female" or "Androgynous"')
                    continue
                description = gendered[:600]
                used = sorted({word for word in re.findall(r"[A-Za-z][\w'-]{2,}", f"{description} {row[2]}")
                               if word.casefold() in terms})
                if used:
                    # The designer never read the book: only real-world words describe a voice.
                    errors.append(f"{row[0]}: description and speech traits must use real-world terms only, "
                                  f"not words from the book ({', '.join(used[:6])})")
                    continue
                traits = []
                for pair in (row[3] if len(row) > 3 and isinstance(row[3], str) else "").split(";"):
                    name, _, value = pair.partition("=")
                    # "ReceivedPronunciation" -> "Received Pronunciation".
                    name, value = name.strip().lower()[:40], re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value.strip())[:80]
                    if name and value:
                        traits.append({"name": name, "value": value, "basis": "casting", "evidence": []})
                result[row[0]] = {"description": description, "speechTraits": speech, "traits": traits}
            missing = sorted(expected - set(result))
            if missing:
                errors.append(f"missing voices: {', '.join(missing[:20])}")
            if errors:
                raise ValueError("; ".join(errors[:12]))
            return result

        return meter.run(requests, "voices", VOICE, payload, validate)

    cast = deepcopy(cast)
    voices = {voice["id"]: voice for voice in cast["voices"]}
    change_profiles = {}
    for result in _parallel(run, batches, workers):
        for short, profile in result.items():
            if short in changes_by_short:
                change_profiles[changes_by_short[short]] = profile
            else:
                voices[by_short[short]["id"]].update(profile)
    return cast, len(entries), change_profiles


def _segment(block_id, start, text, *, role, kind, basis, reason, evidence=None, candidates=None,
             review_required=False, delivery=None):
    return {"id": f"{block_id}:{start}:{start + len(text)}", "block_id": block_id, "start": start,
            "end": start + len(text), "text": text, "kind": kind,
            "role_id": role["id"] if role else None, "voice_id": role["voice_id"] if role else None,
            "delivery": delivery or {"direction": "", "evidence": []},
            "attribution": {"basis": basis, "reason": reason, "evidence": evidence or [],
                            "candidates": candidates or [], "review_required": review_required}}


def _attach_notes(rows, notes, block_text, block_id, by_qid, stats):
    """Turn terse notes into each row's private ``inline_proposals`` (reviewed later).

    Each note must locate a unique exact span in its target row. Invalid notes are
    dropped and counted; the remove-only review still decides every survivor.
    """
    from .attribution_compact import parse_proposals
    pending = {}
    for note in notes:
        kind, value = note["target"]
        if kind == "q":
            row = by_qid.get(value)
            candidates = [row] if row is not None else []
        else:
            candidates = rows
        matches = [row for row in candidates if row["text"].count(note["raw"]["original"]) == 1]
        if len(matches) != 1:
            stats["dropped"] += 1
            continue
        row = matches[0]
        key = "edits" if note["kind"] == "edit" else "controls"
        base = row["delivery"]["direction"]
        try:
            parse_proposals({key: [note["raw"]]}, row["text"], f"{block_id} note", base_direction=base)
        except ValueError:
            stats["dropped"] += 1
            continue
        pending.setdefault(id(row), (row, {"edits": [], "controls": []}))[1][key].append(note["raw"])
    for row, payload in pending.values():
        try:
            cleaned = parse_proposals(payload, row["text"], f"{block_id} notes",
                                      base_direction=row["delivery"]["direction"])
        except ValueError:
            stats["dropped"] += len(payload["edits"]) + len(payload["controls"])
            continue
        if cleaned is not None:
            row["inline_proposals"] = cleaned
            stats["attached"] += len(cleaned["edits"]) + len(cleaned["controls"])


_JOINT = re.compile(r"\b(?:in unison|together|simultaneously|at the same time|at once|in chorus|both)\b", re.IGNORECASE)


def _joint_delivery_evidence(block_id, text, pieces):
    """Narration in the same paragraph that says the speakers speak together."""
    for piece in pieces:
        if piece["who"][0] != "narrator":
            continue
        narration = text[piece["start"]:piece["end"]]
        match = _JOINT.search(narration)
        if match:
            sentence_start = max(narration.rfind(".", 0, match.start()) + 1, 0)
            end = narration.find(".", match.end())
            quote = narration[sentence_start:end + 1 if end != -1 else len(narration)].strip()
            if quote and quote in text:
                return [{"block_id": block_id, "quote": quote[:240]}]
    return []


def change_spans(source, ledger):
    """Ledger voice changes as block-position ranges: {ledger_id: [(start, end, change)]}.

    An identity change ("age", "condition") lasts until its end, else to the end of
    the book; an equipment effect lasts until its end, else to the end of the chapter
    it starts in, so a forgotten "end" never runs past one chapter.
    """
    position = {block["id"]: index for index, block in enumerate(source["blocks"])}
    chapter_end = {}
    for index, block in enumerate(source["blocks"]):
        chapter_end[block["chapter_id"]] = index
    result = {}
    for row in ledger["characters"]:
        for change in row.get("changes", []):
            if change["start"] not in position:
                continue
            start = position[change["start"]]
            if change["kind"] in IDENTITY_CHANGES:
                limit = len(source["blocks"]) - 1
            else:
                limit = chapter_end[source["blocks"][start]["chapter_id"]]
            end = limit
            if change["end"] in position and position[change["end"]] > start:
                # The paragraph that ends the change is already back to normal.
                end = min(limit, position[change["end"]] - 1)
            result.setdefault(row["id"], []).append((start, end, change))
    return result


def _state_profile(profiles, key, fallback_voice, fallback_text):
    designed = (profiles or {}).get(key)
    if designed:
        return deepcopy(designed)
    return {"description": f"{fallback_voice['description']} {fallback_text}",
            "traits": [{"name": trait["name"], "value": trait["value"], "basis": "casting", "evidence": []}
                       for trait in fallback_voice.get("traits", [])],
            **({"speechTraits": fallback_voice["speechTraits"]} if "speechTraits" in fallback_voice else {})}


def add_voice_variants(cast, ledger, *, registry=None, source=None, profiles=None):
    """Voices for a character whose own voice changes during the book.

    The character's main voice is how they sound at the END of the book, which a
    later series book inherits (a child who grows up keeps the grown voice). Every
    other state, such as the young child before an age change, gets a dialogue
    role whose voice is ``derived_from`` that main voice; the renderer clones it
    from the main voice's reference with the change as its instruction. An
    inherited main voice cannot change, so a new book's later states become
    derived voices instead, and the last one carries into the next book
    (``through_end``).

    Returns (cast, {change_id: role_id}, {ledger_id: role_id for lines outside
    any change}). A change whose state is the main voice maps to no role.
    """
    cast = deepcopy(cast)
    roles = {role["id"]: role for role in cast["roles"]}
    voices = {voice["id"]: voice for voice in cast["voices"]}
    character_ids = {character["id"] for character in cast["characters"]}
    variant_roles, default_roles = {}, {}

    def add(row, base_role, base_voice, key, profile, change, instruction, label, *, through_end):
        voice_id, role_id = f"{base_voice['id']}_{key}", f"role_{row['id']}_{key}"
        # The instruction is the short direction the renderer clones with.
        voice = {"id": voice_id, "description": profile["description"], "traits": deepcopy(profile["traits"]),
                 "derived_from": {"voice_id": base_voice["id"], "kind": change["kind"],
                                  "instruction": " ".join(instruction.split())[:240],
                                  "evidence": deepcopy(change["evidence"]), "through_end": through_end}}
        if "speechTraits" in profile:
            voice["speechTraits"] = profile["speechTraits"]
        cast["voices"].append(voice)
        cast["roles"].append({"id": role_id, "kind": "dialogue", "character_id": base_role["character_id"],
                              "voice_id": voice_id, "description": f"{base_role['description']} {label}"})
        return role_id

    for row in ledger["characters"]:
        base_role = roles.get(f"role_{row['id']}")
        if base_role is None:
            continue
        base_voice = voices[base_role["voice_id"]]
        changes = sorted((change for change in row.get("changes", []) if change["kind"] in IDENTITY_CHANGES),
                         key=lambda change: change["number"] or 0)
        carried = row.get("carried_voice")
        if (carried and registry is not None and row["series"] in character_ids
                and base_role["character_id"] == row["series"]):
            # The changed voice this character still had when the earlier book ended.
            if carried not in voices:
                from .series import carried_voice
                voices[carried] = carried_voice(registry, source, carried)
                cast["voices"].append(voices[carried])
            role_id = f"role_{row['id']}_carried"
            cast["roles"].append({"id": role_id, "kind": "dialogue", "character_id": base_role["character_id"],
                                  "voice_id": carried,
                                  "description": f"{base_role['description']} Their changed voice from the earlier book."})
            default_roles[row["id"]] = role_id
        if not changes:
            continue
        if "series_origin" in base_voice:
            # Inherited main voice: each later state is derived from it; the last
            # state still in effect at the end carries into the next book.
            for change in changes:
                profile = _state_profile(profiles, change["id"], base_voice,
                                         f"Later ({change['kind']}): {change['instruction']}.")
                variant_roles[change["id"]] = add(
                    row, base_role, base_voice, change["id"], profile, change, change["instruction"],
                    f"Later in the book ({change['kind']}): {change['instruction']}.",
                    through_end=change["end"] is None and change is changes[-1])
            continue
        # A new character: the state at the end of the book is the main voice.
        final = changes[-1] if changes[-1]["end"] is None else None
        start_profile = _state_profile(profiles, f"start:{row['id']}", base_voice, "")
        if final is not None:
            base_voice.update(_state_profile(profiles, final["id"], base_voice,
                                             f"Later ({final['kind']}): {final['instruction']}."))
            first = changes[0]
            default_roles[row["id"]] = add(
                row, base_role, base_voice, "start", start_profile, first,
                start_profile["description"] or f"before: {first['instruction']}",
                f"Earlier in the book, before the {first['kind']} change.", through_end=False)
        else:
            base_voice.update(start_profile)
        for change in changes:
            if change is final:
                continue
            profile = _state_profile(profiles, change["id"], base_voice,
                                     f"Later ({change['kind']}): {change['instruction']}.")
            variant_roles[change["id"]] = add(
                row, base_role, base_voice, change["id"], profile, change, change["instruction"],
                f"At another point in the book ({change['kind']}): {change['instruction']}.", through_end=False)
    return cast, variant_roles, default_roles


def build_segments(source, cast, spans, labels, extras, ledger, notes=(), stats=None, variant_roles=None,
                   default_roles=None):
    """Exact-coverage segments; returns rows and fills ``stats`` with note/delivery counts."""
    stats = stats if stats is not None else {}
    stats.update({"attached": 0, "dropped": 0, "delivery": 0, "delivery_dropped": 0,
                  "simultaneous": 0, "simultaneous_unproven": 0, "variant_lines": 0, "effect_lines": 0,
                  "voice_exceptions": 0})
    roles = {role["id"]: role for role in cast["roles"]}
    variant_roles, default_roles = variant_roles or {}, default_roles or {}
    ranges = change_spans(source, ledger)
    block_position = {block["id"]: index for index, block in enumerate(source["blocks"])}
    voice_as = {note["target"][1]: note["raw"]["as"] for note in notes if note["kind"] == "voice_as"}
    notes = [note for note in notes if note["kind"] != "voice_as"]

    def voice_state(identifier, block_id, qid):
        """(variant role id or None, effect change or None, effect preset or None) for one line."""
        at = block_position[block_id]
        active = [change for start, end, change in ranges.get(identifier, []) if start <= at <= end]
        identity = [change for change in active if change["kind"] in IDENTITY_CHANGES]
        effects = [change for change in active if change["kind"] in EFFECT_PRESETS]
        exception = voice_as.get(qid)
        # Outside any change: the start-of-book voice (or the voice carried from
        # the earlier book); inside one: that change's voice, or None when it is
        # the main voice.
        baseline = default_roles.get(identifier)
        if exception == "normal":
            return baseline, None, None
        variant = variant_roles.get(identity[-1]["id"]) if identity else baseline
        if exception in EFFECT_PRESETS:
            return variant, None, exception
        return variant, (effects[-1] if effects else None), (effects[-1]["kind"] if effects else None)
    narrator = roles["narrator_main"]
    index = name_index(ledger)
    by_block = {}
    for span in spans:
        by_block.setdefault(span["block_id"], []).append(dict(labels[span["id"]], start=span["start"],
                                                              end=span["end"], tag=span["tag"], qid=span["id"]))
    for item in extras:
        by_block.setdefault(item["block_id"], []).append(dict(item, tag=None))
    notes_by_block = {}
    span_block = {span["id"]: span["block_id"] for span in spans}
    for note in notes:
        block_id = span_block.get(note["target"][1]) if note["target"][0] == "q" else note["target"][1]
        if block_id is not None:
            notes_by_block.setdefault(block_id, []).append(note)
    segments = []
    for block in source["blocks"]:
        text = block["text"]

        def quoted(value):
            return [{"block_id": block["id"], "quote": value}] if value and value.strip() and value in text else []

        narration_kind = "heading" if block["source"]["tag"] in HEADING_TAGS else "narration"
        pieces, cursor = [], 0
        for item in sorted(by_block.get(block["id"], []), key=lambda value: value["start"]):
            if item["start"] < cursor:
                continue
            if item["start"] > cursor:
                pieces.append({"start": cursor, "end": item["start"], "who": ("narrator",)})
            pieces.append(item)
            cursor = item["end"]
        if cursor < len(text):
            pieces.append({"start": cursor, "end": len(text), "who": ("narrator",)})
        rows, by_qid = [], {}
        for piece in pieces:
            who, piece_text = piece["who"], text[piece["start"]:piece["end"]]
            if who[0] == "narrator":
                row = _segment(block["id"], piece["start"], piece_text, role=narrator, kind=narration_kind,
                               basis="explicit", reason="Narration outside quotation marks.")
            else:
                kind = {"t": "thought", "l": "letter"}.get(piece.get("kind"), "dialogue")
                tag = piece.get("tag")
                tag_evidence = quoted(tag["quote"]) if tag else []
                label_evidence = quoted(piece.get("quote", ""))
                if who[0] == "multi" and not label_evidence:
                    label_evidence = _joint_delivery_evidence(block["id"], text, pieces)
                if who[0] == "multi" and not label_evidence:
                    # Joint delivery needs source evidence; otherwise the human chooses.
                    stats["simultaneous_unproven"] += 1
                    who = ("unresolved", who[1])
                delivery = {"direction": "", "evidence": []}
                if piece.get("delivery"):
                    # A direction is plain words; brackets are engine control syntax.
                    piece["delivery"] = " ".join(re.sub(r"[\[\]<>{}]", " ", piece["delivery"]).split())
                if piece.get("delivery"):
                    # The model's quote must support the delivery; an explicit tag only
                    # counts when it contains the delivery word ("whispered" for "whisper").
                    stem = re.sub(r"[^a-z]", "", piece["delivery"].split()[0].casefold())[:5]
                    tag_supports = bool(stem) and any(stem in item["quote"].casefold() for item in tag_evidence)
                    support = label_evidence or (tag_evidence if tag_supports else [])
                    if support:
                        delivery = {"direction": piece["delivery"], "evidence": support}
                        stats["delivery"] += 1
                    else:
                        stats["delivery_dropped"] += 1
                if who[0] == "unresolved":
                    candidates = [f"role_{identifier}" for identifier in who[1] if f"role_{identifier}" in roles]
                    row = _segment(block["id"], piece["start"], piece_text, role=None, kind=kind,
                                   basis="unresolved", reason="The narrator read could not settle the speaker.",
                                   candidates=candidates, review_required=True)
                elif who[0] == "multi":
                    row = _segment(block["id"], piece["start"], piece_text, role=None, kind=kind,
                                   basis="explicit", reason="The text says these characters speak together.",
                                   evidence=label_evidence, delivery=delivery)
                    row["simultaneous_speakers"] = [{"role_id": f"role_{identifier}",
                                                     "voice_id": roles[f"role_{identifier}"]["voice_id"]}
                                                    for identifier in who[1]]
                    stats["simultaneous"] += 1
                else:
                    role = roles[f"role_{who[1]}"]
                    variant, effect_change, preset = voice_state(who[1], block["id"], piece.get("qid"))
                    if piece.get("qid") in voice_as:
                        stats["voice_exceptions"] += 1
                    if variant is not None:
                        role = roles[variant]
                        stats["variant_lines"] += 1
                    explicit = bool(tag and tag["name"] and tag_character(index, tag["name"]) == who[1])
                    row = _segment(block["id"], piece["start"], piece_text, role=role, kind=kind,
                                   basis="explicit" if explicit else "context",
                                   reason=("Explicit speech tag." if explicit else
                                           "Verified against wider context." if piece.get("verified") else
                                           "Labeled by the narrator read from context."),
                                   evidence=tag_evidence if explicit else [], delivery=delivery)
                    if preset and kind == "dialogue":
                        # The change's own quotation, or this paragraph for a one-line exception.
                        evidence = deepcopy(effect_change["evidence"]) if effect_change else [
                            {"block_id": block["id"], "quote": piece_text}]
                        row["voice_effect"] = {"preset": preset, "evidence": evidence}
                        stats["effect_lines"] += 1
            previous = rows[-1] if rows else None
            if (previous is not None and previous["role_id"] == row["role_id"] and previous["kind"] == row["kind"]
                    and previous["attribution"]["basis"] == row["attribution"]["basis"] == "explicit"
                    and row["role_id"] == "narrator_main" and previous["end"] == row["start"]):
                previous["text"] += row["text"]
                previous["end"] = row["end"]
                previous["id"] = f"{block['id']}:{previous['start']}:{previous['end']}"
                continue
            rows.append(row)
            if piece.get("qid"):
                by_qid[piece["qid"]] = row
        if block["id"] in notes_by_block:
            _attach_notes(rows, notes_by_block[block["id"]], text, block["id"], by_qid, stats)
        segments.extend(rows)
    return segments


def narrated_analyze(source, client, cache_dir, *, series_registry=None, policy=None, workers=1,
                     repair_attempts=1, max_request_chars=200000, prep_window_chars=PREP_WINDOW_CHARS,
                     markup_chars=MARKUP_CHARS, cast_notes="", attribution_notes="",
                     progress=lambda message: None, meter_callback=lambda fraction, stage: None):
    """Run the experimental narrator read and return a validated analysis document.

    ``cast_notes`` and ``attribution_notes`` are optional operator notes (the
    configured review notes), sent as fallible "hints" to the ledger/identity
    requests and to markup/verification respectively.
    """
    from .contracts import readiness
    from .narration_boundaries import repair_narration_boundaries
    from .performance import BRACKET_DIRECTION, apply_bracket_delivery
    from .pipeline import DEFAULT_POLICY, CachedRequests, base_document, check_cast, fingerprint
    from .prompts import PROMPT_VERSION as GLOBAL_PROMPT_VERSION
    from .series import add_series_issues, reconcile_series_cast, series_context

    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from 1 to 8")
    if not source["blocks"]:
        raise ValueError("No text blocks were extracted")
    policy = policy or DEFAULT_POLICY
    requests = CachedRequests(client, cache_dir, repair_attempts=repair_attempts,
                              max_request_chars=max_request_chars, progress=progress)
    hints = hint_modes(cast_notes, attribution_notes)
    meter = Meter(client, hints)
    started_all = time.monotonic()
    try:
        prior = series_context(series_registry, source)["characters"] if series_registry is not None else []
        ledger = new_ledger(prior)
        spans = quote_spans(source)

        def stage(name, fraction, function):
            meter_callback(fraction, f"Narrator read: {name}")
            started = time.monotonic()
            try:
                return function()
            finally:
                meter.timed(name, time.monotonic() - started)

        ledger, windows = stage("prep", 0.0, lambda: prep_read(source, requests, meter, ledger,
                                                                 window_chars=prep_window_chars, progress=progress))
        ledger, merge_outcomes = stage("identity_check", 0.35, lambda: check_merges(
            source, requests, meter, ledger, workers=workers, progress=progress))
        ledger, alias_outcomes = stage("identity_check", 0.37, lambda: check_aliases(
            source, requests, meter, ledger, workers=workers, progress=progress))
        labels, extras, notes, dropped_markup, groups = stage("markup", 0.4, lambda: markup(
            source, requests, meter, ledger, spans, policy=policy, workers=workers,
            group_chars=markup_chars, progress=progress))
        ledger, new_speakers = add_new_speakers(ledger, labels, extras)
        ledger, labels, extras, duplicate_outcomes = stage("identity_check", 0.65, lambda: check_duplicates(
            source, requests, meter, ledger, labels, extras, spans, workers=workers, progress=progress))
        ledger, series_outcomes = stage("identity_check", 0.7, lambda: check_series(
            source, requests, meter, ledger, labels, spans, workers=workers, progress=progress))
        labels, verify_stats = stage("verify", 0.75, lambda: verify(
            source, requests, meter, ledger, spans, labels, workers=workers, progress=progress))
        cast = build_cast(source, ledger, _speaking_ids(labels, extras))
        check_cast(source, cast)
        issues = []
        series_provenance = None
        if series_registry is not None:
            confirmed = {f"char_{row['id']}": row["series"] for row in ledger["characters"]
                         if row.get("series_confirmed") and row["series"] is not None}
            cast, additions, series_provenance = reconcile_series_cast(source, cast, series_registry,
                                                                       checked_matches=confirmed)
            add_series_issues(issues, additions, series_provenance)
        partners = _partners(source, labels, extras, spans)
        cast, designed, change_profiles = stage("voices", 0.85, lambda: design_voices(
            source, requests, meter, ledger, cast, partners, workers=workers, progress=progress))
        check_cast(source, cast)
        for question in ledger["questions"]:
            if question["answer"] is None:
                issues.append({"id": f"narrator_question_{question['id']}", "kind": "identity_question",
                               "target": question["block_id"], "reason": question["text"], "resolved": False})
        for outcome in merge_outcomes:
            if outcome["verdict"] not in ("y", "superseded"):
                issues.append({"id": f"narrator_merge_{outcome['from']}_{outcome['into']}", "kind": "identity_merge",
                               "target": f"char_{outcome['into']}",
                               "reason": f"A prep-read merge was not confirmed ({outcome['verdict']}); entries kept separate.",
                               "resolved": False})
        for outcome in duplicate_outcomes:
            if outcome["verdict"] == "?" and character(ledger, outcome["from"]) and character(ledger, outcome["into"]):
                issues.append({"id": f"narrator_duplicate_{outcome['from']}_{outcome['into']}", "kind": "identity_duplicate",
                               "target": f"char_{outcome['from']}",
                               "reason": f"Possibly the same person as {character(ledger, outcome['into'])['name']}; "
                                         "kept separate with separate voices. Review casting.",
                               "resolved": False})
        for outcome in series_outcomes:
            if outcome["verdict"] == "n":
                issues.append({"id": f"narrator_series_{outcome['id']}", "kind": "series_identity",
                               "target": outcome["series"],
                               "reason": "The series check judged a same-named character in this book to be a different "
                                         "person; it received a new voice. Review if this is the returning character.",
                               "resolved": False})
        warnings = cast_audit(cast)
        for warning in warnings:
            issues.append({"id": f"narrator_cast_{warning['kind']}_{warning['character_id']}", "kind": "cast_audit",
                           "target": warning["character_id"], "reason": warning["reason"], "resolved": False})
        cast, variant_roles, default_roles = add_voice_variants(cast, ledger, registry=series_registry, source=source,
                                                                profiles=change_profiles)
        check_cast(source, cast)
        for row in ledger["characters"]:
            for change in row.get("changes", []):
                if change["kind"] in EFFECT_PRESETS and change["end"] is None:
                    issues.append({"id": f"narrator_effect_{change['id']}", "kind": "voice_change",
                                   "target": change["start"],
                                   "reason": f"The {change['kind']} effect on {row['name']} has no end in the text; "
                                             "it was ended at the end of its chapter. Check where it stops.",
                                   "resolved": False})
        segment_stats = {}
        segments = build_segments(source, cast, spans, labels, extras, ledger, notes, segment_stats,
                                  variant_roles=variant_roles, default_roles=default_roles)
        metrics = meter.summary()
        metrics["wall_seconds"] = round(time.monotonic() - started_all, 3)
        metrics["cache_hits"] = requests.cache_hits
        provenance = {
            "analyzer_version": "0.1.0", "prompt_version": GLOBAL_PROMPT_VERSION,
            "source_fingerprint": fingerprint(source), "provider": client.identity,
            "performance_policy": policy, "attribution_workers": workers, "strategy": STRATEGY,
            "narrator_read": {
                "version": VERSION, "prompt_version": PROMPT_VERSION,
                "prep_windows": windows, "markup_groups": groups, "quote_spans": len(spans),
                "extra_spans": len(extras), "dropped_extra_spans": dropped_markup["extras"],
                "dropped_notes": dropped_markup["notes"], "subdivided_markup_groups": dropped_markup["subdivided_groups"],
                "segments": segment_stats,
                "new_markup_speakers": new_speakers, "merges": merge_outcomes,
                "series_checks": series_outcomes, "duplicate_checks": duplicate_outcomes,
                "alias_checks": alias_outcomes,
                "verification": verify_stats, "voices_designed": designed,
                "self_check": self_check_report(verify_stats, duplicate_outcomes, warnings,
                                                sum(1 for row in segments if row["role_id"] is None)),
                "dropped_ledger_claims": dict(ledger["dropped"]),
                "ledger": {key: deepcopy(ledger[key]) for key in ("characters", "narration", "questions")},
                "voice_changes": [{"character": row["id"], **{key: change[key] for key in ("id", "kind", "instruction", "start", "end")},
                                   "role_id": variant_roles.get(change["id"])}
                                  for row in ledger["characters"] for change in row.get("changes", [])],
                "hints": {kind: notes.strip() for kind, notes in (("cast", cast_notes), ("attribution", attribution_notes))
                          if notes.strip()},
                "metrics": metrics,
            },
            "corrections": [],
        }
        if series_provenance is not None:
            provenance["series"] = series_provenance
        document = base_document(source, cast, segments, issues, provenance)
        if any("inline_proposals" in row for row in document["segments"]):
            from .attribution_performance import review_inline_performance
            meter.default_stage = "performance"
            started = time.monotonic()
            try:
                document = review_inline_performance(document, requests, workers=min(5, workers), progress=progress)
            finally:
                meter.default_stage = None
                meter.timed("performance", time.monotonic() - started)
        else:
            for row in document["segments"]:
                row.pop("inline_proposals", None)
        metrics = meter.summary()
        metrics["wall_seconds"] = round(time.monotonic() - started_all, 3)
        metrics["cache_hits"] = requests.cache_hits
        document["provenance"]["narrator_read"]["metrics"] = metrics
        document = apply_bracket_delivery(document, policy.get("bracketed_text", BRACKET_DIRECTION))
        document = repair_narration_boundaries(document, progress=progress)
        state = readiness(document)
        if state["errors"]:
            raise ValueError("; ".join(state["errors"][:12]))
        document["readiness"] = state
        meter_callback(1.0, "Narrator read complete")
        total = metrics["total"]
        report = document["provenance"]["narrator_read"]["self_check"]
        progress("Narrator read self-check: rechecked "
                 + (", ".join(f"{count} {reason.replace('_', ' ')}" for reason, count in sorted(report["lines_rechecked"].items())) or "nothing")
                 + f"; changed {sum(report['lines_changed'].values())}; "
                 + f"{sum(report['cast_warnings'].values())} cast warning(s); {report['unresolved_lines']} line(s) left for speaker choice")
        progress(f"Narrator read: {total['provider_calls']} provider call(s), {total['logical_requests']} request(s); "
                 f"prompt {total['prompt_tokens']} / completion {total['completion_tokens']} tokens "
                 f"({total['reasoning_tokens']} reasoning); {metrics['wall_seconds']:.0f}s")
        return document
    finally:
        meter.close()
