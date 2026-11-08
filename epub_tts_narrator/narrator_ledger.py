"""Running character ledger carried through the narrator read.

The ledger is the compact state a human narrator keeps while reading: who
appears, their names, short notes, how they sound (exact quotes), narration
mode, and open identity questions. It is rendered as short text lines for the
prompt and updated from terse model deltas. Unsupported optional claims (bad
quotes, unknown IDs) are dropped and counted rather than repaired, so a small
model mistake never triggers another request. An inexact quote keeps only its
longest verbatim piece (`exact_quote`).
"""

from copy import deepcopy
import re

MAX_NOTE = 120
MAX_ALIASES = 10
MAX_QUOTE = 240
SEXES = {"m", "f", "n", "?"}


def _clean(value, limit):
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("|", "/").split())[:limit]


# Where models elide, join or comment on a quotation.
_QUOTE_BREAKS = re.compile(r'\s*(?:\.\.\.|…|—|–|"|“|”)\s*')


def exact_quote(quote, text, limit=MAX_QUOTE):
    """The model's quote when it is exact text; otherwise its longest exact piece.

    A quote is split where models elide, join or annotate (ellipses, dashes,
    quotation marks); a piece of at least three words that occurs verbatim in
    ``text`` is kept. The result is always verbatim source text, never a repair.
    """
    if not isinstance(quote, str) or not quote.strip():
        return None
    if len(quote) <= limit and quote in text:
        return quote
    pieces = [piece.strip(" ,;:") for piece in _QUOTE_BREAKS.split(quote)]
    exact = [piece for piece in pieces if len(re.findall(r"\w+", piece)) >= 3 and len(piece) <= limit and piece in text]
    return max(exact, key=len) if exact else None


def _norm(name):
    return " ".join(re.findall(r"\w+", name.casefold()))


def ledger_id(token):
    """Normalize a model-supplied character ID: 12, "12", "c12" and "s12" all mean "12"."""
    if type(token) is int and token > 0:
        return str(token)
    if isinstance(token, str):
        match = re.fullmatch(r"\s*[A-Za-z]?(\d+)\s*", token)
        if match:
            return str(int(match.group(1)))
    return None


def new_ledger(prior_characters=()):
    """Seed with prior-series characters (from series_context) as unseen entries.

    Every character ID is a plain number in one namespace (prior characters
    first), so a model that drops a prefix can never pick the wrong entry.
    """
    ledger = {"characters": [], "narration": {"mode": None, "narrator": None},
              "questions": [], "merges": [], "next": len(prior_characters) + 1, "dropped": {}}
    for index, prior in enumerate(prior_characters, 1):
        ledger["characters"].append({
            "id": str(index), "name": _clean(prior["name"], 80),
            "aliases": [_clean(alias, 80) for alias in prior.get("aliases", [])][:MAX_ALIASES],
            "sex": "?", "note": _clean(prior.get("description", ""), MAX_NOTE),
            # The earlier book's description, kept for the series check when the
            # prep read rewrites the note from this book's text.
            "prior_note": _clean(prior.get("description", ""), MAX_NOTE), "first": None,
            "voice": [], "series": prior["id"], "series_evidence": [], "seen": False, "merged": []})
    return ledger


def character(ledger, identifier):
    identifier = ledger_id(identifier)
    return next((row for row in ledger["characters"] if row["id"] == identifier), None)


def render(ledger, *, include_unseen=True, include_questions=True):
    """Return (cast_text, prior_text) prompt lines."""
    def line(row):
        return "|".join([row["id"], row["name"], ";".join(row["aliases"]), row["sex"], row["note"]])
    seen = [line(row) for row in ledger["characters"] if row["series"] is None or row["seen"]]
    prior = [line(row) for row in ledger["characters"] if row["series"] is not None and not row["seen"]]
    narration = ledger["narration"]
    if narration["mode"]:
        seen.insert(0, f"narration|{narration['mode']}|{narration['narrator'] or ''}")
    for question in ledger["questions"] if include_questions else []:
        if question["answer"] is None:
            seen.append(f"question|{question['id']}|{question['text']}")
    return "\n".join(seen), ("\n".join(prior) if include_unseen else "")


def _drop(ledger, kind):
    ledger["dropped"][kind] = ledger["dropped"].get(kind, 0) + 1


def _rows(value, kind, minimum, ledger):
    """Return well-formed rows; malformed optional entries are dropped, not repaired."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{kind} must be an array of arrays")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) < minimum:
            _drop(ledger, kind)
            continue
        rows.append(row)
    return rows


def apply_delta(ledger, delta, blocks, window_numbers):
    """Apply one prep-read response; return the updated copy.

    ``blocks`` maps block_id -> text; ``window_numbers`` maps the paragraph
    numbers shown in this window to block IDs. Raises ValueError only for
    structural problems the model must repair.
    """
    if not isinstance(delta, dict):
        raise ValueError("response must be an object")
    allowed = {"new", "seen", "alias", "note", "voice", "same", "nar", "ask", "done"}
    ledger = deepcopy(ledger)
    for key in set(delta) - allowed:
        # Unknown optional keys are ignored and counted, not repaired.
        _drop(ledger, "unknown_key")
    by_name = {}

    def quote_ok(number, quote):
        block_id = window_numbers.get(number) if type(number) is int else None
        quote = exact_quote(quote, blocks[block_id]) if block_id is not None else None
        return {"block_id": block_id, "quote": quote} if quote else None

    def resolve(token):
        row = character(ledger, token)
        if row is None and isinstance(token, str):
            row = by_name.get(_norm(token))
        return row

    for row in _rows(delta.get("new"), "new", 1, ledger):
        name = _clean(row[0], 80)
        if not name:
            _drop(ledger, "new")
            continue
        matches = [item for item in ledger["characters"]
                   if _norm(name) in {_norm(item["name"]), *map(_norm, item["aliases"])}]
        # Only a main-name match is the same entry. A match on another entry's alias
        # (which the model may have attached wrongly) creates the entry anyway; the
        # duplicate check decides.
        existing = next((item for item in matches if (item["series"] is None or item["seen"])
                         and _norm(item["name"]) == _norm(name)), None)
        if existing is None and any(item["series"] is None or item["seen"] for item in matches):
            _drop(ledger, "new_alias_overlap")
            matches = []
        if existing is not None:
            by_name[_norm(name)] = existing
            _drop(ledger, "new_duplicate")
            continue
        if len(matches) == 1:
            # The name matches exactly one prior-book character. A name never
            # establishes identity: mark it seen without evidence, so the
            # identity check (or human review) must confirm it.
            matches[0]["seen"] = True
            number = row[4] if len(row) > 4 else None
            if matches[0]["first"] is None and type(number) is int:
                matches[0]["first"] = window_numbers.get(number)
            by_name[_norm(name)] = matches[0]
            _drop(ledger, "new_matched_prior_name")
            continue
        aliases = [_clean(alias, 80) for alias in (row[1] if len(row) > 1 and isinstance(row[1], str) else "").split(";")]
        sex = row[2] if len(row) > 2 and isinstance(row[2], str) and row[2] in SEXES else "?"
        number = row[4] if len(row) > 4 else None
        record = {"id": str(ledger["next"]), "name": name,
                  "aliases": [alias for alias in dict.fromkeys(aliases) if alias and alias != name][:MAX_ALIASES],
                  "sex": sex, "note": _clean(row[3], MAX_NOTE) if len(row) > 3 else "",
                  "first": window_numbers.get(number) if type(number) is int else None,
                  "voice": [], "series": None, "series_evidence": [], "seen": True, "merged": []}
        ledger["next"] += 1
        ledger["characters"].append(record)
        by_name[_norm(name)] = record
    for row in _rows(delta.get("seen"), "seen", 1, ledger):
        record = resolve(row[0])
        if record is None or record["series"] is None:
            _drop(ledger, "seen")
            continue
        record["seen"] = True
        evidence = quote_ok(row[1], row[2]) if len(row) > 2 else None
        if evidence is not None and record["first"] is None:
            record["first"] = evidence["block_id"]
        if evidence is not None and evidence not in record["series_evidence"]:
            record["series_evidence"].append(evidence)
    for row in _rows(delta.get("alias"), "alias", 2, ledger):
        record, alias = resolve(row[0]), _clean(row[1], 80)
        if record is None or not alias:
            _drop(ledger, "alias")
            continue
        if alias != record["name"] and alias not in record["aliases"] and len(record["aliases"]) < MAX_ALIASES:
            record["aliases"].append(alias)
    for row in _rows(delta.get("note"), "note", 2, ledger):
        record = resolve(row[0])
        if record is None:
            _drop(ledger, "note")
            continue
        record["note"] = _clean(row[1], MAX_NOTE)
    for row in _rows(delta.get("voice"), "voice", 3, ledger):
        record, evidence = resolve(row[0]), quote_ok(row[1], row[2])
        if record is None or evidence is None:
            _drop(ledger, "voice")
            continue
        if evidence not in record["voice"] and len(record["voice"]) < 6:
            record["voice"].append(evidence)
    for row in _rows(delta.get("same"), "same", 4, ledger):
        source, target = resolve(row[0]), resolve(row[1])
        evidence = quote_ok(row[2], row[3])
        if (source is None or target is None or source is target or evidence is None
                or (source["series"] and target["series"] and source["series"] != target["series"])):
            _drop(ledger, "same")
            continue
        merge(ledger, source["id"], target["id"], evidence)
    nar = delta.get("nar")
    if nar is not None:
        if not isinstance(nar, list) or not nar or not isinstance(nar[0], str) or nar[0] not in {"third", "first", "mixed"}:
            _drop(ledger, "nar")
        else:
            narrator = resolve(nar[1]) if len(nar) > 1 else None
            ledger["narration"] = {"mode": nar[0], "narrator": narrator["id"] if narrator else None}
    for row in _rows(delta.get("ask"), "ask", 2, ledger):
        block_id = window_numbers.get(row[0]) if type(row[0]) is int else None
        text = _clean(row[1], 160)
        if block_id is None or not text:
            _drop(ledger, "ask")
            continue
        ledger["questions"].append({"id": f"k{len(ledger['questions']) + 1}", "block_id": block_id,
                                    "text": text, "answer": None})
    for row in _rows(delta.get("done"), "done", 2, ledger):
        question = next((item for item in ledger["questions"] if item["id"] == row[0]), None)
        answer = resolve(row[1])
        if question is None or question["answer"] is not None:
            _drop(ledger, "done")
            continue
        question["answer"] = answer["id"] if answer else "-"
    return ledger


def merge(ledger, source_id, target_id, evidence):
    source, target = character(ledger, source_id), character(ledger, target_id)
    snapshot = deepcopy(source)
    for alias in [source["name"], *source["aliases"]]:
        if alias != target["name"] and alias not in target["aliases"]:
            target["aliases"].append(alias)
    target["aliases"] = target["aliases"][:MAX_ALIASES * 2]
    target["voice"].extend(item for item in source["voice"] if item not in target["voice"])
    if target["series"] is None and source["series"] is not None:
        target["series"], target["series_evidence"] = source["series"], source["series_evidence"]
    target["seen"] = target["seen"] or source["seen"]
    target["merged"].append(snapshot)
    ledger["characters"].remove(source)
    if ledger["narration"]["narrator"] == source_id:
        ledger["narration"]["narrator"] = target_id
    ledger["merges"].append({"from": source_id, "into": target_id, "evidence": evidence, "status": "applied"})


def undo_merge(ledger, source_id, target_id):
    """Split a rejected merge back into its two original entries."""
    ledger = deepcopy(ledger)
    target = character(ledger, target_id)
    snapshot = next(item for item in target["merged"] if item["id"] == source_id)
    target["merged"].remove(snapshot)
    removed = {snapshot["name"], *snapshot["aliases"]}
    target["aliases"] = [alias for alias in target["aliases"] if alias not in removed]
    target["voice"] = [item for item in target["voice"] if item not in snapshot["voice"]]
    if snapshot["series"] is not None and target["series"] == snapshot["series"]:
        target["series"], target["series_evidence"] = None, []
    ledger["characters"].append(snapshot)
    for row in ledger["merges"]:
        if row["from"] == source_id and row["into"] == target_id and row["status"] == "applied":
            row["status"] = "rejected"
    return ledger


def contextual_series_evidence(row):
    """Series evidence quotes containing words beyond the character's own names."""
    names = set(" ".join(_norm(name) for name in [row["name"], *row["aliases"]]).split())
    return [item for item in row["series_evidence"]
            if set(re.findall(r"\w+", item["quote"].casefold())) - names]


def name_index(ledger, *, include_unseen=False):
    """Normalized name/alias (and unique first word) -> character id; ambiguous keys omitted.

    A main name outranks other entries' aliases, so a wrongly attached alias never
    hides the character who actually bears that name.
    """
    primary, owners = {}, {}
    rows = [row for row in ledger["characters"]
            if not (row["series"] is not None and not row["seen"] and not include_unseen)]
    for row in rows:
        key = _norm(row["name"])
        if key:
            primary.setdefault(key, set()).add(row["id"])
    for row in rows:
        for name in [row["name"], *row["aliases"]]:
            key = _norm(name)
            if not key:
                continue
            owners.setdefault(key, set()).add(row["id"])
            words = key.split()
            if len(words) > 1:
                owners.setdefault("first:" + words[0], set()).add(row["id"])
    for key, ids in primary.items():
        owners[key] = ids
    return {key: next(iter(ids)) for key, ids in owners.items() if len(ids) == 1}


def tag_character(index, name):
    """Map an explicit tag name to one ledger id, if unambiguous."""
    key = _norm(name)
    return index.get(key) or index.get("first:" + key)
