"""Capitalization families for review, keeping exact stored identities intact.

These conservative spelling rules are not contextual word-sense analysis. Known
English capitonyms and short initialisms remain separate; different spelling,
accents, apostrophes, and hyphens never become aliases through this module.
"""

from collections import defaultdict
from copy import deepcopy
import json


_CAPITONYMS = frozenset({"polish", "job", "reading", "august", "nice", "mobile"})
# Even an unfortunate name classification must not turn the country initialism
# into the ordinary pronoun. Other short all-capital forms use the guard below.
_INITIALISMS = frozenset({"US"})


def natural_spelling(word):
    return (word.isupper(), word.lower(), word)


def named_families(entries):
    """A natural-case proper name can identify its shouted spelling."""
    return {entry["word"].lower() for entry in entries
            if entry.get("kind") in {"name", "place"} and not entry["word"].isupper()}


def case_key(word, named=()):
    lowered = word.lower()  # casefold would also conflate distinct ß/ss spellings.
    if lowered in _CAPITONYMS:
        return (lowered, "capital" if word[:1].isupper() else "lower")
    if word in _INITIALISMS or (word.isupper() and sum(letter.isalpha() for letter in word) <= 3
                                and lowered not in named):
        return (lowered, "initialism", word)
    return (lowered,)


def entry_words(entry):
    return tuple(entry.get("variants", [entry["word"]]))


def find_pronunciation(entries, word):
    """Resolve either a displayed spelling or one of its source variants."""
    return next((entry for entry in entries if entry["word"] == word), None) or next(
        (entry for entry in entries if word in entry_words(entry)), None)


def family_words(word, entries, current_words):
    named = named_families(entries)
    key = case_key(word, named)
    return sorted({item for item in current_words if case_key(item, named) == key}, key=natural_spelling)


def group_pronunciations(entries, current_words):
    """Project ordered nearest-book-first records into current spelling groups.

    Human choices beat guesses at any depth; nearest-book human choices beat
    older choices. Conflicting explicit choices in the same book remain exact
    and visible rather than being silently decided by capitalization sorting.
    """
    from .pronunciations import word_id
    named = named_families(entries)
    groups, forms = defaultdict(list), defaultdict(list)
    for entry in entries:
        groups[case_key(entry["word"], named)].append(entry)
    for word in current_words:
        forms[case_key(word, named)].append(word)
    owners = {owner: index for index, owner in enumerate(dict.fromkeys(
        entry.get("origin_directory", "") for entry in entries))}

    def rank(entry):
        return (owners[entry.get("origin_directory", "")], *natural_spelling(entry["word"]))

    def project(selected, variants, *, conflict=False):
        row = deepcopy(selected)
        variants = sorted(variants, key=natural_spelling)
        word = selected["word"] if selected["word"] in variants else variants[0]
        if selected["status"] == "pending" and word.isupper() and not variants[0].isupper():
            word = variants[0]
        if word != selected["word"]:
            row["origin_word"] = selected["word"]
        row.update(word=word, id=word_id(word), variants=variants)
        if conflict:
            row["case_conflict"] = True
        return row

    result = []
    for key, members in groups.items():
        variants = forms.get(key)
        if not variants:
            continue
        explicit = sorted((entry for entry in members if entry["status"] != "pending"), key=rank)
        if explicit:
            nearest = explicit[0].get("origin_directory", "")
            chosen = [entry for entry in explicit if entry.get("origin_directory", "") == nearest]
            signatures = defaultdict(list)
            for entry in chosen:
                signatures[json.dumps([entry["status"], entry.get("pronunciation")], sort_keys=True)].append(entry)
            if len(signatures) == 1:
                result.append(project(chosen[0], variants))
                continue
            # Preserve every legacy explicit choice. Unclaimed capitalization
            # variants cannot inherit one arbitrarily while the choices clash.
            assigned = set()
            for same_choice in signatures.values():
                selected_forms = [entry["word"] for entry in same_choice if entry["word"] in variants]
                if selected_forms:
                    selected = next(entry for entry in same_choice if entry["word"] in selected_forms)
                    result.append(project(selected, selected_forms, conflict=True))
                    assigned.update(selected_forms)
            for word in sorted(set(variants) - assigned, key=natural_spelling):
                pending = next((entry for entry in members if entry["word"] == word and entry["status"] == "pending"), None)
                if pending is None:
                    pending = deepcopy(chosen[0])
                    pending.pop("proposal", None)
                else:
                    pending = deepcopy(pending)
                pending.update(status="pending", reason="Capitalization variants have conflicting saved decisions; review this spelling explicitly.")
                pending.pop("pronunciation", None)
                result.append(project(pending, [word], conflict=True))
            continue
        ordered = sorted(members, key=rank)
        selected = project(ordered[0], variants)
        proposal = next((entry for entry in ordered if "proposal" in entry), None)
        if proposal is not None:
            selected["proposal"] = deepcopy(proposal["proposal"])
            selected["proposal_origin_directory"] = proposal.get("proposal_origin_directory", proposal.get("origin_directory"))
        result.append(selected)
    return sorted(result, key=lambda entry: (entry["word"].lower(), entry["word"]))
