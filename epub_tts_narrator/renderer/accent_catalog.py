"""The accents a book's voices are cast with: short names grouped into families of similar accents.

Accent assignment (``accent_assignment.py``) asks the model to choose from
this list so a series uses the same few words for the same accent, and the
accent screen offers it as a picker that shows each accent's near neighbours
and which groups already use it. Accents in one family sound related, so two
places that should sound related yet distinct (provinces of one kingdom,
rival cities) get two accents of one family; places far apart get different
families. ``NEAR`` links close accents across families.

Fish publishes no list of the accents Drama 3 knows, so ``rare`` is an
estimate, not a measurement: an accent seldom heard in recorded English,
which the engine may blur into its family's common accents. Listening can
change it.

Names are written without the word "accent", which the cue adds.
"""

from __future__ import annotations

import re

# (family, accents); an accent ending in "*" is rarely heard (see module doc).
FAMILIES = (
    ("England, south", ("upper-class English", "Estuary", "Cockney", "West Country", "Cornish*", "East Anglian*")),
    ("England, Midlands and north", ("Brummie", "Black Country*", "Yorkshire", "Lancashire", "Mancunian", "Scouse",
                                     "Geordie", "Cumbrian*")),
    ("Scotland", ("Scottish", "Glaswegian", "Edinburgh", "Highland", "Doric*")),
    ("Ireland", ("Irish", "Dublin", "Cork*", "Kerry*", "Northern Irish")),
    ("Wales and islands", ("Welsh", "North Welsh*", "Manx*")),
    ("US, general and west", ("General American", "Midwestern", "Minnesotan", "Californian")),
    ("US, northeast", ("New York", "Brooklyn", "Boston", "New England*", "Philadelphia*", "Mid-Atlantic")),
    ("US, south", ("Southern American", "Deep South", "Texan", "Appalachian", "Cajun*")),
    ("Canada", ("Canadian", "French Canadian", "Newfoundland*")),
    ("Caribbean", ("Jamaican", "Trinidadian*", "Bahamian*")),
    ("Oceania", ("Australian", "New Zealand")),
    ("Africa", ("South African", "Afrikaner*", "Nigerian", "Ghanaian*", "Kenyan*", "Ethiopian*")),
    ("South Asia", ("Indian English", "Pakistani", "Bangladeshi*", "Sri Lankan*")),
    ("East and Southeast Asia", ("Chinese", "Hong Kong*", "Japanese", "Korean", "Filipino", "Vietnamese*", "Thai*",
                                 "Singaporean", "Malaysian*")),
    ("Middle East and North Africa", ("Arabic", "Egyptian*", "Lebanese*", "Israeli", "Persian", "Turkish")),
    ("Western Europe", ("French", "Belgian*", "Swiss*", "German", "Austrian", "Dutch", "Flemish*")),
    ("Southern Europe", ("Italian", "Sicilian*", "Spanish", "Catalan*", "Portuguese", "Greek")),
    ("Nordic", ("Scandinavian", "Swedish", "Norwegian", "Danish*", "Finnish*", "Icelandic*")),
    ("Eastern Europe", ("Russian", "Ukrainian", "Polish", "Czech*", "Hungarian*", "Romanian*", "Serbian*")),
    ("Latin America", ("Mexican", "Puerto Rican", "Cuban", "Colombian*", "Argentine*", "Brazilian", "Chilean*")),
)

# Close accents in different families (symmetric).
NEAR = (
    ("Mid-Atlantic", "upper-class English"), ("Northern Irish", "Scottish"), ("Northern Irish", "Glaswegian"),
    ("Highland", "Irish"), ("Newfoundland", "Irish"), ("Canadian", "Midwestern"), ("Canadian", "General American"),
    ("New Zealand", "South African"), ("Australian", "South African"), ("Afrikaner", "Dutch"),
    ("French Canadian", "French"), ("Cajun", "French Canadian"), ("Swiss", "German"), ("Flemish", "Dutch"),
    ("Belgian", "French"), ("Mexican", "Spanish"), ("Argentine", "Italian"), ("Brazilian", "Portuguese"),
    ("Singaporean", "Chinese"), ("Scandinavian", "Dutch"), ("Israeli", "Russian"),
)

# Other words for a listed accent; a reply using one is saved under the listed name.
ALIASES = {
    "posh English": "upper-class English", "English": "upper-class English", "British": "upper-class English",
    "Transatlantic": "Mid-Atlantic", "old Hollywood": "Mid-Atlantic", "London": "Cockney",
    "American": "General American", "Southern": "Southern American", "Southern US": "Southern American",
    "Indian": "Indian English", "Kiwi": "New Zealand", "Aussie": "Australian", "Liverpool": "Scouse",
    "Manchester": "Mancunian", "Birmingham": "Brummie", "Newcastle": "Geordie", "Glasgow": "Glaswegian",
    "Scots": "Scottish", "Irish English": "Irish", "Ulster": "Northern Irish", "Belfast": "Northern Irish",
    "Brazilian Portuguese": "Brazilian", "Mexican Spanish": "Mexican", "Quebecois": "French Canadian",
    "Québécois": "French Canadian", "Nordic": "Scandinavian", "Middle Eastern": "Arabic", "Iranian": "Persian",
    "Mandarin": "Chinese", "Cantonese": "Hong Kong", "Castilian": "Spanish", "New Yorker": "New York",
}

STRENGTH = ("light", "soft", "slight", "faint", "mild", "gentle", "broad", "thick", "heavy", "strong", "plummy")


def _entries():
    for family, accents in FAMILIES:
        for accent in accents:
            yield accent.rstrip("*"), family, accent.endswith("*")


ACCENTS = {name: {"family": family, "rare": rare} for name, family, rare in _entries()}
_BY_KEY = {name.casefold(): name for name in ACCENTS}
_BY_KEY.update({alias.casefold(): name for alias, name in ALIASES.items()})
_NEAR = {}
for _first, _second in NEAR:
    _NEAR.setdefault(_first, set()).add(_second)
    _NEAR.setdefault(_second, set()).add(_first)


def split(accent):
    """``(strength, rest)`` of an accent or cue: "heavy Glaswegian accent" -> ("heavy", "Glaswegian")."""
    words = " ".join(str(accent or "").split()).split(" ")
    if words and words[-1].casefold() == "accent":
        words = words[:-1]
    strength = ""
    if len(words) > 1 and words[0].casefold() in STRENGTH:
        strength, words = words[0].casefold(), words[1:]
    return strength, " ".join(words)


def listed(accent):
    """The catalog name an accent or cue names ("Broad glasgow accent" -> "Glaswegian"), or None."""
    _, rest = split(accent)
    return _BY_KEY.get(re.sub(r"\s*-\s*", "-", rest).casefold())


def canonical(cue):
    """A cue in the catalog's own words ("Heavy russian accent" -> "heavy Russian accent"); unlisted cues unchanged."""
    name = listed(cue)
    if name is None:
        return cue
    strength, _ = split(cue)
    return " ".join(part for part in (strength, name, "accent") if part)


def near(name):
    """Accents that sound close to ``name`` (a catalog name): its family first, then ``NEAR`` links."""
    if name not in ACCENTS:
        return []
    family = ACCENTS[name]["family"]
    same = [other for other, entry in ACCENTS.items() if entry["family"] == family and other != name]
    return same + sorted(_NEAR.get(name, ()), key=list(ACCENTS).index)


def relation(first, second):
    """How two accents or cues compare: "same" (one accent), "near" (similar: one family or linked), or None."""
    a, b = listed(first), listed(second)
    if a is None or b is None:
        same = split(first)[1].casefold() == split(second)[1].casefold() and bool(split(first)[1])
        return "same" if same else None
    if a == b:
        return "same"
    if ACCENTS[a]["family"] == ACCENTS[b]["family"] or b in _NEAR.get(a, ()):
        return "near"
    return None


def prompt_block():
    """The list as the model sees it: one line per family, rare accents starred, cross-family links last."""
    lines = [f"{family}: {', '.join(accents)}" for family, accents in FAMILIES]
    lines.append("Close across families: " + "; ".join(f"{a} ≈ {b}" for a, b in NEAR))
    return "\n".join(lines)
