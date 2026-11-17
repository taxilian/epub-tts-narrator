"""Hear whether a person-chosen pronunciation was said as chosen, by its phonemes.

A speech recognizer writes words, so a chosen name it spells differently
("Durnick", "Beric") may still be said right, and one it spells oddly may be
said wrong; ``name_shape`` can only catch consonants moved across a vowel.
This module listens to the name itself: a local phoneme recognizer (wav2vec2
fine-tuned on eSpeak phonemes, ``MODEL_DIRECTORY``) transcribes the name's few
hundred milliseconds (found from the take's word timings, else the whole take
is searched), and the result is aligned with the chosen CMU reading.

The reading is said as chosen when the best alignment costs less than
``MISREAD_COST``: consonants must be the reading's, in its order (each CMU
consonant accepts its usual eSpeak realisations, ``_CONSONANTS``; its voicing
partner costs a little, ``_VOICING``); a stressed vowel must be of the
reading's vowel kind (``_KINDS``: "eh" is not "ay" or "ee"); an unstressed
vowel may be any vowel. A stressed vowel missing or of another kind is a
misreading on its own, as is a consonant missing from the middle of the name.
Sounds around the name (the neighbouring words in the clip) cost nothing.
When the name's own timed words do not hold it, the window widens by a word
on each side (word timings can clip a name), but a match there counts only
when most of the sounds it matched were heard while the name was said or
where no other word was timed (``on_the_name``): the neighbours never stand
in for the name, and a late word timing does not hide it.

Regular variation in English speech is not a misreading: front vowels merge
before an R ("Mary", "marry", "merry"), as do the back vowels ("lore",
"lower"); an L ending a syllable may be swallowed and colours the vowel before
it ("Pulgara" for P AH1 L G ...); an R or L inside a consonant cluster may be
swallowed; two neighbouring vowels may glide into one ("TAH-or" heard as one
syllable); an ER vowel may be said without its R ("DAH-nik") or written with
it after the vowel; a nasal beside a vowel or a breath the reading lacks costs
little; a consonant at the name's very edge, where word timings clip the
audio, costs less. A name the recognizer heard in another language's sounds
(tone marks, symbols outside English) is uncertain, never a misreading. These
rules were set against a listener's verdicts. The recognizer's vocabulary also holds tokens of other
languages: tone digits, stray periods and aspiration marks are dropped, an
r-coloured vowel is a vowel and an R, and a syllabic L is a vowel and an L
(``_normalize``).

A person's ear decides over the model's: when they mark a take good
(``take_feedback``) that this check heard saying a name differently, the
exact phones heard for that name (``core``) count as said right for that
reading in that book from then on (``confirmed``). Only that exact sound is
accepted; a take heard saying it any other way is judged as before.

Verdicts heard from a saved take's audio (``judge``) are kept in its render
work's ``name-phoneme-verdicts.jsonl`` (one appended line per take), keyed by the take's audio, request and
word timings and this module's ``VERSION``, so a sweep of a book's takes
that is interrupted resumes where it stopped and history choices never hear
the same take twice.

The model is optional: ``[verification] name_phoneme_model`` names its
directory (``configure``, from the renderer CLI); unset or missing,
``available()`` is false and nothing here runs. It is loaded once per process,
on first use, and runs on Apple's GPU when present.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading

# Verdicts' identity: saved verdicts and the verdict store are read only
# under this version, so any change to how a name is heard bumps it.
VERSION = 6
# Bumped only when the check may fail a take it passed before: it names the
# check in validation keys and in the sweep of takes in use, so a change that
# only accepts more never re-verifies or re-sweeps a book. (3: an R written
# twice is one sound. 4: ER without its R, a nasal or breath beside a vowel,
# another language's sounds are uncertain. Strictness 3: a secondary-stressed
# vowel may not cross to the other side of the mouth. Strictness 4: the wider
# window counts only a match heard mostly on the name, not on other words.)
STRICTNESS = 4
MODEL_DIRECTORY = None  # set by configure()
MISREAD_COST = 1.0
PAD_SECONDS = 0.12
_LOCK = threading.Lock()
_MODEL = None
STORE_FILENAME = "name-phoneme-verdicts.jsonl"
_STORE_EVERY = 25  # new verdicts between writes of the store
_STORES = {}
_STORE_LOCK = threading.Lock()

_CONSONANTS = {
    "B": {"b", "β"}, "CH": {"tʃ", "ʃ"}, "D": {"d", "ɾ", "dʒ", "ð"}, "DH": {"ð", "d", "θ"},
    "F": {"f", "v"}, "G": {"ɡ", "ɣ", "k"}, "HH": {"h", "x"}, "JH": {"dʒ", "ʒ", "tʃ"},
    "K": {"k", "ɡ", "x"}, "L": {"l", "əl", "ɫ"}, "M": {"m", "n"}, "N": {"n", "ŋ", "m", "ɲ"},
    "NG": {"ŋ", "n"}, "P": {"p", "b"}, "R": {"ɹ", "r", "ɾ", "ʁ", "ɚ"}, "S": {"s", "z", "ts"},
    "SH": {"ʃ", "ʒ", "ɕ"}, "T": {"t", "ɾ", "d", "ts"}, "TH": {"θ", "ð", "f"}, "V": {"v", "f", "β", "w"},
    "W": {"w", "ʋ", "v", "u"}, "Y": {"j", "i"}, "Z": {"z", "s"}, "ZH": {"ʒ", "ʃ", "dʒ"},
}
# Vowel kinds a stressed vowel must keep. A recognizer's eSpeak vowels vary
# with accent, so each kind is broad; what it separates is the difference a
# listener hears as another vowel ("eh" against "ay", "ee", "ah" or "oh").
_FRONT_MID = {"ɛ", "æ", "e", "eː", "ɛ̃", "ɪ"}
_KINDS = {
    "AA": {"ɑ", "ɑː", "a", "aː", "ɔ", "ɒ", "ʌ", "ɐ", "ɑ̃"},
    "AE": _FRONT_MID | {"a", "aː"},
    "AH": {"ʌ", "ə", "ɐ", "ɑ", "a", "ɔ", "ʊ", "oː", "o", "ɚ", "ɜ"},
    "AO": {"ɔ", "ɔː", "ɑ", "ɑː", "o", "oː", "ɒ", "ʌ"},
    "AW": {"aʊ", "a", "ɑ"},
    "AY": {"aɪ", "a", "ɑ"},
    "EH": _FRONT_MID,
    # "ER" also as said without its R ("DAH-nik", "DOH-nik" for D ER1 N IH0 K).
    "ER": {"ɜː", "ɜ", "ɚ", "œ", "ə", "ʌ", "ɔ", "ɛ", "ø", "øː", "ɐ", "a", "aː", "ɑ", "ɑː", "o", "oː"},
    "EY": {"eɪ", "e", "eː", "ɛ"},
    "IH": {"ɪ", "i", "iː", "ᵻ", "e", "ɛ", "ə"},
    "IY": {"i", "iː", "ɪ", "j"},
    "OW": {"oʊ", "o", "oː", "ɔ", "əʊ"},
    "OY": {"ɔɪ", "ɔ", "oɪ"},
    "UH": {"ʊ", "u", "uː", "ə"},
    "UW": {"u", "uː", "ʊ", "ʉ"},
}
_IPA_VOWELS = set().union(*_KINDS.values()) | {"y", "yː", "ɨ", "ɨː", "ʏ", "ɐ", "uɨ", "aɨ"}
# A consonant's voicing partner: often misheard, rarely what a listener notices.
_VOICING = {"D": {"t", "ts"}, "B": {"p"}, "G": {"k"}, "Z": {"s"}, "V": {"f"}, "TH": {"s", "t"}, "DH": {"z", "t"}}
# Vowels that merge before an R in English, front and back.
_BEFORE_R = {"EH": "front", "AE": "front", "EY": "front", "AO": "back", "OW": "back"}
_MERGED_BEFORE_R = {"front": {"ɛ", "æ", "a", "e", "eɪ", "ɛː", "eː", "aː"},
                    "back": {"ɔ", "ɔː", "o", "oː", "oʊ", "əʊ"}}
# A vowel before a syllable-final L, coloured by it.
_BEFORE_DARK_L = {"u", "uː", "ʊ", "o", "oː", "oʊ", "ʉ", "ɔ", "ɔː", "ʌ", "ə", "ɐ", "ɑ", "ɑː", "a", "aʊ", "əʊ"}
# Heard consonants the reading lacks that are rarely a misreading: a nasal
# beside a vowel (a nasalised vowel written as vowel + n) and breath.
_WEAK_EXTRA = {"n": 0.5, "m": 0.5, "ŋ": 0.5, "h": 0.3}
_ASPIRATED = {"t", "k", "p", "tɕ", "ts", "ʈ"}
_R_SYMBOLS = {"r", "ʁ", "ɻ"}


def _targets(reading):
    """``[(kind, base, stressed)]`` for a CMU reading (boundary marks ignored)."""
    from ..performance_plan import phoneme_symbols
    out = []
    for symbol in phoneme_symbols(reading):
        base = symbol.rstrip("012")
        if base in _KINDS:
            out.append(("vowel", base, symbol.endswith("1")))
        elif base in _CONSONANTS:
            out.append(("consonant", base, False))
    return out


_FRONT_KINDS = frozenset({"IY", "IH", "EY", "EH", "AE"})
_BACK_KINDS = frozenset({"UW", "UH", "OW", "AO", "AA", "OY", "AW"})
_HEARD_FRONT = frozenset({"i", "iː", "ɪ", "e", "eː", "eɪ", "ɛ", "ɛː", "æ", "y", "yː", "ʏ"})
_HEARD_BACK = frozenset({"u", "uː", "ʊ", "ʉ", "o", "oː", "oʊ", "əʊ", "ɔ", "ɔː", "ɒ", "ʌ", "ɚ", "ɜ", "ɜː"})


def _crosses(kind, phone):
    """Whether a heard vowel is on the other side of the mouth from a reading's vowel kind."""
    return ((kind in _FRONT_KINDS and phone in _HEARD_BACK) or (kind in _BACK_KINDS and phone in _HEARD_FRONT))


def _is_vowel(phone):
    return phone in _IPA_VOWELS or any(phone.startswith(vowel) for vowel in ("ə", "ɚ", "ɜ"))


def _normalize(phones):
    """The recognizer's tokens as English phones (see the module notes)."""
    return [phone for phone, _ in _normalize_sources(phones)]


def _normalize_sources(phones):
    """``_normalize`` with, for each English phone, the index of the recognizer token it came from."""
    import re
    out = []
    for source, phone in enumerate(phones):
        phone = re.sub(r"[0-9.]+$", "", phone)
        if not phone:
            continue
        if len(phone) > 1 and phone.endswith("h") and phone[:-1] in _ASPIRATED:
            phone = phone[:-1]
        if phone in _R_SYMBOLS:
            phone = "ɹ"  # one R sound however it is written
        if len(phone) > 1 and phone.endswith("ɹ") and _is_vowel(phone[:-1]):
            out += [(phone[:-1], source), ("ɹ", source)]
        elif phone in ("əl", "ɫ̩"):
            out += [("ə", source), ("l", source)]
        else:
            out.append((phone, source))
    # A consonant written twice in a row (an r-coloured vowel before its R) is one sound.
    return [item for index, item in enumerate(out)
            if index == 0 or item[0] != out[index - 1][0] or _is_vowel(item[0])]


def on_the_name(reading, timed, start, end, others=()):
    """Whether the cheapest alignment of ``reading`` in ``timed`` phones was the name's own sounds.

    A wider window around the name holds the neighbouring words too, and the
    alignment is free to find the reading anywhere in it: "city of RAY-vuh"
    matched R IY V AH in "-ty of" (ɾ i ʌ v), and "army TOH" matched T AA AO R
    in "ar-". A matched sound is the name's when it was heard while the name
    was said (between ``start`` and ``end``, padded as when it is heard alone)
    or where no other word was timed (``others``: their ``(start, end)``):
    word timings can run late, leaving the name in the gap before them
    ("Belgarath" heard 0.4 s before its timing). More than half of the sounds
    it matched must be the name's.
    """
    if not timed:
        return False
    phones = [phone for phone, _ in timed]
    _, first, last = _align(reading, phones)
    sources = [source for _, source in _normalize_sources(phones)][first:last]
    if not sources:
        return False

    def the_names(when):
        return (start - PAD_SECONDS <= when <= end + PAD_SECONDS
                or not any(low <= when <= high for low, high in others))
    inside = sum(1 for source in sources if the_names(timed[source][1]))
    return inside * 2 > len(sources)


def cost(reading, phones):
    """The cheapest alignment of ``reading`` (CMU) inside ``phones`` (eSpeak), free at both ends of ``phones``."""
    return _align(reading, phones)[0]


def core(reading, phones):
    """The heard phones the cheapest alignment of ``reading`` covers, as a tuple (the name as heard)."""
    _, start, end = _align(reading, phones)
    return tuple(_normalize(phones)[start:end])


def _align(reading, phones):
    """``(cost, start, end)``: the cheapest alignment and the stretch of normalized ``phones`` it covers."""
    targets = _targets(reading)
    if not targets:
        return 0.0, 0, 0
    from ..performance_plan import phoneme_symbols
    secondary = [symbol.endswith("2") for symbol in phoneme_symbols(reading)
                 if symbol.rstrip("012") in _KINDS or symbol.rstrip("012") in _CONSONANTS]
    phones = _normalize(phones)
    count = len(targets)

    def kind(index):
        return targets[index][0] if 0 <= index < count else None

    def base(index):
        return targets[index][1] if 0 <= index < count else None

    def syllable_final_l(index):  # an L followed by a consonant or nothing
        return base(index) == "L" and kind(index + 1) != "vowel"

    def substitute(index, phone):
        sort, symbol, stressed = targets[index]
        if sort == "consonant":
            if phone in _CONSONANTS[symbol]:
                return 0.0
            return 0.3 if phone in _VOICING.get(symbol, ()) else 1.0
        if not _is_vowel(phone):
            return 1.0
        if secondary[index] and phone not in _KINDS[symbol] and _crosses(symbol, phone):
            # A secondary stress keeps its side of the mouth: "seh-" may reduce
            # to "sih-"/"suh-" (ə) but not become "soo-" or "sur-".
            return 1.0
        if not stressed or phone in _KINDS[symbol]:
            return 0.0
        if syllable_final_l(index + 1) and phone in _BEFORE_DARK_L:
            return 0.0
        if base(index + 1) == "R" and symbol in _BEFORE_R and phone in _MERGED_BEFORE_R[_BEFORE_R[symbol]]:
            return 0.0
        return 1.0

    def drop(index):  # a reading sound not heard
        sort, symbol, stressed = targets[index]
        if sort == "vowel":
            # A secondary stress is a syllable a listener hears: dropping it
            # and hearing another vowel instead is not cheaper than swapping.
            return 1.0 if stressed else 0.7 if secondary[index] else 0.4
        if syllable_final_l(index):
            return 0.3
        if symbol in ("R", "L") and "consonant" in (kind(index - 1), kind(index + 1)):
            return 0.5
        return 0.5 if index in (0, count - 1) else 1.0

    def extra(phone):  # a heard sound the reading lacks, inside the name
        return 0.3 if _is_vowel(phone) else _WEAK_EXTRA.get(phone, 1.0)

    def glides(index, phone):  # two reading vowels heard as one gliding vowel
        if index < 1 or kind(index) != "vowel" or kind(index - 1) != "vowel" or len(phone) < 2:
            return False
        _, symbol, stressed = targets[index - 1]
        return _is_vowel(phone[:1]) and (not stressed or phone[:1] in _KINDS[symbol] or phone[:2] in _KINDS[symbol])
    rows, cols = count, len(phones)
    big = float("inf")
    # table[i][j]: (cost, start) of aligning targets[:i] ending at phones[:j];
    # a free start anywhere in phones, remembered to name the stretch heard.
    table = [[(0.0, j) for j in range(cols + 1)]] + [[(big, 0)] * (cols + 1) for _ in range(rows)]
    for i in range(1, rows + 1):
        table[i][0] = (table[i - 1][0][0] + drop(i - 1), 0)
        for j in range(1, cols + 1):
            options = [(table[i - 1][j - 1][0] + substitute(i - 1, phones[j - 1]), table[i - 1][j - 1][1]),
                       (table[i - 1][j][0] + drop(i - 1), table[i - 1][j][1]),
                       (table[i][j - 1][0] + extra(phones[j - 1]), table[i][j - 1][1])]
            if glides(i - 1, phones[j - 1]):
                options.append(table[i - 2][j - 1])
            if phones[j - 1] == "ɹ" and base(i - 1) == "ER" and j >= 2:
                # The R of an r-coloured ER vowel, written after it ("d ɔ ɹ n").
                options.append(table[i][j - 1])
            table[i][j] = min(options, key=lambda option: option[0])
    end = min(range(cols + 1), key=lambda j: table[rows][j][0])
    spent, start = table[rows][end]
    return round(spent, 2), start, end


def configure(directory):
    """Use the phoneme model in ``directory`` (``None``: no phoneme check)."""
    global MODEL_DIRECTORY, _MODEL
    path = Path(directory).expanduser().resolve() if directory else None
    with _LOCK:
        if path != MODEL_DIRECTORY:
            MODEL_DIRECTORY, _MODEL = path, None


def available():
    """Whether a phoneme model is configured and present (it is optional)."""
    return (MODEL_DIRECTORY is not None and (MODEL_DIRECTORY / "config.json").is_file()
            and (MODEL_DIRECTORY / "vocab.json").is_file())


def _model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_DIRECTORY))
            model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIRECTORY)).eval().to(device)
            vocab = {index: token for token, index in json.loads((MODEL_DIRECTORY / "vocab.json").read_text()).items()}
            _MODEL = (extractor, model, vocab, device)
        return _MODEL


def phones(path, start=None, end=None, *, timed=False):
    """eSpeak phonemes heard in ``path`` (a WAV or its stored FLAC), optionally between two times.

    With ``timed`` each is ``(phone, seconds into the take)``: when the
    recognizer emitted it.
    """
    from math import gcd
    import numpy as np
    import soundfile
    import torch
    from scipy.signal import resample_poly
    from .stored_audio import materialize
    with materialize(Path(path)) as wav:
        data, rate = soundfile.read(str(wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    offset = 0.0
    if start is not None:
        first = max(0, int((start - PAD_SECONDS) * rate))
        offset = first / rate
        data = data[first:int((end + PAD_SECONDS) * rate)]
    if not len(data):
        return []
    seconds = len(data) / rate
    factor = gcd(16000, rate)
    clip = resample_poly(data, 16000 // factor, rate // factor).astype(np.float32)
    extractor, model, vocab, device = _model()
    with _LOCK:
        inputs = extractor(clip, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = torch.argmax(model(**inputs).logits, dim=-1)[0].tolist()
    heard, last = [], None
    for frame, index in enumerate(ids):
        token = vocab.get(index)
        if index != last and token not in (None, "<pad>", "<s>", "</s>", "<unk>"):
            heard.append((token, round(offset + frame * seconds / len(ids), 3)) if timed else token)
        last = index
    return heard


def _spans(report, expected_text, control):
    """Seconds of the words heard in the chosen reading's place, then with a word either side; ``[]`` when unplaced.

    The wider span catches a name whose word timings clipped it; sounds
    around the name cost nothing, so it can only help a reading heard right.
    """
    from difflib import SequenceMatcher
    from .quality import normalize_words
    words = [word for word in report.get("words") or []
             if isinstance(word, dict) and isinstance(word.get("start"), (int, float))
             and isinstance(word.get("end"), (int, float)) and isinstance(word.get("word"), str)]
    if not words:
        return []
    expected = normalize_words(expected_text)
    heard = [(normalize_words(word["word"]) or [""])[0] for word in words]
    first = len(normalize_words(expected_text[:control["start"]]))
    last = first + max(1, len(normalize_words(control.get("text") or ""))) - 1
    for tag, a, b, c, d in SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes():
        if a <= first and last < b and tag in ("equal", "replace") and b - a == d - c:
            low, high = c + first - a, c + last - a
            wide_low, wide_high = max(0, low - 1), min(len(words) - 1, high + 1)
            return [(float(words[low]["start"]), float(words[high]["end"])),
                    (float(words[wide_low]["start"]), float(words[wide_high]["end"]))]
    return []


def _other_words(report, start, end):
    """``(start, end)`` of every timed word outside the name's own span."""
    return [(float(word["start"]), float(word["end"])) for word in report.get("words") or []
            if isinstance(word, dict) and isinstance(word.get("start"), (int, float))
            and isinstance(word.get("end"), (int, float)) and (word["end"] <= start or word["start"] >= end)]


def verdicts(path, report, performance):
    """``[{"text", "key", "heard", "cost", "ok"}]`` for every chosen reading in a take."""
    from .name_shape import _readings, reading_key
    expected_text = report.get("expected_text") or ""
    results, whole = [], None
    for control, phonemes in _readings(performance):
        reading = control["phonemes"].strip()
        best = None
        spans = _spans(report, expected_text, control)
        for index, span in enumerate(spans):
            if index == 0:
                heard = phones(path, *span)
                spent = min(cost(reading, heard), cost(phonemes, heard))  # a possessive with or without its "s"
            else:
                # The wider window counts only where it heard the name itself.
                timed = phones(path, *span, timed=True)
                heard = [phone for phone, _ in timed]
                spent = min([cost(form, heard) for form in (reading, phonemes)
                             if on_the_name(form, timed, *spans[0], others=_other_words(report, *spans[0]))]
                            or [float("inf")])
            if best is None or spent < best[0]:
                best = (spent, heard)
            if spent < MISREAD_COST:
                break
        if best is None:  # no word timings place it: search the whole take
            if whole is None:
                whole = phones(path)
            best = (min(cost(reading, whole), cost(phonemes, whole)), whole)
        spent, heard = best
        verdict = {"text": control["text"], "key": reading_key(phonemes), "heard": " ".join(heard),
                   "cost": round(spent, 2), "ok": spent < MISREAD_COST}
        if not verdict["ok"] and heard is not whole and _foreign(heard):
            # The recognizer heard the name in another language's sounds
            # ("x ai5 tɕh iou2" for "Hettar?"): it cannot tell how it was
            # said, so it is not called a misreading.
            verdict.update(ok=True, uncertain=True)
        results.append(verdict)
    return _confirm(results, _work_of(Path(path).parent))


def _foreign(heard):
    """Whether phonemes heard for a name include another language's tokens (tone marks, unknown symbols)."""
    import re
    consonants = set().union(*_CONSONANTS.values()) | {"ɹ", "h", "ʔ", "ç", "ʈ", "ɖ", "ɭ"} | _R_SYMBOLS
    for phone in heard:
        if re.search(r"[0-9]", phone):
            return True
        for token in _normalize([phone]):
            if not _is_vowel(token) and token not in consonants:
                return True
    return False


_CONFIRMED = {}


def confirmed(work):
    """``{reading key: {phones}}`` a person heard said right in this render work (takes they marked good)."""
    if work is None:
        return {}
    work = Path(work)
    path = work / "take-feedback.json"
    try:
        stamp = path.stat().st_mtime_ns if path.is_file() else None
    except OSError:
        stamp = None
    with _STORE_LOCK:
        cached = _CONFIRMED.get(work)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    found = {}
    if stamp is not None:
        from .common import read_json
        from .take_feedback import clip_folder, load
        for entry in load(work)["takes"].values():
            if entry.get("verdict") != "good" or not isinstance(entry.get("request_hash"), str):
                continue
            folder = clip_folder(work, entry["request_hash"])
            attempt = entry.get("attempt")
            if folder is None or type(attempt) is not int:
                continue
            try:
                record = read_json(folder / f"attempt-{attempt:03d}.json")
            except (ValueError, OSError):
                continue
            if record.get("audio_sha256") != entry.get("audio_sha256"):
                continue
            for word in ((record.get("report") or {}).get("name_phonemes") or {}).get("words") or []:
                if isinstance(word, dict) and isinstance(word.get("key"), str) and isinstance(word.get("heard"), str):
                    found.setdefault(word["key"], set()).add(core(word["key"], word["heard"].split()))
    with _STORE_LOCK:
        _CONFIRMED[work] = (stamp, found)
    return found


def _confirm(words, work):
    """``words`` with readings heard exactly as a person confirmed them marked said right."""
    sounds = confirmed(work)
    if not sounds or not words:
        return words
    result = []
    for word in words:
        if (isinstance(word, dict) and not word.get("ok") and word.get("key") in sounds
                and core(word["key"], str(word.get("heard", "")).split()) in sounds[word["key"]]):
            word = dict(word, ok=True, confirmed=True)
        result.append(word)
    return result


def saved(report):
    """The verdicts a report already holds under this version, or ``None``."""
    found = (report or {}).get("name_phonemes")
    if isinstance(found, dict) and found.get("version") == VERSION and isinstance(found.get("words"), list):
        return found["words"]
    return None


def apply(report, path, performance):
    """Check a passed chapter-speech report's chosen readings by their phonemes; the report.

    A reading not said as chosen fails the take as ``pronunciation_mismatch``.
    The verdicts are kept in ``report["name_phonemes"]`` either way.
    """
    from .name_shape import REASON, _readings
    if report.get("passed") is not True or not available() or not _readings(performance):
        return report
    words = verdicts(path, report, performance)
    report = {**report, "name_phonemes": {"version": VERSION, "words": words}}
    if all(word["ok"] for word in words):
        return report
    return {**report, "passed": False, "reasons": [REASON],
            "passed_method": report.get("text_confirmation_method"), "text_confirmation_method": None}


def _work_of(directory):
    """The render work holding a take folder (``WORK/clips/<hash>`` or its parts), or ``None``."""
    for parent in Path(directory).resolve().parents:
        if parent.name == "clips":
            return parent.parent
    return None


def _store(work):
    with _STORE_LOCK:
        if work not in _STORES:
            path = work / STORE_FILENAME
            entries = {}
            if path.is_file() and not path.is_symlink():
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError):
                    lines = []
                for line in lines:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # a line cut short by an interruption is heard again
                    if (isinstance(row, dict) and row.get("version") == VERSION
                            and isinstance(row.get("key"), str) and isinstance(row.get("words"), list)):
                        entries[row["key"]] = row["words"]
            if not _STORES:
                import atexit
                atexit.register(flush)  # an interrupted render keeps what it heard
            _STORES[work] = {"path": path, "entries": entries, "pending": []}
        return _STORES[work]


def _store_key(record):
    from .common import digest
    report = record.get("report") or {}
    return digest({"audio": record.get("audio_sha256"), "request": record.get("request_hash"),
                   "words": report.get("words"), "expected": report.get("expected_text")})


def flush(work=None):
    """Append kept verdicts not yet written (every render work, or one) to their store."""
    import os
    with _STORE_LOCK:
        for key, store in list(_STORES.items()):
            if (work is not None and key != Path(work).resolve()) or not store["pending"]:
                continue
            lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in store["pending"])
            with open(store["path"], "a", encoding="utf-8") as stream:
                stream.write(lines)
                stream.flush()
                os.fsync(stream.fileno())
            store["pending"] = []


def judge(record, directory):
    """Verdicts for an accepted or candidate take: saved ones, else heard from its audio; ``None`` when unknown."""
    report = record.get("report") or {}
    words = saved(report)
    if words is not None:
        return _confirm(words, _work_of(directory))
    if not available():
        return None
    from .name_shape import _readings
    from .performance_runtime import verification_options
    try:
        performance = verification_options(record.get("request") or {}).get("performance")
    except (ValueError, KeyError, TypeError):
        return None
    if not _readings(performance) or not isinstance(record.get("audio_file"), str):
        return [] if not _readings(performance) else None
    from .stored_audio import audio_present
    path = Path(directory) / record["audio_file"]
    if not audio_present(path):
        return None
    work = _work_of(directory) if isinstance(record.get("audio_sha256"), str) else None
    store = _store(work) if work is not None else None
    key = _store_key(record) if store is not None else None
    if store is not None:
        with _STORE_LOCK:
            kept = store["entries"].get(key)
        if isinstance(kept, list):
            return _confirm(kept, work)
    found = verdicts(path, report, performance)
    if store is not None:
        with _STORE_LOCK:
            store["entries"][key] = found
            store["pending"].append({"version": VERSION, "key": key, "words": found})
            due = len(store["pending"]) >= _STORE_EVERY
        if due:
            flush(work)
    return found


def describe(words):
    """``Marendra heard as m ɑ ɹ n ɛ d ɹ ə`` for the readings not said as chosen."""
    return "; ".join(f"{word['text']} heard as {word['heard']}" for word in words or [] if not word.get("ok"))
