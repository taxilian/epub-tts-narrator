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

Regular variation in English speech is not a misreading: front vowels merge
before an R ("Mary", "marry", "merry"), as do the back vowels ("lore",
"lower"); an L ending a syllable may be swallowed and colours the vowel before
it ("Pulgara" for P AH1 L G ...); an R or L inside a consonant cluster may be
swallowed; two neighbouring vowels may glide into one ("TAH-or" heard as one
syllable); a consonant at the name's very edge, where word timings clip the
audio, costs less. The recognizer's vocabulary also holds tokens of other
languages: tone digits, stray periods and aspiration marks are dropped, an
r-coloured vowel is a vowel and an R, and a syllabic L is a vowel and an L
(``_normalize``).

The model is optional: ``[verification] name_phoneme_model`` names its
directory (``configure``, from the renderer CLI); unset or missing,
``available()`` is false and nothing here runs. It is loaded once per process,
on first use, and runs on Apple's GPU when present.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading

VERSION = 2
MODEL_DIRECTORY = None  # set by configure()
MISREAD_COST = 1.0
PAD_SECONDS = 0.12
_LOCK = threading.Lock()
_MODEL = None

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
    "ER": {"ɜː", "ɜ", "ɚ", "œ", "ə", "ʌ", "ɔ", "ɛ", "ø", "øː", "ɐ"},
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
_BEFORE_DARK_L = {"u", "uː", "ʊ", "o", "oː", "oʊ", "ʉ", "ɔ", "ɔː", "ʌ", "ə", "ɐ", "ɑ", "ɑː", "a"}
_ASPIRATED = {"t", "k", "p", "tɕ", "ts", "ʈ"}


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


def _is_vowel(phone):
    return phone in _IPA_VOWELS or any(phone.startswith(vowel) for vowel in ("ə", "ɚ", "ɜ"))


def _normalize(phones):
    """The recognizer's tokens as English phones (see the module notes)."""
    import re
    out = []
    for phone in phones:
        phone = re.sub(r"[0-9.]+$", "", phone)
        if not phone:
            continue
        if len(phone) > 1 and phone.endswith("h") and phone[:-1] in _ASPIRATED:
            phone = phone[:-1]
        if len(phone) > 1 and phone.endswith("ɹ") and _is_vowel(phone[:-1]):
            out += [phone[:-1], "ɹ"]
        elif phone in ("əl", "ɫ̩"):
            out += ["ə", "l"]
        else:
            out.append(phone)
    # A consonant written twice in a row (an r-coloured vowel before its R) is one sound.
    return [phone for index, phone in enumerate(out)
            if index == 0 or phone != out[index - 1] or _is_vowel(phone)]


def cost(reading, phones):
    """The cheapest alignment of ``reading`` (CMU) inside ``phones`` (eSpeak), free at both ends of ``phones``."""
    targets = _targets(reading)
    if not targets:
        return 0.0
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
            return 1.0 if stressed else 0.4
        if syllable_final_l(index):
            return 0.3
        if symbol in ("R", "L") and "consonant" in (kind(index - 1), kind(index + 1)):
            return 0.5
        return 0.5 if index in (0, count - 1) else 1.0

    def extra(phone):  # a heard sound the reading lacks, inside the name
        return 0.3 if _is_vowel(phone) else 1.0

    def glides(index, phone):  # two reading vowels heard as one gliding vowel
        if index < 1 or kind(index) != "vowel" or kind(index - 1) != "vowel" or len(phone) < 2:
            return False
        _, symbol, stressed = targets[index - 1]
        return _is_vowel(phone[:1]) and (not stressed or phone[:1] in _KINDS[symbol] or phone[:2] in _KINDS[symbol])
    rows, cols = count, len(phones)
    big = float("inf")
    # table[i][j]: cost of aligning targets[:i] ending at phones[:j]; a free start anywhere in phones.
    table = [[0.0] * (cols + 1)] + [[big] * (cols + 1) for _ in range(rows)]
    for i in range(1, rows + 1):
        table[i][0] = table[i - 1][0] + drop(i - 1)
        for j in range(1, cols + 1):
            best = min(table[i - 1][j - 1] + substitute(i - 1, phones[j - 1]),
                       table[i - 1][j] + drop(i - 1),
                       table[i][j - 1] + extra(phones[j - 1]))
            if glides(i - 1, phones[j - 1]):
                best = min(best, table[i - 2][j - 1])
            table[i][j] = best
    return round(min(table[rows]), 2)


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


def phones(path, start=None, end=None):
    """eSpeak phonemes heard in ``path`` (a WAV or its stored FLAC), optionally between two times."""
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
    if start is not None:
        data = data[max(0, int((start - PAD_SECONDS) * rate)):int((end + PAD_SECONDS) * rate)]
    if not len(data):
        return []
    factor = gcd(16000, rate)
    clip = resample_poly(data, 16000 // factor, rate // factor).astype(np.float32)
    extractor, model, vocab, device = _model()
    with _LOCK:
        inputs = extractor(clip, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = torch.argmax(model(**inputs).logits, dim=-1)[0].tolist()
    heard, last = [], None
    for index in ids:
        token = vocab.get(index)
        if index != last and token not in (None, "<pad>", "<s>", "</s>", "<unk>"):
            heard.append(token)
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


def verdicts(path, report, performance):
    """``[{"text", "key", "heard", "cost", "ok"}]`` for every chosen reading in a take."""
    from .name_shape import _readings, reading_key
    expected_text = report.get("expected_text") or ""
    results, whole = [], None
    for control, phonemes in _readings(performance):
        reading = control["phonemes"].strip()
        best = None
        for span in _spans(report, expected_text, control):
            heard = phones(path, *span)
            spent = min(cost(reading, heard), cost(phonemes, heard))  # a possessive with or without its "s"
            if best is None or spent < best[0]:
                best = (spent, heard)
            if spent < MISREAD_COST:
                break
        if best is None:  # no word timings place it: search the whole take
            if whole is None:
                whole = phones(path)
            best = (min(cost(reading, whole), cost(phonemes, whole)), whole)
        spent, heard = best
        results.append({"text": control["text"], "key": reading_key(phonemes), "heard": " ".join(heard),
                        "cost": round(spent, 2), "ok": spent < MISREAD_COST})
    return results


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


def judge(record, directory):
    """Verdicts for an accepted or candidate take: saved ones, else heard from its audio; ``None`` when unknown."""
    report = record.get("report") or {}
    words = saved(report)
    if words is not None:
        return words
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
    return verdicts(path, report, performance)


def describe(words):
    """``Marendra heard as m ɑ ɹ n ɛ d ɹ ə`` for the readings not said as chosen."""
    return "; ".join(f"{word['text']} heard as {word['heard']}" for word in words or [] if not word.get("ok"))
