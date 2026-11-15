from array import array
from copy import deepcopy
from difflib import SequenceMatcher
import math
from pathlib import Path
import re
from threading import Lock
import sys
import wave

from .common import digest, read_json, regular_file
from .stored_audio import audio_hash, audio_present
from .clip_repair_contract import logical_text


def same_span(request, candidate):
    return ({key: value for key, value in request.items() if key != 'recovery'}
            == {key: value for key, value in candidate.items() if key != 'recovery'})


def physical_audio(path):
    try:
        from .stored_audio import open_wave, regular_audio
        with open_wave(regular_audio(path)) as audio:
            channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
            if channels != 1 or width != 2 or rate <= 0 or frames <= 0 or audio.getcomptype() != 'NONE':
                return None
            peak, active, square, read = 0, 0, 0, 0
            while raw := audio.readframes(32768):
                samples = array('h', raw)
                if sys.byteorder != 'little':
                    samples.byteswap()
                read += len(samples)
                peak = max(peak, max(abs(value) for value in samples))
                active += sum(abs(value) >= 16 for value in samples)
                square += sum(value * value for value in samples)
            if read != frames or peak < 32 or active < max(1, rate // 200):
                return None
            return {'duration_seconds': frames / rate, 'peak_dbfs': 20 * math.log10(peak / 32768),
                    'rms_dbfs': 10 * math.log10(square / frames / 32768 ** 2)}
    except (OSError, ValueError, wave.Error, EOFError):
        return None


def stylized_spelling_advisory(text):
    """Chooser evidence for deliberately elongated or stylized spellings.

    A word carrying an alphabetic run of three or more equal letters (all-caps
    Roman numerals excluded) is an elongated stylization such as ``Wheerrre``;
    a candidate transcript in its fluent collapsed form may be the intended
    vocal realization. Returns None for ordinary text.
    """
    words = []
    for word in re.findall(r'[A-Za-z]+', text or ''):
        if word.isupper() and set(word) <= set('IVXLCDM'):
            continue
        if any(letter * 3 in word.lower() for letter in set(word.lower())):
            words.append(word)
    if not words:
        return None
    return {'elongated_words': words[:20],
            'guidance': 'Deliberately elongated or stylized spelling; a candidate transcript in its '
                        'fluent collapsed form may be the intended vocal realization.'}


def _compact_inputs(value):
    """A take's generation record with every nested review audit replaced by its digest.

    A take made by a clip repair records that repair's audit, whose request
    holds the earlier failed takes' own records, and so on: repeated repairs
    of one passage nested into a multi-megabyte request no model accepts.
    The digest keeps the record bound; the model never needed the audits.
    """
    if isinstance(value, dict):
        return {key: ({'audit_sha256': digest(item)} if key == 'audit' and isinstance(item, dict)
                      else _compact_inputs(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_inputs(item) for item in value]
    return deepcopy(value)


def candidate_summary(request, record, physical, *, compact=True):
    """A failed take as the chooser and repair planner see it (``compact=False``: the form saved before 2026-11-15)."""
    expected, heard = logical_text(request), record.get('report', {}).get('transcript') or ''
    expected_words, heard_words = re.findall(r'\w+', expected.lower()), re.findall(r'\w+', heard.lower())
    clusters = [{'kind': kind, 'expected': expected_words[a:b], 'heard': heard_words[c:d]}
                for kind, a, b, c, d in SequenceMatcher(None, expected_words, heard_words).get_opcodes()
                if kind != 'equal']
    return {'candidate_id': digest({'request': record['request_hash'], 'attempt': record['attempt'],
                                    'audio': record['audio_sha256']}),
            'attempt': record['attempt'], 'audio_sha256': record['audio_sha256'],
            'record_sha256': digest(record), 'request_sha256': record['request_hash'],
            'transcript': heard, 'original_report': deepcopy(record.get('report', {})),
            'actual_inputs': (_compact_inputs if compact else deepcopy)(record.get('engine_metadata', {})),
            'physical': physical,
            'warnings': {'transcript_available': bool(heard), 'word_counts': [len(expected_words), len(heard_words)],
                         'rough_syllables': [len(re.findall(r'[aeiouy]+', text.lower())) for text in (expected, heard)],
                         'difference_clusters': clusters,
                         'numbers': [re.findall(r'\d+', text) for text in (expected, heard)],
                         'negation': [[word for word in words if word in {'not', 'no', 'never', 'cannot', "n't"}]
                                      for words in (expected_words, heard_words)]}}


_SPANS = {}
_SPANS_LOCK = Lock()


def _span_key(request):
    """Equal for two requests exactly when ``same_span`` holds."""
    return digest({key: value for key, value in request.items() if key != 'recovery'})


def _folder_spans(folder):
    """The span keys of a take folder's attempt requests, reread only after the folder changed.

    Every record write replaces its file, which changes the folder's
    modification time, so a cached answer is current while that time is.
    """
    try:
        stamp = folder.stat().st_mtime_ns
    except OSError:
        return frozenset()
    with _SPANS_LOCK:
        cached = _SPANS.get(str(folder))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    spans = set()
    for path in folder.glob('attempt-*.json'):
        if not re.fullmatch(r'attempt-\d+\.json', path.name):
            continue
        try:
            record = read_json(path)
        except (ValueError, OSError):
            continue
        actual = record.get('request') if isinstance(record, dict) else None
        if isinstance(actual, dict):
            spans.add(_span_key(actual))
    with _SPANS_LOCK:
        _SPANS[str(folder)] = (stamp, frozenset(spans))
    return frozenset(spans)


def retained_candidates(directory, request):
    directory = Path(directory)
    folders = {directory}
    # Sibling join directories belong only if their complete original request
    # matches. A book holds tens of thousands of take folders, so which request
    # each holds is indexed once and refreshed per changed folder, instead of
    # every record of the book being read for each passage's recovery.
    if directory.parent.name == 'clips':
        wanted = _span_key(request)
        folders.update(folder for folder in directory.parent.iterdir()
                       if folder.is_dir() and not folder.is_symlink() and wanted in _folder_spans(folder))
    candidates = []
    for folder in sorted(folders):
        if folder.is_symlink():
            continue
        for path in sorted(folder.glob('attempt-*.json')):
            if not re.fullmatch(r'attempt-\d+\.json', path.name):
                continue
            record = read_json(regular_file(path))
            actual = record.get('request')
            if not isinstance(actual, dict) or not same_span(request, actual):
                continue
            if record.get('request_hash') != digest(actual):
                raise ValueError('Retained full-clip request binding changed')
            name = record.get('audio_file')
            if name is None:
                continue
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError('Invalid retained candidate audio filename')
            audio = folder / name
            if not audio_present(audio):
                continue
            if audio_hash(audio) != record.get('audio_sha256'):
                raise ValueError('Retained candidate audio changed')
            from .exhaustion_selection import validate_record
            validate_record(record)
            physical = physical_audio(audio)
            if physical is not None:
                candidates.append({'audio': audio, 'record': record,
                                   'summary': candidate_summary(request, record, physical)})
    return sorted(candidates, key=lambda item: (item['record']['attempt'], item['summary']['candidate_id']))
