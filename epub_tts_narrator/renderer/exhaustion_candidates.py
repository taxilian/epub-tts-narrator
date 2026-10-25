from array import array
from copy import deepcopy
from difflib import SequenceMatcher
import math
from pathlib import Path
import re
import sys
import wave

from .common import digest, file_hash, read_json, regular_file
from .clip_repair_contract import logical_text


def same_span(request, candidate):
    return ({key: value for key, value in request.items() if key != 'recovery'}
            == {key: value for key, value in candidate.items() if key != 'recovery'})


def physical_audio(path):
    try:
        with wave.open(str(regular_file(path)), 'rb') as audio:
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


def candidate_summary(request, record, physical):
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
            'actual_inputs': deepcopy(record.get('engine_metadata', {})), 'physical': physical,
            'warnings': {'transcript_available': bool(heard), 'word_counts': [len(expected_words), len(heard_words)],
                         'rough_syllables': [len(re.findall(r'[aeiouy]+', text.lower())) for text in (expected, heard)],
                         'difference_clusters': clusters,
                         'numbers': [re.findall(r'\d+', text) for text in (expected, heard)],
                         'negation': [[word for word in words if word in {'not', 'no', 'never', 'cannot', "n't"}]
                                      for words in (expected_words, heard_words)]}}


def retained_candidates(directory, request):
    directory = Path(directory)
    folders = {directory}
    # Sibling join directories belong only if their complete original request matches.
    if directory.parent.name == 'clips':
        folders.update(path.parent for path in directory.parent.glob('*/attempt-*.json'))
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
            if not audio.exists() and not audio.is_symlink():
                continue
            if file_hash(regular_file(audio)) != record.get('audio_sha256'):
                raise ValueError('Retained candidate audio changed')
            from .exhaustion_selection import validate_record
            validate_record(record)
            physical = physical_audio(audio)
            if physical is not None:
                candidates.append({'audio': audio, 'record': record,
                                   'summary': candidate_summary(request, record, physical)})
    return sorted(candidates, key=lambda item: (item['record']['attempt'], item['summary']['candidate_id']))
