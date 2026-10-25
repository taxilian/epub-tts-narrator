from copy import deepcopy
import os
from pathlib import Path
import tempfile
import uuid

from .assembly import normalize_clip
from .common import digest, file_hash, read_json, regular_file, write_json
from .clip_repair_contract import approved, confirm_part, logical_text
from .clip_repair_state import ClipRecoveryUnavailable
from .quality import text_confirmation_unavailable, validate_audio, validation_key
from .retry_budget import generation_attempts
from .speech_recovery import _concatenate


def execute_plan(session, generate, verifier, config, *, budget, episode, reserve):
    from .interrupts import stop_if_requested
    state, request, directory = session.state, session.request, session.path.parent
    if state['phase'] in {'failed', 'rejected'}:
        return None
    plan = approved(request, state['audit'])
    maximum = config.get('verification', {}).get('max_resolution_attempts', 12)
    if 'attempt' not in state:
        state.update(attempt=max(generation_attempts(directory), default=0) + 1, phase='executing')
        state['parts'] = [{'part': part, 'starts': 0} for part in plan['parts']]
        session.save()
    attempt = state['attempt']
    candidate_path = directory / f'attempt-{attempt:03d}.json'
    target = directory / f'attempt-{attempt:03d}.wav'
    key = validation_key(logical_text(request), verifier, config['quality'], purpose='speech')
    base = {'request': request, 'request_hash': digest(request), 'attempt': attempt,
            'seed': request.get('base_seed', config['render']['seed']), 'validation_key': key}
    if candidate_path.exists():
        previous = read_json(regular_file(candidate_path))
        if (previous.get('audio_file') == target.name and previous.get('report', {}).get('reasons')
                not in (['generation_incomplete'], ['verification_pending'])):
            from .exhaustion_selection import validate_record
            validate_record(previous)
            if (previous.get('request_hash') != digest(request)
                    or previous.get('engine_metadata', {}).get('clip_repair', {}).get('audit') != state['audit']
                    or previous.get('audio_sha256') != state.get('joined_sha256')
                    or file_hash(regular_file(target)) != previous['audio_sha256']):
                raise ValueError('Completed clip repair candidate changed')
            state['phase'] = 'evaluated'
            session.save()
            if previous['report'].get('passed') is True:
                write_json(directory / 'accepted.json', previous)
                return target, previous
            return None
    if not candidate_path.exists():
        write_json(candidate_path, {**base, 'audio_file': None, 'audio_sha256': None,
                                   'engine_metadata': {}, 'report': {'passed': False, 'reasons': ['generation_incomplete']}})
    folder = directory / 'clip-repair-parts'
    if folder.is_symlink():
        raise ValueError('Clip repair parts must not use symlinks')
    folder.mkdir(exist_ok=True)
    for number, saved in enumerate(state['parts'], 1):
        part = plan['parts'][number - 1]
        if saved['part'] != part:
            raise ValueError('Pending clip repair part changed')
        if 'audio_sha256' in saved:
            if file_hash(regular_file(folder / saved['audio_file'])) != saved['audio_sha256']:
                raise ValueError('Completed clip repair part changed')
            confirm_part(part, saved['metadata'])
            continue
        if 'raw_sha256' not in saved:
            stop_if_requested()
            claimed = budget.claim(directory, digest(request), maximum, episode=episode,
                                   episode_maximum=maximum, episode_reserve=reserve)
            if claimed is None:
                state.update(phase='failed', error_code='shared_budget_exhausted')
                session.save()
                return None
            saved['starts'] += 1
            saved['raw_file'] = f'part-{number:03d}-start-{saved["starts"]:03d}-raw.wav'
            session.save()
            seed = int(digest({'request': request, 'part': number, 'start': saved['starts']})[:8], 16) % 2**32
            try:
                metadata = generate(part, folder / saved['raw_file'], seed)
                confirm_part(part, metadata)
            except (OSError, RuntimeError, ValueError) as exc:
                state.update(phase='failed', error_code='synthesis_or_input_failure', error_type=type(exc).__name__)
                session.save()
                return None
            saved.update(metadata=metadata, raw_sha256=file_hash(regular_file(folder / saved['raw_file'])))
            session.save()
        raw = regular_file(folder / saved['raw_file'])
        if file_hash(raw) != saved['raw_sha256']:
            raise ValueError('Pending clip repair raw audio changed')
        wav = folder / f'part-{number:03d}.wav'
        with tempfile.TemporaryDirectory(prefix='.normalize-part-', dir=folder) as staging:
            normalized = Path(staging) / 'normalized.wav'
            normalize_clip(raw, normalized, sample_rate=config['render']['sample_rate'], ffmpeg=config['render']['ffmpeg'])
            if wav.exists() or wav.is_symlink():
                regular_file(wav).rename(folder / f'part-{number:03d}-orphan-{uuid.uuid4().hex}.wav')
            os.replace(normalized, wav)
        saved.update(audio_file=wav.name, audio_sha256=file_hash(regular_file(wav)))
        session.save()
    if 'joined_sha256' not in state:
        with tempfile.TemporaryDirectory(prefix='.clip-join-', dir=directory) as staging:
            joined = Path(staging) / 'joined.wav'
            _concatenate([(folder / part['audio_file'], part) for part in state['parts']],
                         joined, config['render']['sample_rate'])
            checksum = file_hash(joined)
            if target.exists() and file_hash(regular_file(target)) != checksum:
                raise ValueError('Interrupted clip repair join changed')
            os.replace(joined, target)
        state['joined_sha256'] = checksum
        session.save()
    if file_hash(regular_file(target)) != state['joined_sha256']:
        raise ValueError('Clip repair joined audio changed')
    marker = {'audit': deepcopy(state['audit']), 'parts': [
        {key: deepcopy(part[key]) for key in ('part', 'metadata', 'audio_sha256')} for part in state['parts']],
        'audio_sha256': state['joined_sha256']}
    metadata = {'synthesis_text': ' '.join(part['text'] for part in plan['parts']), 'clip_repair': marker}
    pending = {**base, 'audio_file': target.name, 'audio_sha256': state['joined_sha256'], 'engine_metadata': metadata,
               'report': {'passed': False, 'reasons': ['verification_pending']}}
    write_json(candidate_path, pending)
    report = validate_audio(target, logical_text(request), verifier, config['quality'], purpose='speech')
    if text_confirmation_unavailable(report):
        raise ClipRecoveryUnavailable('Clip repair verification unavailable; completed synthesis retained')
    record = {**pending, 'report': report}
    write_json(candidate_path, record)
    state['phase'] = 'evaluated'
    session.save()
    if report.get('passed') is True:
        write_json(directory / 'accepted.json', record)
        return target, record
    return None
