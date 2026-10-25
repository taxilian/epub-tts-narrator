from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import tempfile

from .common import digest, file_hash, read_json, regular_file, write_json
from .clip_repair_contract import logical_text, plain, synthesis_request
from .clip_repair_prompts import SELECT
from .clip_repair_state import ClipRecoveryUnavailable, model_call
from .exhaustion_candidates import candidate_summary, retained_candidates, same_span


def policy(provider):
    return {'name': 'llm_exhaustion_selection', 'version': 1, 'provider': deepcopy(provider),
            'prompt_sha256': digest(SELECT), 'eligibility': 'bound_full_span_physical_pcm16_v1',
            'selection': 'decisive_no_lexical_or_acoustic_advisory_reveto'}


def _choice_error(response, candidate_ids, pinned=None):
    if not isinstance(response, dict) or set(response) != {'candidate_id', 'reason'}:
        return 'invalid_json_shape'
    if not isinstance(response['candidate_id'], str) or response['candidate_id'] not in candidate_ids:
        return 'unknown_candidate_id'
    if pinned is not None and response['candidate_id'] != pinned:
        return 'preserve_selected_candidate'
    if not plain(response['reason'], 2000):
        return 'invalid_reason'
    return None


def _format_correction(responses, candidate_ids):
    if not isinstance(responses, list) or not responses:
        raise ValueError('Format correction requires preserved invalid responses')
    pinned = None
    for response in responses:
        error = _choice_error(response, candidate_ids, pinned)
        if error is None:
            raise ValueError('A valid chooser selection must not be reconsidered')
        if pinned is None and isinstance(response, dict) and response.get('candidate_id') in candidate_ids:
            pinned = response['candidate_id']
    return {'previous_responses': deepcopy(responses), 'allowed_candidate_ids': list(candidate_ids),
            'preserve_candidate_id': pinned, 'error': error,
            'instruction': 'Correct JSON structure only: return exactly candidate_id and a brief plain reason. '
                           'Select from the supplied candidate IDs. Preserve preserve_candidate_id when present; '
                           'do not reconsider the semantic ranking or request new audio.'}


def _correction_payload(base, responses):
    result = deepcopy(base)
    if responses:
        result['format_correction'] = _format_correction(responses, [row['candidate_id'] for row in base['candidates']])
    return result


def validate_selection(request, marker):
    if not isinstance(marker, dict) or set(marker) != {'request', 'audit', 'selected_record'}:
        raise ValueError('Invalid exhaustion selection authority')
    original, audit, selected = marker['request'], marker['audit'], marker['selected_record']
    if not isinstance(original, dict) or not isinstance(selected, dict) or not isinstance(selected.get('request'), dict):
        raise ValueError('Invalid selected request binding')
    if request != original and request != synthesis_request(original):
        raise ValueError('Exhaustion selection belongs to another source or reference')
    if (not isinstance(audit, dict) or set(audit) != {'identity', 'payload', 'response'}
            or not isinstance(audit['identity'], dict) or not isinstance(audit['identity'].get('provider'), dict)
            or not isinstance(audit['payload'], dict) or not isinstance(audit['payload'].get('candidates'), list)
            or any(not isinstance(row, dict) for row in audit['payload']['candidates'])
            or audit['identity'] != policy(audit['identity'].get('provider'))
            or audit['payload'].get('mode') != 'exhaustion_selection'
            or audit['payload'].get('request') != original
            or audit['payload'].get('request_sha256') != digest(original)
            or audit['payload'].get('requested_text') != logical_text(original)):
        raise ValueError('Exhaustion chooser policy or input changed')
    response = audit['response']
    candidate_ids = [row.get('candidate_id') for row in audit['payload']['candidates']]
    correction = audit['payload'].get('format_correction')
    if 'format_correction' in audit['payload']:
        if (not isinstance(correction, dict)
                or correction != _format_correction(correction.get('previous_responses'), candidate_ids)):
            raise ValueError('Chooser format-correction context changed')
    if _choice_error(response, candidate_ids, correction['preserve_candidate_id'] if correction else None) is not None:
        raise ValueError('Chooser must select one supplied candidate with a reason')
    matches = [row for row in audit['payload']['candidates'] if row.get('candidate_id') == response['candidate_id']]
    if (len(matches) != 1 or selected.get('request_hash') != digest(selected['request'])
            or not same_span(original, selected['request'])
            or 'exhaustion_selection' in selected.get('engine_metadata', {})):
        raise ValueError('Chooser selected an unrelated or unbound candidate')
    physical = matches[0].get('physical')
    if (not isinstance(physical, dict) or not all(type(physical.get(key)) in (int, float) and math.isfinite(physical[key])
                for key in ('duration_seconds', 'peak_dbfs', 'rms_dbfs'))
            or physical['duration_seconds'] <= 0
            or candidate_summary(original, selected, physical) != matches[0]):
        raise ValueError('Selected audio evidence changed')
    return selected


def validate_record(record):
    metadata = record.get('engine_metadata', {})
    marker = metadata.get('exhaustion_selection')
    if marker is None:
        if 'clip_repair' in metadata:
            from .clip_repair_contract import confirm_plan_metadata
            confirm_plan_metadata(record['request'], metadata)
            if record.get('audio_sha256') != metadata['clip_repair'].get('audio_sha256'):
                raise ValueError('Clip repair audio differs from its generation proof')
        return
    selected = validate_selection(record['request'], marker)
    validate_record(selected)
    if (record.get('request_hash') != digest(record['request'])
            or record.get('audio_sha256') != selected['audio_sha256']
            or record.get('validation_key') != digest({'selection': marker, 'audio': selected['audio_sha256']})
            or {key: value for key, value in metadata.items() if key != 'exhaustion_selection'}
                != selected.get('engine_metadata', {})
            or record['report'].get('text_confirmation_method') != 'llm_exhaustion_selection'
            or record['report'].get('acceptance_policy') != marker['audit']['identity']
            or record['report'].get('ordinary_verification_passed') is not (selected['report'].get('passed') is True)
            or record['report'].get('original_failed_report') != selected['report']
            or record['report'].get('expected_text') != logical_text(record['request'])
            or record['report'].get('passed') is not True or record['report'].get('reasons') != []):
        raise ValueError('Exhaustion acceptance differs from its decisive selection')


def selected_generation_identity(request, voices, metadata):
    from .speech_identity import generation_identity
    selected = validate_selection(request, metadata['exhaustion_selection'])
    if {key: value for key, value in metadata.items() if key != 'exhaustion_selection'} != selected.get('engine_metadata', {}):
        raise ValueError('Selected synthesis metadata changed')
    validate_record(selected)
    # The chooser authorizes this recorded whole performance, not the old lexical
    # policy. Replaying old rewrite/word gates here would silently revoke its vote.
    return generation_identity(request, voices)


def select_retained(directory, request, client):
    directory = Path(directory)
    candidates = retained_candidates(directory, request)
    if not candidates:
        raise ValueError('No physically usable bound full-span candidate remains')
    payload = {'mode': 'exhaustion_selection', 'request_sha256': digest(request),
               'requested_text': logical_text(request), 'request': deepcopy(request),
               'candidates': [item['summary'] for item in candidates]}
    path = directory / 'exhaustion-selection.json'
    identity = policy(client.identity)
    state = {'audit': {'identity': identity, 'payload': payload, 'response': None}, 'response_received': False}
    if path.exists() or path.is_symlink():
        saved = read_json(regular_file(path))
        state = {key: value for key, value in saved.items() if key != 'record_sha256'}
        if saved.get('record_sha256') != digest(state) or state['audit']['identity'] != identity:
            raise ValueError('Pending exhaustion selection inputs changed')
        responses = []
        for audit in state.get('invalid_responses', []):
            if audit.get('identity') != identity or audit.get('payload') != _correction_payload(payload, responses):
                raise ValueError('Saved chooser correction history changed')
            responses.append(audit['response'])
        if state['audit']['payload'] != _correction_payload(payload, responses):
            raise ValueError('Pending exhaustion selection inputs changed')
    else:
        write_json(path, {**state, 'record_sha256': digest(state)})
    calls = 0
    candidate_ids = [item['summary']['candidate_id'] for item in candidates]
    while True:
        correction = state['audit']['payload'].get('format_correction', {})
        if state['response_received']:
            if _choice_error(state['audit']['response'], candidate_ids, correction.get('preserve_candidate_id')) is None:
                break
            history = state.setdefault('invalid_responses', [])
            history.append(deepcopy(state['audit']))
            state['audit'] = {'identity': identity,
                              'payload': _correction_payload(payload, [audit['response'] for audit in history]),
                              'response': None}
            state['response_received'] = False
            write_json(path, {**state, 'record_sha256': digest(state)})
        if calls == 3:
            raise ClipRecoveryUnavailable('Chooser format corrections exhausted this invocation; pending choice retained')
        calls += 1
        state['audit']['response'] = model_call(client, SELECT, state['audit']['payload'])
        state['response_received'] = True
        write_json(path, {**state, 'record_sha256': digest(state)})
    response = state['audit']['response']
    selected = next((item for item in candidates if isinstance(response, dict)
                     and item['summary']['candidate_id'] == response.get('candidate_id')), None)
    if selected is None:
        raise ValueError('Exhaustion chooser did not select a supplied candidate')
    marker = {'request': deepcopy(request), 'audit': state['audit'], 'selected_record': selected['record']}
    validate_selection(request, marker)
    record = deepcopy(selected['record'])
    report = deepcopy(record['report'])
    record.update(request=deepcopy(request), request_hash=digest(request),
                  audio_file='selection-' + record['audio_sha256'] + '.wav',
                  validation_key=digest({'selection': marker, 'audio': record['audio_sha256']}),
                  report={**report, 'passed': True, 'reasons': [], 'expected_text': logical_text(request),
                          'text_confirmation_method': 'llm_exhaustion_selection',
                          'acceptance_policy': state['audit']['identity'],
                          'ordinary_verification_passed': report.get('passed') is True,
                          'original_failed_report': report})
    record['engine_metadata'] = {**record.get('engine_metadata', {}), 'exhaustion_selection': marker}
    validate_record(record)
    target = directory / record['audio_file']
    if target.exists() or target.is_symlink():
        if file_hash(regular_file(target)) != record['audio_sha256']:
            raise ValueError('Previously selected audio changed')
    else:
        with tempfile.TemporaryDirectory(prefix='.selection-', dir=directory) as staging:
            copied = Path(staging) / 'audio.wav'
            shutil.copyfile(regular_file(selected['audio']), copied)
            if file_hash(copied) != record['audio_sha256']:
                raise ValueError('Selected audio changed during publication')
            os.replace(copied, target)
    if file_hash(regular_file(target)) != record['audio_sha256']:
        raise ValueError('Selected audio changed during publication')
    write_json(directory / 'accepted.json', record)
    return target, record
