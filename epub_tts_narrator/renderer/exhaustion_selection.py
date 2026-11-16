from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import tempfile

from .audio_naming import speech_take_filename
from .common import digest, read_json, regular_file, write_json
from .stored_audio import StoredAudioError, audio_hash, audio_present, copy_audio
from .clip_repair_contract import logical_text, plain, synthesis_request
from .clip_repair_prompts import SELECT
from .clip_repair_state import ClipRecoveryUnavailable, ClipRepairRefused, model_call
from .diagnostics import safe_error_text
from .exhaustion_candidates import candidate_summary, retained_candidates, same_span, stylized_spelling_advisory


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


def _advisory_compatible(payload):
    # The stylized_spelling advisory derives only from requested_text, which is
    # already pinned, so pending states saved before it existed stay resumable.
    # So do states saved before candidates' nested audits were sent as digests
    # (exhaustion_candidates._compact_inputs): candidate IDs never changed.
    from .exhaustion_candidates import _compact_inputs
    result = deepcopy(payload)
    result.pop('stylized_spelling', None)
    for row in result.get('candidates') or []:
        if isinstance(row, dict) and 'actual_inputs' in row:
            row['actual_inputs'] = _compact_inputs(row['actual_inputs'])
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
            or matches[0] not in (candidate_summary(original, selected, physical),
                                  candidate_summary(original, selected, physical, compact=False))):
        raise ValueError('Selected audio evidence changed')
    return selected


def validate_record(record):
    metadata = record.get('engine_metadata', {})
    if 'tone_filter' in metadata:
        # A take with a steady tone filtered out (steady_tone.filtered_fallback):
        # its generation proof is the take it was filtered from.
        notice = metadata['tone_filter']
        if not isinstance(notice, dict) or notice.get('version') != 1 or not isinstance(
                notice.get('source_audio_sha256'), str):
            raise ValueError('Saved tone filtering has an unknown record')
        validate_record({**record, 'audio_sha256': notice['source_audio_sha256'],
                         'engine_metadata': {key: value for key, value in metadata.items() if key != 'tone_filter'}})
        return
    if 'manual_selection' in metadata:
        from .take_selection import validate_manual_selection
        validate_manual_selection(record)
        return
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


def _failure_context(failure_report):
    if not isinstance(failure_report, dict):
        return ''
    error = failure_report.get('provider_error') or failure_report.get('error')
    if not error:
        return ''
    return '; original failure: ' + safe_error_text(str(error))


def select_retained(directory, request, client, *, failure_report=None):
    from .history_echo import archive, candidate_echo
    directory = Path(directory)
    retained = retained_candidates(directory, request)
    # A take that replayed another line of chapter history is not a rendition
    # of this passage at all, so it is never a best-effort candidate.
    echoes = {item['summary']['candidate_id']: echo for item in retained
              if (echo := candidate_echo(item['record'], logical_text(request))) is not None}
    # A take a person listened to and marked bad is never chosen either.
    from .take_feedback import judged_bad_audio
    bad_audio = judged_bad_audio(directory)
    judged_bad = {item['summary']['candidate_id'] for item in retained
                  if item['record'].get('audio_sha256') in bad_audio}
    # A take that went silent before ending stopped short of its text; a take
    # finished from such a start (take_completion) that still failed is a
    # splice of one, never a whole best-effort rendition.
    stopped = {item['summary']['candidate_id'] for item in retained
               if (item['record'].get('report') or {}).get('generation_stopped') == 'silence'
               or 'completion' in (item['record'].get('engine_metadata') or {})}
    # A take that misread a name whose pronunciation a person chose is never
    # published, however good the rest of it is (``name_shape``).
    # Judged again from each take's transcript: a take verified where the
    # check was once skipped carries no rejection reason (name_shape.misread).
    from .name_shape import misread as misread_words
    misread = {item['summary']['candidate_id']: words for item in retained
               if (words := misread_words(item['record'], require_heard=True))}
    # And takes whose chosen names were not said as chosen, by their phonemes.
    from .name_phonemes import describe as phoneme_misreads, judge
    for item in retained:
        if item['summary']['candidate_id'] in misread:
            continue
        verdicts = judge(item['record'], item['audio'].parent)
        if verdicts and not all(verdict['ok'] for verdict in verdicts):
            misread[item['summary']['candidate_id']] = phoneme_misreads(verdicts)
    # Nor is a take carrying a steady high-pitched tone (steady_tone).
    from .steady_tone import REASON as TONE
    tonal = {item['summary']['candidate_id'] for item in retained
             if TONE in ((item['record'].get('report') or {}).get('reasons') or [])}
    candidates = [item for item in retained
                  if item['summary']['candidate_id'] not in echoes and item['summary']['candidate_id'] not in judged_bad
                  and item['summary']['candidate_id'] not in stopped and item['summary']['candidate_id'] not in misread
                  and item['summary']['candidate_id'] not in tonal]
    if not candidates:
        context = _failure_context(failure_report) + f'. Inspect {directory}'
        if tonal and not misread:
            raise ValueError('Every usable retained take carries a steady high-pitched tone, so none is published'
                             + context)
        if misread:
            heard = "; ".join(sorted({words for words in misread.values() if words}))
            raise ValueError('Every usable retained take misread a name whose pronunciation you chose'
                             + (f' ({heard})' if heard else '') + ', so none is published; edit the passage, '
                             'its direction or the pronunciation and retry' + context)
        if stopped and not echoes and not judged_bad:
            raise ValueError('Every retained take went silent before finishing its text (or was finished '
                             'from such a start and still failed), so none is published' + context)
        if judged_bad:
            raise ValueError('Every usable retained take was marked bad by a person listening, so none is '
                             'published' + context)
        if echoes:
            raise ValueError('Every retained take repeats an earlier line of chapter history (history echo); '
                             'none is a rendition of this passage, so none is published' + context)
        raise ValueError('No physically usable bound full-span candidate remains' + context)
    payload = {'mode': 'exhaustion_selection', 'request_sha256': digest(request),
               'requested_text': logical_text(request), 'request': deepcopy(request),
               'candidates': [item['summary'] for item in candidates]}
    advisory = stylized_spelling_advisory(logical_text(request))
    if advisory is not None:
        payload['stylized_spelling'] = advisory
    path = directory / 'exhaustion-selection.json'
    identity = policy(client.identity)
    state = {'audit': {'identity': identity, 'payload': payload, 'response': None}, 'response_received': False}
    if path.exists() or path.is_symlink():
        saved = read_json(regular_file(path))
        offered = {item.get('candidate_id') for item in
                   saved.get('audit', {}).get('payload', {}).get('candidates', [])}
        if offered & (set(echoes) | judged_bad | set(misread) | tonal):
            # A pending choice that offered echoes, takes a person marked bad
            # or takes that misread a chosen name is retired, not trusted.
            archive(directory, ['exhaustion-selection.json'],
                    {'retired_candidates': sorted(offered & (set(echoes) | judged_bad | set(misread) | tonal))})
    if path.exists() or path.is_symlink():
        saved = read_json(regular_file(path))
        state = {key: value for key, value in saved.items() if key != 'record_sha256'}
        if saved.get('record_sha256') != digest(state) or state['audit']['identity'] != identity:
            raise ValueError('Pending exhaustion selection inputs changed')
        responses = []
        for audit in state.get('invalid_responses', []):
            if (audit.get('identity') != identity
                    or _advisory_compatible(audit.get('payload')) != _advisory_compatible(_correction_payload(payload, responses))):
                raise ValueError('Saved chooser correction history changed')
            responses.append(audit['response'])
        if _advisory_compatible(state['audit']['payload']) != _advisory_compatible(_correction_payload(payload, responses)):
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
        try:
            state['audit']['response'] = model_call(client, SELECT, state['audit']['payload'])
        except ClipRepairRefused as exc:
            raise ValueError(f'{exc}; no best-effort take is chosen. Inspect {directory}') from exc
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
    filename = speech_take_filename(request, record['attempt'], role='selection')
    legacy_name = 'selection-' + record['audio_sha256'] + '.wav'
    if audio_present(directory / legacy_name):
        filename = legacy_name
    record.update(request=deepcopy(request), request_hash=digest(request),
                  audio_file=filename,
                  validation_key=digest({'selection': marker, 'audio': record['audio_sha256']}),
                  report={**report, 'passed': True, 'reasons': [], 'expected_text': logical_text(request),
                          'text_confirmation_method': 'llm_exhaustion_selection',
                          'acceptance_policy': state['audit']['identity'],
                          'ordinary_verification_passed': report.get('passed') is True,
                          'original_failed_report': report})
    record['engine_metadata'] = {**record.get('engine_metadata', {}), 'exhaustion_selection': marker}
    validate_record(record)
    target = directory / record['audio_file']
    if audio_present(target):
        if audio_hash(target) != record['audio_sha256']:
            raise ValueError('Previously selected audio changed')
    else:
        with tempfile.TemporaryDirectory(prefix='.selection-', dir=directory) as staging:
            try:
                copy_audio(selected['audio'], target, expected_sha256=record['audio_sha256'], staging=staging)
            except StoredAudioError as exc:
                raise ValueError('Selected audio changed during publication') from exc
    if audio_hash(target) != record['audio_sha256']:
        raise ValueError('Selected audio changed during publication')
    from .take_selection import current_acceptance
    chosen = current_acceptance(directory, digest(request))
    if chosen is not None:  # A person chose a take while the selection ran: theirs is used.
        return directory / chosen['audio_file'], chosen
    from .acceptance_gate import refusal
    why = refusal(record, directory)
    if why is not None:  # The candidate filters above make this unreachable; never publish it.
        raise ValueError(f'The chosen take cannot be published: {why}. Inspect {directory}')
    write_json(directory / 'accepted.json', record)
    return target, record
