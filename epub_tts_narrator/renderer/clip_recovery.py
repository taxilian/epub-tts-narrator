from pathlib import Path

from .common import digest, read_json, regular_file
from .clip_repair_contract import approved, logical_text, synthesis_request
from .continuity import history_recovery
from .history_echo import candidate_echo
from .clip_repair_execute import execute_plan
from .clip_repair_state import ClipRepairState, ClipRecoveryUnavailable
from .exhaustion_candidates import retained_candidates, same_span
from .exhaustion_selection import select_retained
from .retry_budget import SpeechRetryBudget


def _repair_resumable(directory, state):
    """Whether a saved clip repair still has work that must resume first."""
    phase = state.get('phase')
    if phase in {'rejected', 'evaluated'}:
        return False
    if phase != 'failed':
        return True
    folder = Path(directory) / 'clip-repair-parts'
    for part in state.get('parts') or []:
        if not isinstance(part, dict):
            continue
        for file_key in ('raw_file', 'audio_file'):
            name = part.get(file_key)
            if isinstance(name, str) and (folder / name).is_file():
                return True
    return False


def _saved_failure_report(directory, request):
    """Return the latest saved failed report for this exact request, if any."""
    failures = []
    for path in sorted(Path(directory).glob('attempt-*.json')):
        try:
            record = read_json(regular_file(path))
        except (ValueError, OSError):
            continue
        actual = record.get('request')
        if (not isinstance(actual, dict) or not same_span(request, actual)
                or record.get('request_hash') != digest(actual)
                or not isinstance(record.get('report'), dict)
                or record['report'].get('passed') is not False):
            continue
        failures.append(record)
    if not failures:
        return None
    latest = max(failures, key=lambda record: record.get('attempt', 0))
    return latest['report']


def run(ordinary, text, generate, directory, request, verifier, config, *, generate_plan_part=None, **options):
    from .workflow import SpeechVerificationFailed, VerificationUnavailable
    resolver = getattr(verifier, 'speech_resolver', None)
    client = getattr(resolver, 'clip_recovery_client', None)
    actual = synthesis_request(request)
    bound = (actual.get('chunk', {}).get('spoken_text') == text
             and bool(actual.get('voice_asset_hash') or actual.get('voice_assets')))
    enabled = (client is not None and options.get('purpose', 'speech') == 'speech'
               and bound
               and request.get('kind') in {'speech', 'speech_recovery_part', 'paragraph_speech'})
    directory = Path(directory)
    if enabled:
        from .speech_recovery import _directory
        directory = _directory(directory)
    accepted = directory / 'accepted.json'
    selector = directory / 'exhaustion-selection.json'
    repair_path = directory / 'clip-repair.json'
    repair_session = None
    repair_pending = False
    if enabled and (repair_path.exists() or repair_path.is_symlink()):
        repair_session = ClipRepairState(directory, request, client)
        repair_pending = _repair_resumable(directory, repair_session.state)
    terminal_repair = repair_session is not None and not repair_pending
    pending = selector.exists() or selector.is_symlink() or repair_pending
    if enabled and options.get('retry_budget') is None and 'recovery' not in request:
        options['retry_budget'] = SpeechRetryBudget(retry_paragraphs=True)
    failure_report = None
    if not enabled or not pending or accepted.exists() or accepted.is_symlink():
        ordinary_options = dict(options)
        if terminal_repair:
            ordinary_options['allow_resolution'] = False
        try:
            return ordinary(text, generate, directory, request, verifier, config,
                            broad_recovery=enabled and generate_plan_part is not None, **ordinary_options)
        except SpeechVerificationFailed as exc:
            if not enabled or exc.request != request or exc.report.get('resolution_action') == 'split':
                raise
            failure_report = exc.report
    notify = options.get('notify', print)
    try:
        if (not selector.exists() and not selector.is_symlink() and generate_plan_part is not None
                and options.get('allow_resolution', True)):
            session = repair_session or ClipRepairState(directory, request, client)
            budget = options['retry_budget']
            episode = options.get('resolution_episode') or request.get('parent_request_hash', digest(request))
            maximum = config.get('verification', {}).get('max_resolution_attempts', 12)
            reserve = options.get('episode_reserve', 0)
            candidates = retained_candidates(directory, request)
            failures = [item['summary'] for item in candidates]
            if not failures and failure_report:
                failures = [{'original_report': failure_report}]
            plan = session.prepare(failures, maximum - budget.episode_started(episode) - reserve)
            if (plan is None and session.state['phase'] == 'failed'
                    and session.state.get('error_code') in
                    {'synthesis_or_input_failure', 'shared_budget_exhausted'}):
                session.state['phase'] = 'approved'
                session.save()
                plan = approved(request, session.state['audit'])
            if plan is not None or session.state['phase'] == 'evaluated':
                notify('  Restoring the completed clip repair.' if session.state['phase'] == 'evaluated'
                       else '  Executing a clip-local repair approved by two independent reviewers.')
                # Repair parts that heard the history this line echoed would
                # repeat it again; they keep only the reference voice.
                expected = logical_text(request)
                echoed = any(candidate_echo(item['record'], expected) for item in candidates)
                with history_recovery('none' if echoed else None):
                    result = execute_plan(session, generate_plan_part, verifier, config,
                                          budget=budget, episode=episode, reserve=reserve)
                if result is not None:
                    return result
        notify('  Selecting the best physically usable full-clip take under the best-effort exhaustion policy.')
        failure_context = failure_report if failure_report is not None else _saved_failure_report(directory, request)
        return select_retained(directory, request, client, failure_report=failure_context)
    except ClipRecoveryUnavailable as exc:
        raise VerificationUnavailable(str(exc)) from exc
