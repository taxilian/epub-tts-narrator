from pathlib import Path

from .common import digest
from .clip_repair_contract import synthesis_request
from .clip_repair_execute import execute_plan
from .clip_repair_state import ClipRepairState, ClipRecoveryUnavailable
from .exhaustion_candidates import retained_candidates
from .exhaustion_selection import select_retained
from .retry_budget import SpeechRetryBudget


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
    pending = any((directory / name).exists() or (directory / name).is_symlink()
                  for name in ('clip-repair.json', 'exhaustion-selection.json'))
    if enabled and options.get('retry_budget') is None and 'recovery' not in request:
        options['retry_budget'] = SpeechRetryBudget(retry_paragraphs=True)
    failure_report = None
    if not enabled or not pending or accepted.exists() or accepted.is_symlink():
        try:
            return ordinary(text, generate, directory, request, verifier, config,
                            broad_recovery=enabled and generate_plan_part is not None, **options)
        except SpeechVerificationFailed as exc:
            if not enabled or exc.request != request or exc.report.get('resolution_action') == 'split':
                raise
            failure_report = exc.report
    notify = options.get('notify', print)
    try:
        selector = directory / 'exhaustion-selection.json'
        if (not selector.exists() and not selector.is_symlink() and generate_plan_part is not None
                and options.get('allow_resolution', True)):
            session = ClipRepairState(directory, request, client)
            budget = options['retry_budget']
            episode = options.get('resolution_episode') or request.get('parent_request_hash', digest(request))
            maximum = config.get('verification', {}).get('max_resolution_attempts', 12)
            reserve = options.get('episode_reserve', 0)
            candidates = retained_candidates(directory, request)
            failures = [item['summary'] for item in candidates]
            if not failures and failure_report:
                failures = [{'original_report': failure_report}]
            plan = session.prepare(failures, maximum - budget.episode_started(episode) - reserve)
            if plan is not None or session.state['phase'] == 'evaluated':
                notify('  Restoring the completed clip repair.' if session.state['phase'] == 'evaluated'
                       else '  Executing a clip-local repair approved by two independent reviewers.')
                result = execute_plan(session, generate_plan_part, verifier, config,
                                      budget=budget, episode=episode, reserve=reserve)
                if result is not None:
                    return result
        notify('  Selecting the best physically usable full-clip take under the best-effort exhaustion policy.')
        return select_retained(directory, request, client)
    except ClipRecoveryUnavailable as exc:
        raise VerificationUnavailable(str(exc)) from exc
