from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

from ..provider import ProviderError
from .common import digest, read_json, regular_file, write_json
from .clip_repair_contract import approved, parse_plan, policy, review_payload
from .clip_repair_prompts import PLAN, REVIEW
from .render_resource_context import current_owner


class ClipRecoveryUnavailable(RuntimeError):
    pass


class ClipRepairRefused(RuntimeError):
    """The provider refused this request for its size (context too long, HTTP 413).

    Not an outage: sending the same request again cannot succeed, so only this
    passage's repair ends and its other recoveries go on. Any other refusal
    (a bad model name or setting) stays an outage that stops the run once,
    rather than silently ending every repair it reaches.
    """


def model_call(client, prompt, payload):
    try:
        return client.complete(prompt, deepcopy(payload))
    except ProviderError as exc:
        if getattr(exc, "status_code", None) == 413 or getattr(exc, "code", None) == "context_length_exceeded":
            raise ClipRepairRefused(f"The recovery model refused the repair request ({exc})") from exc
        raise ClipRecoveryUnavailable('Clip recovery model unavailable; pending progress retained') from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ClipRecoveryUnavailable('Clip recovery model unavailable; pending progress retained') from exc


class ClipRepairState:
    def __init__(self, directory, request, client):
        self.path = Path(directory) / 'clip-repair.json'
        self.request, self.client = request, client
        self.lock = Lock()
        self.binding = {'request_sha256': digest(request), 'policy': policy(client.identity, request)}
        self.state = {'binding': self.binding, 'phase': 'pending', 'audit': None, 'parts': []}
        if self.path.exists() or self.path.is_symlink():
            saved = read_json(regular_file(self.path))
            body = {key: value for key, value in saved.items() if key != 'record_sha256'}
            if saved.get('record_sha256') != digest(body) or body.get('binding') != self.binding:
                raise ValueError('Saved clip repair binding changed; refusing to discard its progress')
            self.state = body
            if body.get('phase') == 'rejected' and body.get('error_code') == 'unsupported_performance_controls':
                # Refused only because the line had scoped delivery controls, which a repair now
                # replaces with its parts' own delivery: its received proposal goes on to review.
                self.state = {key: value for key, value in body.items() if key != 'error_code'}
                self.state['phase'] = 'pending'

    def save(self):
        if self.path.exists() or self.path.is_symlink():
            regular_file(self.path)
        write_json(self.path, {**self.state, 'record_sha256': digest(self.state)})

    def prepare(self, failures, remaining):
        if self.state['phase'] in {'rejected', 'evaluated', 'failed'}:
            return None
        if self.state['audit'] is not None and not self.state.get('proposal_received'):
            # Nothing was received for the saved request, so it is rebuilt from
            # the current failures: a request saved before failure records were
            # compacted (exhaustion_candidates) could be too large to ever send.
            self.state['audit'] = None
        if self.state['audit'] is None:
            payload = {'mode': 'clip_repair', 'request': deepcopy(self.request),
                       'request_sha256': digest(self.request),
                       'failures': deepcopy(failures), 'remaining_starts': max(0, remaining)}
            from .clip_repair_performance import controlled_repair_guidance
            guidance = controlled_repair_guidance(self.request)
            if guidance is not None:
                payload['controlled_pronunciation'] = guidance
            self.state['audit'] = {'identity': self.binding['policy'], 'payload': payload,
                                   'proposal': None, 'reviews': [None, None]}
            self.state['proposal_received'] = False
            self.save()
        audit = self.state['audit']
        if self.state['phase'] in {'approved', 'executing'}:
            return approved(self.request, audit)
        if remaining <= 0:
            # Only this run's shared allowance is spent: the repair stays
            # pending for a later run instead of being rejected for good.
            return None
        if not self.state['proposal_received']:
            try:
                audit['proposal'] = model_call(self.client, PLAN, audit['payload'])
            except ClipRepairRefused:
                self.state.update(phase='failed', error_code='request_refused')
                self.save()
                return None
            self.state['proposal_received'] = True
            self.save()
        try:
            plan = parse_plan(audit['proposal'], audit['payload']['remaining_starts'])
        except ValueError:
            self.state['phase'] = 'rejected'
            self.save()
            return None
        if not plan['parts']:
            self.state['phase'] = 'rejected'
            self.save()
            return None
        try:
            from .clip_repair_performance import project_plan_performance
            project_plan_performance(self.request, plan)
        except ValueError:
            self.state.update(phase='rejected', error_code='invalid_control_projection')
            self.save()
            return None
        missing = [number for number in (1, 2) if audit['reviews'][number - 1] is None]
        # A copy: completing votes remove from pending_reviews while the
        # submissions below iterate `missing` on another thread.
        self.state['pending_reviews'] = list(missing)
        self.save()
        errors = []

        def review(number):
            payload = review_payload(audit['payload'], plan, number)
            response = model_call(self.client, REVIEW, payload)
            with self.lock:
                audit['reviews'][number - 1] = {'payload': payload, 'response': response}
                self.state['pending_reviews'].remove(number)
                self.save()

        resources = self._review_capacity()
        executor = None
        if resources is not None:
            # Both blind votes run on the shared review-coordination capacity.
            # ResourceLane copies the caller's scoped context into each vote,
            # so an owner lost while votes wait never starts new vote work.
            futures = [resources.review_coordination.submit(review, number,
                                                            owner=current_owner())
                       for number in missing]
        else:
            executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='clip-repair-review')
            futures = [executor.submit(review, number) for number in missing]
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except (ClipRecoveryUnavailable, ClipRepairRefused) as exc:
                    errors.append(exc)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        if any(isinstance(error, ClipRepairRefused) for error in errors):
            self.state.update(phase='failed', error_code='request_refused')
            self.state.pop('pending_reviews', None)
            self.save()
            return None
        if errors:
            raise errors[0]
        try:
            plan = approved(self.request, audit)
        except (ValueError, TypeError, KeyError):
            self.state['phase'] = 'rejected'
            self.save()
            return None
        self.state['phase'] = 'approved'
        self.save()
        return plan

    @staticmethod
    def _review_capacity():
        """Pipeline coordination capacity, or None for the legacy executor."""
        from .render_resource_context import current_resources
        return current_resources()
