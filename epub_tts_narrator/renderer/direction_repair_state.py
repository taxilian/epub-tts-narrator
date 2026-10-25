"""One durable direction candidate and one synthesis trial per original request."""

from copy import deepcopy
from pathlib import Path

from ..provider import ProviderError
from .common import digest, file_hash, read_json, regular_file, write_json
from .direction_repair import (make_payload, policy, proposal_valid, review_payload,
                               review_valid, validate_marker)
from .direction_repair_prompts import PROPOSE, REVIEW


class DirectionRepairUnavailable(RuntimeError):
    pass


class DirectionRepair:
    def __init__(self, directory, request, client):
        self.path = Path(directory) / 'direction-repair.json'
        self.request, self.client = request, client
        self.binding = {'request_sha256': digest(request), 'policy': policy(client.identity)}
        self.state = None
        if self.path.exists() or self.path.is_symlink():
            saved = read_json(regular_file(self.path))
            body = {key: value for key, value in saved.items() if key != 'record_sha256'}
            if (saved.get('record_sha256') != digest(body) or body.get('binding') != self.binding
                    or body.get('phase') not in {'pending', 'approved', 'generating', 'failed', 'rejected'}
                    or not isinstance(body.get('audit'), dict)):
                raise ValueError('Saved direction repair identity or state changed; refusing to reroll it')
            self.state = body

    def _save(self):
        if self.path.exists() or self.path.is_symlink():
            regular_file(self.path)
        write_json(self.path, {**self.state, 'record_sha256': digest(self.state)})

    def _call(self, prompt, payload):
        self.state['pending_call'] = digest({'prompt': prompt, 'payload': payload})
        self._save()
        if self.client.identity != self.binding['policy']['provider']:
            raise DirectionRepairUnavailable('Direction repair provider changed; pending work retained')
        try:
            return self.client.complete(prompt, deepcopy(payload))
        except (ProviderError, OSError, RuntimeError, ValueError) as exc:
            raise DirectionRepairUnavailable('Direction repair model unavailable; pending work retained') from exc

    def prepare(self, failures):
        if self.state is None:
            self.state = {'binding': self.binding, 'phase': 'pending', 'proposal_received': False, 'audit': {
                'identity': self.binding['policy'], 'payload': make_payload(self.request, failures),
                'proposal': None, 'reviews': []}}
            self._save()
        if self.state['phase'] in {'failed', 'rejected'}:
            return None
        audit = self.state['audit']
        if self.state['phase'] in {'approved', 'generating'}:
            validate_marker(self.request, audit)
            return deepcopy(audit)
        if audit['payload'] != make_payload(self.request, audit['payload']['failures']):
            raise ValueError('Pending direction repair context changed')
        if not self.state['proposal_received']:
            audit['proposal'] = self._call(PROPOSE, audit['payload'])
            self.state['proposal_received'] = True
            self._save()
        proposal = audit['proposal']
        if (not proposal_valid(proposal) or not proposal['control_related']
                or proposal['replacement_direction'] == audit['payload']['original_direction']):
            self.state['phase'] = 'rejected'
            self._save()
            return None
        for number in range(len(audit['reviews']) + 1, 4):
            payload = review_payload(audit['payload'], proposal, number)
            response = self._call(REVIEW, payload)
            audit['reviews'].append({'payload': payload, 'response': deepcopy(response)})
            self._save()
            if not review_valid(response):
                self.state['phase'] = 'rejected'
                self._save()
                return None
        try:
            validate_marker(self.request, audit)
        except ValueError:
            self.state['phase'] = 'rejected'
            self._save()
            return None
        self.state['phase'] = 'approved'
        self.state.pop('pending_call', None)
        self._save()
        return deepcopy(audit)

    def begin(self, attempt):
        if self.state is None or self.state['phase'] != 'approved':
            raise ValueError('Direction repair trial already consumed; retained takes must not be rerolled')
        validate_marker(self.request, self.state['audit'])
        self.state.update(phase='generating', attempt=attempt)
        self._save()

    def check_trial(self):
        if self.state is None or self.state['phase'] not in {'generating', 'failed'}:
            return
        attempt = self.state.get('attempt')
        path = self.path.parent / f'attempt-{attempt:03d}.json'
        record = read_json(regular_file(path))
        if (self.state['phase'] == 'generating'
                and record.get('request_hash') == self.binding['request_sha256']
                and record.get('report', {}).get('reasons') == ['generation_incomplete']
                and record.get('audio_sha256') is None and record.get('engine_metadata') == {}):
            # No outcome or actual conditioning was committed. Retain this orphan
            # but never verify it; a fresh budgeted start uses the same approval.
            self.state.setdefault('abandoned_attempts', []).append(attempt)
            self.state['phase'] = 'approved'
            self.state.pop('attempt')
            self._save()
            return
        if (record.get('request_hash') != self.binding['request_sha256']
                or record.get('audio_file') != f'attempt-{attempt:03d}.wav'
                or record.get('engine_metadata', {}).get('direction_repair') != self.state['audit']
                or file_hash(regular_file(self.path.parent / record['audio_file'])) != record.get('audio_sha256')):
            raise ValueError('Interrupted direction trial lacks bound audio; refusing another synthesis trial')

    def abandoned(self, attempt):
        return self.state is not None and attempt in self.state.get('abandoned_attempts', [])

    def active(self):
        if self.state is None or self.state['phase'] not in {'approved', 'generating'}:
            return None
        validate_marker(self.request, self.state['audit'])
        return deepcopy(self.state['audit'])

    def failed(self, record):
        if self.state and self.state.get('attempt') == record['attempt']:
            self.state.update(phase='failed', failed_record_sha256=digest(record))
            self._save()
            return True
        return self.state is not None and self.state['phase'] == 'failed'
