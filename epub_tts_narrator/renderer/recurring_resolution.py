"""Persistent, request-scoped recurrence tracking and bounded model decisions."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from .common import digest, file_hash, read_json, regular_file, write_json
from .quality import compare_words
from .retry_budget import generation_attempts


def failure_signatures(expected, report):
    """Count a source discrepancy once per take, even if both ASRs report it.

    Different spellings of the same problematic source span share a signature;
    the actual spellings remain in the model evidence, never normalized away.
    """
    if report.get('passed') or not report.get('reasons'):
        return set()
    if report['reasons'] != ['text_mismatch']:
        acoustic = {'silent_audio', 'audio_too_quiet', 'excessive_silence',
                    'internal_silence_stall', 'excessive_clipping', 'audio_too_short'}
        return {digest({'acoustic': reason}) for reason in report['reasons'] if reason in acoustic}
    found = set()
    for transcript in (report.get('transcript'), report.get('secondary_transcript'),
                       report.get('corroboration', {}).get('transcript')):
        if not isinstance(transcript, str) or not transcript.strip():
            continue
        changes = []
        def flush():
            if changes:
                found.add(digest({'start': changes[0]['expected_start'],
                                  'end': changes[-1]['expected_end'],
                                  'expected': [word for row in changes for word in row['expected']],
                                  # A source span that disappears and one that
                                  # gets misrecognized are the same unresolved
                                  # source content. Keep insertions separate;
                                  # their anchor has no missing source words.
                                  'kind': 'extra' if all(row['operation'] == 'insert' for row in changes)
                                  else 'source'}))
                changes.clear()
        for row in compare_words(expected, transcript)['alignment']:
            if row['operation'] == 'equal':
                flush()
            else:
                changes.append(row)
        flush()
    return found


def failed_take_context(expected_text, record):
    report = record['report']
    metadata = record.get('engine_metadata', {})
    marker = metadata.get('speech_resolution') or {}
    result = {'attempt': record['attempt'], 'audio_sha256': record.get('audio_sha256'),
            'transcript': report.get('transcript') or '',
            'secondary_transcript': (report.get('secondary_transcript')
                                     or report.get('corroboration', {}).get('transcript') or ''),
            'tts_text': marker.get('tts_text', expected_text),
            'reasons': report.get('reasons', [])}
    # Give the planner actual synthesis context instead of making it infer
    # performance problems from spellings alone. These remain data, never
    # instructions or evidence that an omitted sound really occurred.
    generation = {key: metadata[key] for key in ('synthesis_text', 'direction', 'applied_control')
                  if isinstance(metadata.get(key), str) and len(metadata[key]) <= 20000}
    if generation:
        result['generation'] = generation
    if isinstance(report.get('metrics'), dict):
        # Full alignments/ASR segments repeat the supplied transcripts and
        # can exhaust the planner's input limit after many long takes.
        result['metrics'] = {key: report['metrics'][key] for key in (
            'duration_seconds', 'rms_dbfs', 'peak_dbfs', 'silence_fraction',
            'longest_internal_silence_seconds', 'clipped_fraction',
            'insertions', 'deletions', 'substitutions', 'expected_word_count',
            'actual_word_count', 'wer') if type(report['metrics'].get(key)) in (int, float)}
    return result


class RecurringResolution:
    def __init__(self, directory, request, text, resolver, settings, *, can_rewrite, can_split,
                 can_plain_delivery=False):
        self.directory, self.request, self.text = Path(directory), request, text
        self.resolver, self.identity = resolver, resolver.identity
        self.can_rewrite, self.can_split = can_rewrite, can_split
        self.can_plain_delivery = can_plain_delivery
        self.threshold = settings.get('recurrence_threshold', 3)
        self.maximum_rounds = settings.get('max_resolution_rounds', 2)
        self.rounds = 0
        self.path = self.directory / 'speech-resolution.json'
        self.binding = {'version': 2, 'request_hash': digest(request), 'expected_text': text,
                        'resolver': self.identity, 'threshold': self.threshold,
                        'can_rewrite': can_rewrite, 'can_split': can_split}
        if can_plain_delivery:
            self.binding['can_plain_delivery'] = True
        self.state = {'binding': self.binding, 'handled': {}, 'history': []}
        if self.path.exists() or self.path.is_symlink():
            saved = read_json(regular_file(self.path))
            if not isinstance(saved, dict):
                raise ValueError('Invalid saved speech resolution record')
            body = {key: value for key, value in saved.items() if key != 'record_sha256'}
            if saved.get('record_sha256') != digest(body):
                raise ValueError('Saved speech resolution integrity check failed')
            if (set(body) != {'binding', 'handled', 'history'}
                    or not isinstance(body['binding'], dict)
                    or not isinstance(body['handled'], dict)
                    or any(not isinstance(key, str) or type(value) is not int or value < 1
                           for key, value in body['handled'].items())
                    or not isinstance(body['history'], list)
                    or any(not isinstance(row, dict) or not isinstance(row.get('audit'), dict)
                           or not isinstance(row.get('failures'), list)
                           for row in body['history'])):
                raise ValueError('Invalid saved speech resolution state')
            if body.get('binding') == self.binding:
                self.state = body
            elif (set(body['binding']) == set(self.binding)
                  and {k: v for k, v in body['binding'].items() if k != 'can_split'}
                      == {k: v for k, v in self.binding.items() if k != 'can_split'}):
                # A failed assembled performance removes splitting as a useful
                # action. Keep already reviewed spelling/delivery repairs and
                # their outcomes while reconsidering the remaining actions.
                archive = self.directory / ('speech-resolution-' + digest(body) + '.json')
                if not archive.exists():
                    write_json(archive, saved)
                self.state = dict(body, binding=self.binding, handled={})
            elif body.get('binding') == dict(self.binding, version=1):
                # Source-span signatures changed, not the reviewed evidence.
                # Retain approved rewrites and feedback; recalculate recurrence
                # from the saved distinct takes using the new signature policy.
                self.state = dict(body, binding=self.binding, handled={})
            else:
                # Retain old decisions for inspection when a model/policy changes.
                archive = self.directory / ('speech-resolution-' + digest(body) + '.json')
                if not archive.exists():
                    write_json(archive, saved)
        self.records = {}
        for number in generation_attempts(self.directory):
            path = self.directory / f'attempt-{number:03d}.json'
            if not path.exists():
                continue
            record = read_json(regular_file(path))
            if record.get('request_hash') != digest(request):
                raise ValueError('Speech resolution attempt belongs to another request')
            if record.get('attempt') != number:
                raise ValueError('Speech resolution attempt number changed')
            if record.get('report', {}).get('passed') is not True:
                self.records[number] = record

    def _save(self):
        write_json(self.path, {**self.state, 'record_sha256': digest(self.state)})

    def _failure(self, record):
        return failed_take_context(self.text, record)

    def _feedback(self, history):
        """Keep bounded outcomes, not nested prompts or old failed-take payloads."""
        limits = self.identity.get('limits', {})
        maximum = limits.get('max_prior_decisions', 5)
        character_limit = limits.get('max_feedback_chars', 30000)

        def bounded(value, limit):
            try:
                encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError, OverflowError, RecursionError):
                return {'summary': 'The earlier model response was not valid bounded JSON.'}
            if len(encoded) <= limit:
                return deepcopy(value)
            return {'truncated': True, 'preview': encoded[:limit]}

        result = []
        for row in reversed(history):
            audit = row['audit']
            if (type(row.get('attempt')) is not int or row['attempt'] < 1
                    or audit.get('decision') not in {'resolved', 'blocked', 'unavailable'}):
                continue
            entry = {
                'attempt': row['attempt'], 'proposal': bounded(audit.get('proposal'), 12000),
                'decision': audit['decision'],
                'error_code': (audit['error_code'][:100] if isinstance(audit.get('error_code'), str) else None),
                'validation_error': (audit['validation_error'][:800]
                                     if isinstance(audit.get('validation_error'), str) else None),
                'sound_reviews': [
                    {'reviewer': review['reviewer'], 'response': bounded(review.get('response'), 4000)}
                    for review in audit.get('reviews', [])[:3]
                    if isinstance(review, dict) and type(review.get('reviewer')) is int
                    and 1 <= review['reviewer'] <= 3],
            }
            candidate = [entry, *result]
            if len(json.dumps(candidate, ensure_ascii=False, allow_nan=False)) > character_limit:
                break
            result = candidate
            if len(result) >= maximum:
                break
        return result

    def _saved_feedback(self, index):
        row = self.state['history'][index]
        if 'previous_decisions' not in row:
            # Earlier state records did not separately bind feedback. Their
            # existing audit keeps its original empty/historical payload.
            return None
        expected = self._feedback(self.state['history'][:index])
        if row['previous_decisions'] != expected:
            raise ValueError('Saved speech resolution feedback differs from its preceding decisions')
        return expected

    def active_rewrite(self):
        from .speech_resolver import validate_resolution
        for index in range(len(self.state['history']) - 1, -1, -1):
            row = self.state['history'][index]
            if (row['audit'].get('decision') != 'resolved'
                    or row['audit'].get('proposal', {}).get('action') != 'rewrite'):
                continue
            proposal = validate_resolution(row['audit'], self.text, row['failures'], self.identity,
                                           can_rewrite=self.can_rewrite,
                                           can_split='split' in row['audit'].get('payload', {}).get('permitted_actions', []),
                                           previous_decisions=self._saved_feedback(index))
            if proposal is None:
                raise ValueError('Saved speech resolution no longer validates')
            if proposal['action'] == 'rewrite':
                return {'version': 1, 'original_text': self.text,
                        'tts_text': proposal['tts_text'], 'audit': row['audit'],
                        'failures': row['failures']}
        return None

    def active_delivery(self):
        """Restore a validated performance repair without asking or spending again."""
        if not self.can_plain_delivery:
            return None
        from .speech_resolver import validate_resolution
        for index in range(len(self.state['history']) - 1, -1, -1):
            row = self.state['history'][index]
            if (row['audit'].get('decision') != 'resolved'
                    or row['audit'].get('proposal', {}).get('action') != 'plain_delivery'):
                continue
            proposal = validate_resolution(
                row['audit'], self.text, row['failures'], self.identity,
                can_rewrite=self.can_rewrite,
                can_split='split' in row['audit'].get('payload', {}).get('permitted_actions', []),
                can_plain_delivery=True,
                previous_decisions=self._saved_feedback(index))
            if proposal is None:
                raise ValueError('Saved plain delivery decision no longer validates')
            return {**proposal, 'audit': row['audit'], 'failures': row['failures']}
        return None

    def note_failed_split(self, report):
        """Give the planner the renderer's actual strategy outcome once.

        This is local feedback, not a model vote or permission to accept audio.
        It removes the exhausted strategy without discarding previous repairs.
        """
        if (not isinstance(report, dict) or report.get('expected_text') != self.text
                or report.get('passed') is not False or not report.get('reasons')):
            raise ValueError('Split feedback must describe failed verification of the complete source')
        evidence = {'reasons': report['reasons'], 'transcript': report.get('transcript'),
                    'secondary_transcript': (report.get('secondary_transcript')
                        or report.get('corroboration', {}).get('transcript'))}
        fingerprint = digest(evidence)
        if any(row.get('strategy_feedback_sha256') == fingerprint for row in self.state['history']):
            return
        summary = {key: value[:300] if isinstance(value, str) else value
                   for key, value in evidence.items()}
        self.state['history'].append({
            'attempt': max(self.records, default=1), 'failures': [], 'origin': 'renderer',
            'strategy_feedback_sha256': fingerprint,
            'audit': {'decision': 'blocked', 'error_code': 'split_full_verification_failed',
                      'proposal': {'action': 'split', 'edits': [],
                                   'reason': 'Parts passed individually, but the complete joined recording failed verification. Do not repeat that assembly.'},
                      'validation_error': json.dumps(summary, ensure_ascii=False)[:800]}})
        self.state['handled'] = {}
        self._save()

    def _check_candidate_audio(self, record):
        wav = self.directory / str(record.get('audio_file'))
        if (record.get('audio_file') != f"attempt-{record['attempt']:03d}.wav"
                or file_hash(regular_file(wav)) != record.get('audio_sha256')):
            raise ValueError('Speech resolution candidate audio changed')

    def consider(self, record, *, notify):
        from .speech_resolver import validate_resolution
        self.records[record['attempt']] = record
        # A decision is saved before its result is published. Replay only that
        # latest, still-applicable action after an interruption, without buying
        # more generations or counting the same take again. Fresh recognition
        # and acoustic results must match the evidence originally reviewed.
        previous = self.state['history'][-1] if self.state['history'] else None
        if (previous is not None and previous['audit'].get('decision') == 'resolved'
                and previous.get('attempt') == record['attempt']
                and previous.get('audio_sha256') == record.get('audio_sha256')
                and previous['failures']
                and previous['failures'][-1] == self._failure(record)):
            proposal = validate_resolution(
                previous['audit'], self.text, previous['failures'], self.identity,
                can_rewrite=self.can_rewrite, can_split=self.can_split,
                previous_decisions=self._saved_feedback(len(self.state['history']) - 1))
            if proposal is not None and proposal['action'] in {'accept', 'split'}:
                self._check_candidate_audio(record)
                notify('  Resuming saved speech recovery decision: ' + proposal['action'] + '.')
                return {**proposal, 'audit': previous['audit'], 'failures': previous['failures']}
        signatures = failure_signatures(self.text, record['report'])
        if not signatures or self.rounds >= self.maximum_rounds:
            return None
        counts = Counter(signature for saved in self.records.values()
                         for signature in failure_signatures(self.text, saved['report']))
        ready = sorted(signature for signature in signatures
                       if counts[signature] - self.state['handled'].get(signature, 0) >= self.threshold)
        if not ready:
            return None
        # Historical reports can trigger review, but only a current byte-bound
        # waveform can be accepted. The latest failure is always last in payload.
        self._check_candidate_audio(record)
        failures = [self._failure(saved) for number, saved in sorted(self.records.items())
                    if number <= record['attempt'] and failure_signatures(self.text, saved['report'])][-30:]
        notify(f'  Recurring speech discrepancy ({max(counts[s] for s in ready)} takes); asking the analysis model how to recover.')
        self.rounds += 1
        feedback = self._feedback(self.state['history'])
        # Once selected, plain delivery remains active for subsequent takes.
        # Do not waste a later review choosing the already-applied repair again.
        can_plain = self.can_plain_delivery and self.active_delivery() is None
        audit = self.resolver.resolve(self.text, failures, can_rewrite=self.can_rewrite,
                                      can_split=self.can_split, previous_decisions=feedback,
                                      can_plain_delivery=can_plain)
        row = {'trigger_counts': {key: counts[key] for key in ready},
               'attempt': record['attempt'], 'audio_sha256': record['audio_sha256'],
               'failures': failures, 'audit': audit, 'previous_decisions': feedback}
        self.state['history'].append(row)
        proposal = validate_resolution(audit, self.text, failures, self.identity,
                                       can_rewrite=self.can_rewrite, can_split=self.can_split,
                                       previous_decisions=feedback, can_plain_delivery=can_plain)
        if audit.get('decision') != 'unavailable':
            self.state['handled'].update(row['trigger_counts'])
        self._save()  # Commit the decision before changing input or accepting audio.
        if audit.get('decision') == 'unavailable':
            return {'action': 'unavailable', 'audit': audit, 'failures': failures}
        if proposal is None:
            detail = audit.get('validation_error') or audit.get('error_code') or 'invalid recovery decision'
            notify('  The recovery proposal did not pass validation: ' + detail
                   + '. Continuing within the remaining attempt limit.')
            return None
        return {**proposal, 'audit': audit, 'failures': failures}


def accepted_report(report, resolution):
    """Keep raw evidence and acoustic results; record the separate accepting vote."""
    if report.get('reasons') != ['text_mismatch'] or not report.get('transcript'):
        raise ValueError('A text resolution cannot override failed acoustic checks')
    return {**deepcopy(report), 'passed': True, 'reasons': [],
            'previous_reasons': list(report['reasons']),
            'text_confirmation_method': 'recurring_resolution',
            'verified_transcript': report['transcript'],
            'recurring_resolution': resolution}


def confirm_rewritten_report(report, marker):
    """Bridge a reviewed pronunciation spelling back to the unchanged source.

    Only complete local equivalence to the reviewed alternate input is usable;
    this never turns a second LLM's broad impression into missing-word approval.
    """
    from .speech_resolver import validate_resolution
    from .quality import word_spacing_equivalence
    from .phonetic_match import phonetic_equivalence
    if (not isinstance(marker, dict) or marker.get('original_text') != report.get('expected_text')
            or not isinstance(marker.get('audit'), dict)):
        raise ValueError('Pronunciation recovery is not bound to the original verification text')
    proposal = validate_resolution(marker['audit'], marker['original_text'], marker.get('failures'),
                                   marker['audit'].get('identity'), can_rewrite=True,
                                   can_split='split' in marker['audit'].get('payload', {}).get('permitted_actions', []))
    if (proposal is None or proposal.get('action') != 'rewrite'
            or proposal.get('tts_text') != marker.get('tts_text')):
        raise ValueError('Pronunciation recovery audit is invalid')
    if report.get('reasons') != ['text_mismatch']:
        return report
    transcript = report.get('transcript')
    if not isinstance(transcript, str) or not transcript.strip():
        return report
    rewritten = marker['tts_text']
    if (compare_words(rewritten, transcript)['wer'] == 0
            or word_spacing_equivalence(rewritten, transcript) is not None
            or phonetic_equivalence(rewritten, transcript) is not None):
        result = accepted_report(report, {'action': 'rewrite', **marker})
        result['text_confirmation_method'] = 'pronunciation_rewrite'
        return result
    return report


def confirm_recovery_report(report, request, metadata, directory):
    """Validate the full joined audio against independently accepted part wording.

    This carries scoped pronunciation decisions through sentence assembly. It
    still requires one complete final transcript; recognizer fragments are never
    stitched together to invent a passing transcription of the joined audio.
    """
    from .quality import word_spacing_equivalence
    from .phonetic_match import phonetic_equivalence
    if report.get('reasons') != ['text_mismatch']:
        return report
    planned = request['recovery']['parts']
    evidence = metadata['recovery'].get('parts', [])
    if len(evidence) != len(planned) or ''.join(part['text'] for part in planned) != report['expected_text']:
        raise ValueError('Recovered speech does not preserve its original source parts')
    accepted_text, proof = [], []
    for part, row in zip(planned, evidence):
        if any(row.get(key) != part[key] for key in ('start', 'end', 'text')):
            raise ValueError('Recovered speech source range changed')
        relative = Path(row['audio_file'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 3
                or relative.parts[0] != 'parts' or relative.parts[1] != row['request_hash']):
            raise ValueError('Recovered speech part path is invalid')
        audio = Path(directory) / relative
        if audio.parent.is_symlink() or audio.parent.parent.is_symlink():
            raise ValueError('Recovered speech parts must not use symlinks')
        acceptance_path = regular_file(audio.parent / 'accepted.json')
        if file_hash(acceptance_path) != row.get('accepted_sha256'):
            return report  # Older assemblies lack this binding; use normal verification.
        acceptance = read_json(acceptance_path)
        child = acceptance.get('report', {})
        if (acceptance.get('request_hash') != row['request_hash']
                or digest(acceptance.get('request')) != row['request_hash']
                or acceptance.get('audio_sha256') != row['audio_sha256']
                or acceptance.get('audio_file') != audio.name
                or file_hash(regular_file(audio)) != row['audio_sha256']
                or child.get('passed') is not True or child.get('reasons')
                or child.get('expected_text') != part['text']):
            raise ValueError('Recovered speech part lost its verified source binding')
        transcript = child.get('verified_transcript', child.get('transcript'))
        if not isinstance(transcript, str) or not transcript.strip():
            return report
        accepted_text.append(transcript.strip())
        proof.append({'start': part['start'], 'end': part['end'],
                      'accepted_sha256': row['accepted_sha256'], 'audio_sha256': row['audio_sha256'],
                      'transcript': transcript})
    expected, heard = ' '.join(accepted_text), report.get('transcript')
    if not isinstance(heard, str):
        return report
    if (compare_words(expected, heard)['wer'] == 0
            or word_spacing_equivalence(expected, heard) is not None
            or phonetic_equivalence(expected, heard) is not None):
        result = accepted_report(report, {'action': 'verified_parts', 'parts': proof})
        result['text_confirmation_method'] = 'verified_recovery_parts'
        return result
    return report
