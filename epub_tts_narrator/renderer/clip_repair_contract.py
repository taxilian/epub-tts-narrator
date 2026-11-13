from copy import deepcopy
import re
import unicodedata
from typing import TypedDict

from .common import digest
from .clip_repair_prompts import PLAN, REVIEW
from .engines import _directed_text
from .performance_compiler import earlier_direction_text
from .speech_text import trim_speech_text


class PlanPart(TypedDict):
    text: str
    delivery: str
    speech_traits: str


class ClipPlan(TypedDict):
    parts: list[PlanPart]
    reason: str


def uses_controlled_pronunciation(request):
    """Whether this request carries confirmed structured pronunciation controls."""
    if not isinstance(request, dict):
        return False
    chunk = request.get('chunk', {})
    records = []
    if request.get('kind') == 'speech' and isinstance(chunk, dict):
        records = [chunk.get('expressive_performance')]
    elif request.get('kind') == 'paragraph_speech' and isinstance(chunk, dict):
        records = [turn.get('expressive_performance') for turn in chunk.get('paragraph_turns', [])
                   if isinstance(turn, dict)]
    return any(isinstance(record, dict) and any(
        isinstance(control, dict) and control.get('kind') == 'pronunciation'
        for control in record.get('controls', [])) for record in records)


def synthesis_request(request):
    return request.get('synthesis_request', request) if request.get('kind') == 'speech_recovery_part' else request


def logical_text(request):
    return request['part']['text'] if request.get('kind') == 'speech_recovery_part' else request['chunk']['spoken_text']


def plain(value, maximum, *, empty=False):
    return (isinstance(value, str) and len(value) <= maximum and value == value.strip()
            and (empty or bool(value)) and not any(char in value for char in '[]<>`')
            and not any(unicodedata.category(char) in {'Cc', 'Cf', 'Cs', 'Zl', 'Zp'} for char in value))


def parse_plan(value, maximum=12) -> ClipPlan:
    if (type(maximum) is not int or maximum < 0
            or not isinstance(value, dict) or set(value) != {'parts', 'reason'}
            or not plain(value['reason'], 1000) or not isinstance(value['parts'], list)
            or len(value['parts']) > maximum):
        raise ValueError('Invalid bounded clip repair plan')
    parts: list[PlanPart] = []
    for part in value['parts']:
        if (not isinstance(part, dict) or set(part) != {'text', 'delivery', 'speech_traits'}
                or not plain(part['text'], 20000)
                or not plain(part['delivery'], 400, empty=True)
                or not plain(part['speech_traits'], 400, empty=True)):
            raise ValueError('Invalid clip-local synthesis inputs')
        _directed_text(part['text'], part_direction(part))
        parts.append({'text': part['text'], 'delivery': part['delivery'], 'speech_traits': part['speech_traits']})
    if sum(len(part['text']) for part in parts) > 20000:
        raise ValueError('Clip repair text exceeds its bound')
    return {'parts': parts, 'reason': value['reason']}


def part_direction(part):
    return '; '.join(value for value in (part['speech_traits'], part['delivery']) if value)


def policy(provider, request=None):
    result = {'name': 'broad_clip_repair', 'version': 1, 'provider': deepcopy(provider),
              'plan_sha256': digest(PLAN), 'review_sha256': digest(REVIEW), 'reviewers': 2,
              'required_approvals': 2, 'candidates': 1}
    if uses_controlled_pronunciation(request):
        # The prompt bytes and two-vote authorization are unchanged. Only a
        # request that actually carries protected controls gets the projection
        # identity; legacy plain repair records retain their exact old policy.
        result['controlled_pronunciation'] = {'version': 1, 'projection': 'exact_word_v1'}
    return result


def review_payload(payload, plan, reviewer):
    return {**deepcopy(payload), 'mode': 'clip_repair_review', 'parts': deepcopy(plan['parts']), 'reviewer': reviewer}


def approved(request, audit):
    if (not isinstance(audit, dict) or set(audit) != {'identity', 'payload', 'proposal', 'reviews'}
            or not isinstance(audit['identity'], dict) or not isinstance(audit['identity'].get('provider'), dict)
            or not isinstance(audit['payload'], dict) or not isinstance(audit['reviews'], list)
            or type(audit['payload'].get('remaining_starts')) is not int
            or audit['identity'] != policy(audit['identity'].get('provider'), request)
            or audit['payload'].get('request') != request
            or audit['payload'].get('request_sha256') != digest(request)
            or audit['payload'].get('mode') != 'clip_repair'):
        raise ValueError('Clip repair authority differs from the source request')
    plan = parse_plan(audit['proposal'], audit['payload']['remaining_starts'])
    if not plan['parts'] or len(audit['reviews']) != 2:
        raise ValueError('Clip repair needs two independent approvals')
    for number, review in enumerate(audit['reviews'], 1):
        if (not isinstance(review, dict) or set(review) != {'payload', 'response'}
                or review.get('payload') != review_payload(audit['payload'], plan, number)
                or not isinstance(review.get('response'), dict)
                or set(review['response']) != {'approve', 'reason'}
                or review['response']['approve'] is not True
                or not plain(review['response']['reason'], 1000)):
            raise ValueError('Clip repair lacks both blind approvals')
    return plan


def confirm_part(request, part, metadata, *, performance=None):
    if performance is None:
        text = trim_speech_text(part['text'])
        directed, control = _directed_text(text, part_direction(part))
        if (metadata.get('synthesis_text') not in (text, directed, earlier_direction_text(directed, part_direction(part)))
                or any(metadata.get(key) != control[key]
                       for key in ('direction', 'direction_applied', 'applied_control'))):
            raise ValueError('Actual clip repair inputs differ from the approved plan')
        return
    from .clip_repair_performance import repair_traits
    from .engines import prepare_synthesis

    traits = repair_traits(request)
    _, directed, control = prepare_synthesis(
        part['text'], part_direction(part), performance=performance, traits=traits)
    # A part recorded before each direction part became its own tag carries
    # that earlier engine text, also inside its compiler audit.
    earlier = earlier_direction_text(directed, traits.strip(), control['direction'])
    audit = control['expressive_performance']
    if metadata.get('synthesis_text') == earlier != directed:
        audit = dict(audit, text=earlier)
    if (metadata.get('synthesis_text') not in (directed, earlier)
            or any(metadata.get(key) != control[key]
                   for key in ('direction', 'direction_applied', 'applied_control'))
            or metadata.get('expressive_performance') != audit):
        raise ValueError('Actual controlled clip repair inputs differ from the approved plan')


def confirm_plan(request, marker):
    if (not isinstance(marker, dict) or set(marker) != {'audit', 'parts', 'audio_sha256'}
            or not isinstance(marker['audit'], dict) or not isinstance(marker['audit'].get('payload'), dict)
            or not isinstance(marker['audit']['payload'].get('request'), dict)
            or not isinstance(marker['parts'], list)
            or not re.fullmatch(r'[0-9a-f]{64}', str(marker['audio_sha256']))):
        raise ValueError('Invalid clip repair generation proof')
    original = marker['audit']['payload']['request']
    if request != original and request != synthesis_request(original):
        raise ValueError('Clip repair reference or source request changed')
    plan = approved(original, marker['audit'])
    if len(marker['parts']) != len(plan['parts']):
        raise ValueError('Incomplete clip repair generation provenance')
    from .clip_repair_performance import project_plan_performance, repair_traits

    projected = project_plan_performance(original, plan)
    for number, (part, evidence) in enumerate(zip(plan['parts'], marker['parts']), 1):
        fields = {'part', 'metadata', 'audio_sha256'}
        if projected[number - 1] is not None:
            fields |= {'performance', 'traits'}
        if (not isinstance(evidence, dict) or set(evidence) != fields
                or evidence['part'] != part or not isinstance(evidence['metadata'], dict)
                or not re.fullmatch(r'[0-9a-f]{64}', str(evidence['audio_sha256']))
                or evidence.get('performance') != projected[number - 1]
                or (projected[number - 1] is not None
                    and evidence.get('traits') != repair_traits(original))):
            raise ValueError('Clip repair part binding changed')
        confirm_part(original, part, evidence['metadata'],
                     performance=evidence.get('performance'))
    return plan


def confirm_plan_metadata(request, metadata):
    if not isinstance(metadata, dict) or set(metadata) != {'synthesis_text', 'clip_repair'}:
        raise ValueError('Unexpected clip repair generation fields')
    plan = confirm_plan(request, metadata['clip_repair'])
    if metadata['synthesis_text'] != ' '.join(part['text'] for part in plan['parts']):
        raise ValueError('Recorded joined clip repair text changed')
