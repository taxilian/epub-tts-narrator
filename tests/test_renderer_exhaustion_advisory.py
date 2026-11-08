from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.exhaustion_candidates import stylized_spelling_advisory
from epub_tts_narrator.renderer.exhaustion_selection import select_retained, validate_record
from test_renderer_exhaustion_candidates import ExhaustionCandidatesTests
from test_renderer_speech_identity import reference, speech_request


class StylizedSpellingAdvisoryTests(unittest.TestCase):
    def test_elongated_words_are_flagged_with_collapse_guidance(self):
        advisory = stylized_spelling_advisory(' Wheerrre issss sheeee?')
        self.assertEqual(advisory['elongated_words'], ['Wheerrre', 'issss', 'sheeee'])
        self.assertIn('collapsed form', advisory['guidance'])

    def test_ordinary_text_and_roman_numerals_carry_no_advisory(self):
        for text in ('Where is she?', 'Chapter VIII of XXIII', 'The mess in the hall.'):
            self.assertIsNone(stylized_spelling_advisory(text), text)


class ChooserClient:
    identity = {'model': 'offline-stylized-chooser', 'base_url': 'https://fixture.invalid'}

    def __init__(self):
        self.calls = []

    def complete(self, prompt, payload):
        self.calls.append(deepcopy(payload))
        return {'candidate_id': payload['candidates'][0]['candidate_id'],
                'reason': 'Fluent collapsed realization of stylized spelling.'}


class ExhaustionAdvisoryTests(ExhaustionCandidatesTests):
    def test_chooser_sees_stylized_spelling_advisory_for_elongated_passages(self):
        self.request = speech_request(reference(), 'Wheerrre issss sheeee?')
        self.candidate(1, transcript='Where is she?')
        client = ChooserClient()

        wav, record = select_retained(self.directory, self.request, client)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]['stylized_spelling']['elongated_words'],
                         ['Wheerrre', 'issss', 'sheeee'])
        self.assertTrue(record['report']['passed'])
        validate_record(record)
        saved = read_json(self.directory / 'exhaustion-selection.json')
        self.assertEqual(saved['audit']['payload']['stylized_spelling']['elongated_words'],
                         ['Wheerrre', 'issss', 'sheeee'])

    def test_plain_passages_send_no_advisory(self):
        self.candidate(1, transcript='Do leave.')
        client = ChooserClient()

        select_retained(self.directory, self.request, client)

        self.assertNotIn('stylized_spelling', client.calls[0])

    def test_legacy_pending_without_advisory_resumes_without_reasking(self):
        self.request = speech_request(reference(), 'Wheerrre issss sheeee?')
        self.candidate(1, transcript='Where is she?')
        first = ChooserClient()
        select_retained(self.directory, self.request, first)

        state = read_json(self.directory / 'exhaustion-selection.json')
        state['audit']['payload'].pop('stylized_spelling')
        state['record_sha256'] = digest({key: value for key, value in state.items()
                                         if key != 'record_sha256'})
        write_json(self.directory / 'exhaustion-selection.json', state)

        resumed = ChooserClient()
        wav, record = select_retained(self.directory, self.request, resumed)

        self.assertEqual(len(resumed.calls), 0)
        self.assertTrue(record['report']['passed'])
        validate_record(record)

    def test_pending_input_drift_beyond_the_advisory_still_fails(self):
        self.request = speech_request(reference(), 'Wheerrre issss sheeee?')
        self.candidate(1, transcript='Where is she?')
        select_retained(self.directory, self.request, ChooserClient())

        state = read_json(self.directory / 'exhaustion-selection.json')
        state['audit']['payload']['requested_text'] = 'Where is she?'
        state['record_sha256'] = digest({key: value for key, value in state.items()
                                         if key != 'record_sha256'})
        write_json(self.directory / 'exhaustion-selection.json', state)

        with self.assertRaises(ValueError):
            select_retained(self.directory, self.request, ChooserClient())


if __name__ == '__main__':
    unittest.main()
