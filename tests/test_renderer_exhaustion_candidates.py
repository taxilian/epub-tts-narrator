from copy import deepcopy
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.exhaustion_candidates import physical_audio, retained_candidates
from test_renderer_speech_identity import reference, speech_request
from test_renderer_workflow import write_tone


class ExhaustionCandidatesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.request = speech_request(reference(), 'Do not leave.')

    def candidate(self, number, request=None, transcript=None):
        request = self.request if request is None else request
        wav = self.directory / f'attempt-{number:03d}.wav'
        write_tone(wav, 200 + number * 30)
        record = {'request': deepcopy(request), 'request_hash': digest(request), 'attempt': number,
                  'audio_file': wav.name, 'audio_sha256': file_hash(wav), 'engine_metadata': {},
                  'report': {'passed': False, 'reasons': ['text_mismatch'], 'transcript': transcript}}
        write_json(self.directory / f'attempt-{number:03d}.json', record)
        return wav

    def test_every_bound_full_span_is_nominated_without_word_or_syllable_veto(self):
        self.candidate(1, transcript='Do leave.')
        self.candidate(2, transcript=None)
        self.candidate(3, transcript='a ' * 100)
        child = {'kind': 'speech_recovery_part', 'part': {'start': 0, 'end': 2, 'text': 'Do'},
                 'synthesis_request': self.request}
        self.candidate(4, request=child, transcript='Do')
        unrelated = dict(self.request, voice_asset_hash=digest('unrelated character'))
        self.candidate(5, request=unrelated, transcript='Do not leave.')
        found = retained_candidates(self.directory, self.request)
        self.assertEqual([item['record']['attempt'] for item in found], [1, 2, 3])
        self.assertFalse(found[1]['summary']['warnings']['transcript_available'])
        self.assertEqual(found[0]['summary']['warnings']['negation'], [['not'], []])

    def test_a_take_made_by_a_repair_carries_its_audit_only_as_a_digest(self):
        from epub_tts_narrator.renderer.exhaustion_candidates import candidate_summary
        audit = {'payload': {'failures': [{'actual_inputs': {'clip_repair': {'audit': {'payload': 'x' * 10000}}}}]}}
        record = {'request_hash': 'r', 'attempt': 1, 'audio_sha256': 'a', 'report': {},
                  'engine_metadata': {'synthesis_text': 'Do not leave.', 'clip_repair': {'audit': audit, 'parts': [1]}}}
        summary = candidate_summary(self.request, record, {'duration_seconds': 1.0})
        self.assertEqual(summary['actual_inputs'], {'synthesis_text': 'Do not leave.',
                                                    'clip_repair': {'audit': {'audit_sha256': digest(audit)}, 'parts': [1]}})
        legacy = candidate_summary(self.request, record, {'duration_seconds': 1.0}, compact=False)
        self.assertEqual(legacy['actual_inputs'], record['engine_metadata'])

    def test_sibling_folders_are_found_and_a_changed_folder_is_reread(self):
        clips = self.directory / 'clips'
        own, sibling = clips / ('a' * 64), clips / ('b' * 64)
        own.mkdir(parents=True)
        sibling.mkdir()
        self.directory = sibling
        child = dict(self.request, recovery={'kind': 'join'})
        self.candidate(1, request=child, transcript='Do not leave.')
        self.directory = own
        self.assertEqual([item['record']['attempt'] for item in retained_candidates(own, self.request)], [1])
        self.directory = sibling
        self.candidate(2, request=child, transcript='Do not leave.')
        self.assertEqual([item['record']['attempt'] for item in retained_candidates(own, self.request)], [1, 2])

    def test_empty_silent_corrupt_and_nonfinite_audio_are_not_usable(self):
        for name, frames in [('empty', b''), ('silent', bytes(88200))]:
            path = self.directory / (name + '.wav')
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 44100, 0, 'NONE', ''))
                audio.writeframes(frames)
            self.assertIsNone(physical_audio(path))
        corrupt = self.directory / 'corrupt.wav'
        write_tone(corrupt)
        corrupt.write_bytes(corrupt.read_bytes()[:-100])
        self.assertIsNone(physical_audio(corrupt))
        nonfinite = self.directory / 'nonfinite.wav'
        data = struct.pack('<f', float('nan')) * 100
        fmt = struct.pack('<HHIIHH', 3, 1, 44100, 176400, 4, 32)
        nonfinite.write_bytes(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt '
                             + struct.pack('<I', 16) + fmt + b'data' + struct.pack('<I', len(data)) + data)
        self.assertIsNone(physical_audio(nonfinite))

    def test_clipped_nonempty_audio_is_ranking_evidence_not_an_exclusion(self):
        path = self.directory / 'clipped.wav'
        with wave.open(str(path), 'wb') as audio:
            audio.setparams((1, 2, 44100, 0, 'NONE', ''))
            audio.writeframes(struct.pack('<h', 32767) * 44100)
        self.assertGreater(physical_audio(path)['peak_dbfs'], -1)
