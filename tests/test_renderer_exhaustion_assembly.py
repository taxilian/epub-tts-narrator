from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import assemble, render
from audition_fixtures import prepare_bundle
from test_renderer_clip_repair import BroadClient
import test_renderer_workflow as fixtures


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, 'FFmpeg and ffprobe required')
class ExhaustionAssemblyTests(unittest.TestCase):
    setUp = fixtures.RendererWorkflowTests.setUp

    def test_decisive_selection_survives_render_cache_and_model_free_assembly(self):
        directory = self.root / 'bundle'
        shutil.copytree(fixtures.EXAMPLE / 'bundle', directory, ignore=shutil.ignore_patterns('.*'))
        self.bundle = prepare_bundle(load_bundle(directory))
        client = BroadClient()
        client.plan = {'parts': [], 'reason': 'Use the best retained take.'}
        self.config['verification']['max_resolution_attempts'] = 3
        self.config['verification']['recurrence_threshold'] = 3
        first_text = self.bundle['chapters'][0]['rows'][0]['text']
        transcribe = self.verifier.transcribe

        def heard(path):
            result = transcribe(path)
            return fixtures.timed_transcript(first_text + ' not') if result['text'] == first_text else result

        recognizer = patch.object(self.verifier, 'transcribe', side_effect=heard)
        recognizer.start()
        self.addCleanup(recognizer.stop)
        verifier = RecurringSpeechVerifier(self.verifier, SpeechResolver(client))
        render(self.bundle, self.config, self.engine, verifier, notify=lambda _: None, chapters_only=True)
        work = Path(self.config['render']['work_dir'])
        accepted = [read_json(path) for path in (work / 'clips').glob('*/accepted.json')]
        selected = [row for row in accepted if row['report'].get('text_confirmation_method') == 'llm_exhaustion_selection']
        self.assertTrue(selected)
        self.assertTrue(all(not row['report']['original_failed_report']['passed'] for row in selected))
        calls = len(self.engine.speech_calls), len(client.selection_calls), len(self.verifier.calls)
        with patch.object(self.engine, 'synthesize', side_effect=AssertionError('cached render invoked TTS')):
            render(self.bundle, self.config, self.engine, verifier, notify=lambda _: None, chapters_only=True)
            result = assemble(self.bundle, self.config, notify=lambda _: None)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(list(Path(self.config['render']['output_dir']).glob('*.m4b')))
        self.assertEqual((len(self.engine.speech_calls), len(client.selection_calls), len(self.verifier.calls)), calls)
