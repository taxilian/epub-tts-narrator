import contextlib
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from audition_fixtures import prepare_document
from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.name_pronunciations import (
    FILENAME, LABEL_VERSION, load_local, save_decisions, save_definition, tagging_state,
)
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.series import link_analysis
from shared_fixtures import renderer_config
from test_name_pronunciations import FakeModel, ROSE, homonym, rose_document
from test_series import fixture, propose


class RenderPronunciationPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "settings.toml"
        self.config.write_text('[provider]\nmodel="fixture"\nbase_url="http://127.0.0.1:9/v1"\n'
                               '[analysis]\nattribution_workers=1\nrepair_attempts=0\n'
                               '[engine]\ncheckpoint="./tts-model"\n'
                               '[verification]\nmodel="./asr-model"\n'
                               '[render]\nrequire_cover=false\nparagraph_synthesis=false\n')
        self.books = [self.root / "first", self.root / "second"]
        for directory in self.books:
            export_bundle(prepare_document(rose_document()), directory)
            save_definition(directory, "Rose", ROSE, status="confirmed", source="manual")

    def command(self, *books, model, command="render"):
        output = StringIO()
        with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(output),
              patch("epub_tts_narrator.provider.OpenAICompatibleClient", return_value=model) as provider,
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()),
              patch("epub_tts_narrator.renderer.workflow.render", return_value={"chapters": []}) as render):
            status = main([command, *map(str, books), "--config", str(self.config),
                           *(["--chapters-only"] if command == "render" else [])])
        return status, output.getvalue(), provider, render

    def phones(self, directory):
        plan = make_plan(load_bundle(directory), renderer_config(paragraph_synthesis=False))
        return [control["phonemes"] for chapter in plan["chapters"] for chunk in chapter["chunks"]
                for control in chunk.get("expressive_performance", {}).get("controls", [])
                if control["kind"] == "pronunciation"]

    def test_single_and_queued_render_tag_confirmed_words_before_planning(self):
        for count in (1, 2):
            with self.subTest(books=count):
                model = FakeModel(decide=homonym)
                status, output, _, render = self.command(*self.books[:count], model=model)
                self.assertEqual(status, 0, output)
                self.assertEqual(render.call_count, count)
                for directory in self.books[:count]:
                    self.assertEqual(self.phones(directory), [ROSE, ROSE + " Z", ROSE, ROSE + " Z"])
                self.assertEqual({payload["mode"] for payload in model.payloads}, {"word_part_of_speech"})

    def test_current_tags_render_without_provider_configuration_or_registry_writes(self):
        status, output, _, _ = self.command(self.books[0], model=FakeModel(decide=homonym))
        self.assertEqual(status, 0, output)
        before = (self.books[0] / FILENAME).read_bytes()
        self.config.write_text(self.config.read_text().replace(
            '[provider]\nmodel="fixture"\nbase_url="http://127.0.0.1:9/v1"\n', ''))
        status, output, provider, render = self.command(self.books[0], model=FakeModel())
        self.assertEqual(status, 0, output)
        self.assertTrue(render.called)
        provider.assert_not_called()
        self.assertEqual((self.books[0] / FILENAME).read_bytes(), before)

    def test_unsure_tags_are_checkpointed_and_stop_render_without_retrying_them(self):
        model = FakeModel(decide=lambda text: "unsure")
        status, output, _, render = self.command(self.books[0], model=model)
        self.assertEqual(status, 2, output)
        self.assertIn("0 untagged, 5 unsure", output)
        render.assert_not_called()
        self.assertTrue(load_local(self.books[0])["mentions"]["rose"])
        status, output, provider, render = self.command(self.books[0], model=FakeModel())
        self.assertEqual(status, 2, output)
        provider.assert_not_called()
        render.assert_not_called()

    def test_plan_leaves_missing_tags_untouched_and_never_calls_the_provider(self):
        before = (self.books[0] / FILENAME).read_bytes()
        status, output, provider, _ = self.command(self.books[0], model=FakeModel(), command="plan")
        self.assertEqual(status, 2, output)
        self.assertIn("5 untagged", output)
        provider.assert_not_called()
        self.assertEqual((self.books[0] / FILENAME).read_bytes(), before)

    def test_render_refreshes_outdated_labels_but_preserves_person_decisions(self):
        directory = self.books[0]
        status, output, _, _ = self.command(directory, model=FakeModel(decide=homonym))
        self.assertEqual(status, 0, output)
        source = load_bundle(directory)["source"]
        occurrence = tagging_state(directory, source, "Rose")["occurrences"][0]
        decision = {key: occurrence[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
        save_decisions(directory, source, "Rose", [{**decision, "basis": "user", "use": False}])
        model = FakeModel(decide=homonym)
        with (patch("epub_tts_narrator.name_pronunciations.LABEL_VERSION", LABEL_VERSION + 1),
              patch("epub_tts_narrator.name_pronunciation_model.LABEL_VERSION", LABEL_VERSION + 1)):
            status, output, _, _ = self.command(directory, model=model)
            self.assertEqual(status, 0, output)
            self.assertEqual(self.phones(directory), [ROSE + " Z", ROSE, ROSE + " Z"])
        self.assertEqual(sum(len(payload["occurrences"]) for payload in model.payloads), 4)
        saved = load_local(directory)["mentions"]["rose"]
        self.assertEqual([(item["basis"], item["use"]) for item in saved if item["basis"] == "user"],
                         [("user", False)])

    def test_render_tags_inherited_readings_without_writing_the_parent(self):
        first, second = self.root / "series-first", self.root / "series-second"
        document = fixture()
        registry = fresh_registry(first)
        publish_directory(prepare_document(link_analysis(document, registry)), registry, first, export_bundle)
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        before = (first / FILENAME).read_bytes()
        document = fixture(sequel=True)
        registry = continuation_registry(first, document["source"])
        propose(document, load_bundle(first)["cast"]["characters"][0]["id"])
        publish_directory(prepare_document(link_analysis(document, registry)), registry, second, export_bundle,
                          continue_from=first)
        status, output, _, _ = self.command(second, model=FakeModel())
        self.assertEqual(status, 0, output)
        self.assertEqual(self.phones(second), ["M AA1 R AH0"])
        self.assertEqual((first / FILENAME).read_bytes(), before)

    def test_provider_failure_stops_before_synthesis_and_rerender_resumes_tagging(self):
        status, output, _, render = self.command(self.books[0], model=FakeModel(fail_after=0))
        self.assertEqual(status, 2, output)
        self.assertIn("provider stopped", output)
        render.assert_not_called()
        status, output, _, _ = self.command(self.books[0], model=FakeModel(decide=homonym))
        self.assertEqual(status, 0, output)
        self.assertEqual(self.phones(self.books[0]), [ROSE, ROSE + " Z", ROSE, ROSE + " Z"])
