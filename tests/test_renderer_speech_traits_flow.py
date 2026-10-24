from copy import deepcopy
from pathlib import Path
import tempfile
from threading import Lock
import unittest

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget, _render_chunk
from test_renderer_plain_delivery_integration import DirectedFixtureEngine, DIRECTION
from test_renderer_recurring_resolution import PlanClient
from test_renderer_speech_identity import paragraph_request, reference, speech_request
from test_renderer_speech_recovery_integration import RecoveryEngine, RecoveryVerifier, TEXT
from test_renderer_speech_traits import TRAITS
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for recovery integration")
class RendererSpeechTraitsFlowTests(unittest.TestCase):
    def test_paragraph_fallback_replanning_retains_current_traits_on_individual_passages(self):
        from audition_fixtures import prepare_document
        from test_contracts import valid_document
        from epub_tts_narrator.cli import export_bundle
        from epub_tts_narrator.renderer.book_context import resolve_book_config
        from epub_tts_narrator.renderer.bundle import load_bundle
        from epub_tts_narrator.renderer.paragraph_recovery import record_paragraph_fallback
        from epub_tts_narrator.renderer.speech_traits import performance_direction
        from epub_tts_narrator.renderer.workflow import make_render_plan

        document = valid_document()
        document["segments"][0].update(kind="dialogue", role_id="mara", voice_id="v1")
        document["cast"]["voices"][1]["speechTraits"] = TRAITS
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "book"
            export_bundle(prepare_document(document), directory)
            bundle = load_bundle(directory)
            config = resolve_book_config(bundle, deepcopy(DEFAULTS))
            initial = make_render_plan(bundle, config)
            paragraph = next(chunk for chunk in initial["chapters"][0]["chunks"]
                             if "paragraph_turns" in chunk)
            record_paragraph_fallback(config["render"]["work_dir"],
                                      {"kind": "paragraph_speech", "chunk": paragraph})

            recovered = make_render_plan(bundle, config)

            passages = [chunk for chunk in recovered["chapters"][0]["chunks"]
                        if chunk["segment"]["voice_id"] == "v1"]
            self.assertEqual(len(passages), 2)
            self.assertTrue(all("paragraph_turns" not in chunk for chunk in passages))
            self.assertEqual([performance_direction(chunk, "v1") for chunk in passages],
                             [TRAITS, TRAITS])

    def fixture(self):
        config = deepcopy(DEFAULTS)
        config["quality"] = validate_policy({})
        config["render"].update(
            max_attempts=1, sample_rate=44100, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        registry = {}
        return config, RecoveryEngine(registry), RecoveryVerifier(registry)

    def test_paragraph_and_simultaneous_lanes_receive_each_performers_composed_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            config, engine, verifier = self.fixture()
            first = reference("first")
            second = reference("second")
            voices = {first["voice_id"]: first, second["voice_id"]: second}
            paragraph = paragraph_request([
                ("Hello. ", first, "softly"),
                ("Again.", second, "quickly"),
            ])["chunk"]
            paragraph["chunk_id"] = "paragraph"
            paragraph["paragraph_turns"][0]["speech_traits"] = {"first": "slow cadence"}
            paragraph["paragraph_turns"][1]["speech_traits"] = {"second": "rapid cadence"}

            _render_chunk(
                paragraph, 1, 1, voices, engine, engine.identity, verifier, config, work, {}, Lock(),
                notify=lambda _: None, retry_budget=SpeechRetryBudget())
            self.assertEqual(
                [turn["direction"] for turn in engine.dialogue_calls[0]],
                ["slow cadence; softly", "rapid cadence; quickly"],
            )

            simultaneous = speech_request(first, "Together.")["chunk"]
            simultaneous["chunk_id"] = "simultaneous"
            simultaneous["segment"].update(role_id=None, voice_id=None, simultaneous_speakers=[
                {"role_id": "one", "voice_id": "first"},
                {"role_id": "two", "voice_id": "second"},
            ])
            simultaneous["speech_traits"] = {"first": "slow cadence", "second": "rapid cadence"}
            _render_chunk(
                simultaneous, 1, 1, voices, engine, engine.identity, verifier, config, work, {}, Lock(),
                notify=lambda _: None, retry_budget=SpeechRetryBudget())
            self.assertEqual(
                [call["direction"] for call in engine.calls[-2:]],
                ["slow cadence", "rapid cadence"],
            )

    def test_traits_reach_whole_and_split_generation_and_trait_changes_miss_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            config = deepcopy(DEFAULTS)
            config["quality"] = validate_policy({})
            config["render"].update(
                max_attempts=3, sample_rate=44100, ffmpeg=FFMPEG, ffprobe=FFPROBE)
            registry = {}
            engine = RecoveryEngine(registry)
            verifier = RecoveryVerifier(registry)
            voice = {"voice_id": "voice-1", "asset_hash": "reference-asset-1",
                     "kind": "reference", "reference_text": "Fixture voice reference.",
                     "reference_sha256": "1" * 64}
            voices = {voice["voice_id"]: voice}
            chunk = {
                "chunk_id": "chapter-6-segment-42", "spoken_text": TEXT,
                "silent_control": False,
                "source_ranges": [{"block_id": "b000123", "start": 7, "end": 7 + len(TEXT)}],
                "speech_traits": {"voice-1": TRAITS},
                "segment": {"id": "segment-42", "role_id": "insang", "voice_id": "voice-1",
                            "delivery": {"direction": "startled"}},
            }

            def render(value):
                return _render_chunk(
                    value, 1, 1, voices, engine, engine.identity, verifier, config, work, {}, Lock(),
                    notify=lambda _: None, retry_budget=SpeechRetryBudget())

            render(chunk)
            self.assertTrue(engine.calls)
            self.assertTrue(all(call["direction"] == TRAITS + "; startled" for call in engine.calls))
            calls = len(engine.calls)

            changed = deepcopy(chunk)
            changed["speech_traits"]["voice-1"] = "Specific Welsh accent; lilting rhythm"
            render(changed)
            self.assertGreater(len(engine.calls), calls)
            self.assertTrue(all(
                call["direction"] == "Specific Welsh accent; lilting rhythm; startled"
                for call in engine.calls[calls:]))

    def test_plain_delivery_drops_scene_direction_but_keeps_traits_and_reuses_accepted_take(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            config = deepcopy(DEFAULTS)
            config["quality"] = validate_policy({})
            config["render"].update(
                max_attempts=3, sample_rate=44100, ffmpeg=FFMPEG, ffprobe=FFPROBE)
            registry = {}
            engine = DirectedFixtureEngine(registry)
            engine.heard_override = lambda text, direction, attempt: (
                text if direction == TRAITS else
                (text.replace("Arrgh", "Oh"), text.replace("Arrgh", ""), text.replace("Arrgh", "Are"))[
                    (attempt - 1) % 3]
            )
            client = PlanClient([{
                "action": "plain_delivery", "edits": [],
                "reason": "The scene direction may obscure the requested words.",
            }])
            verifier = RecurringSpeechVerifier(
                RecoveryVerifier(registry), SpeechResolver(client),
            )
            voice = reference("character")
            voice_path = work / "reference.wav"
            write_tone(voice_path)
            voice.update(reference_audio=str(voice_path), reference_sha256=file_hash(voice_path))
            chunk = speech_request(voice, "Arrgh! The pain has returned.")["chunk"]
            chunk["segment"]["delivery"]["direction"] = DIRECTION
            chunk["speech_traits"] = {voice["voice_id"]: TRAITS}
            voices = {voice["voice_id"]: voice}
            budget = SpeechRetryBudget()

            def render():
                return _render_chunk(
                    chunk, 1, 1, voices, engine, engine.identity, verifier, config, work, {}, Lock(),
                    notify=lambda _: None, retry_budget=budget)

            result, accepted = render()
            self.assertEqual(
                [call["direction"] for call in engine.calls],
                [TRAITS + "; " + DIRECTION] * 3 + [TRAITS],
            )
            record = read_json(Path(result["wav_path"]).parent / "accepted.json")
            self.assertEqual(record["engine_metadata"]["direction"], TRAITS)
            calls = len(engine.calls)
            self.assertEqual(render(), (result, accepted))
            self.assertEqual(len(engine.calls), calls)
