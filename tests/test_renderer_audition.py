"""Terminal casting behavior and playback boundaries without audio hardware."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from audiobook_renderer.audition import AudioPlayer, run_casting
from audiobook_renderer.cli import main
from audiobook_renderer.config import load_config
from audiobook_renderer.voices import VoiceStore
from epub_audiobook.cli import export_bundle
from test_contracts import valid_document
from audition_fixtures import prepare_document


class SessionDouble:
    def __init__(self):
        self.config = {"engine": {"type": "drama3_mlx"}}
        self.events = []
        self.data = [
            {"voice_id": "v1", "characters": ["Mara"], "roles": ["mara"],
             "prompt": "A gentle voice.", "reference_text": "Hello there.",
             "selected": None, "candidates": []},
            {"voice_id": "vn", "characters": [], "roles": ["narrator"],
             "prompt": "A measured voice.", "reference_text": "Hello there.",
             "selected": None, "candidates": []},
        ]
        self.generation_error = None

    def rows(self):
        return deepcopy(self.data)

    def row(self, voice_id):
        return next(row for row in self.data if row["voice_id"] == voice_id)

    def generate(self, voice_id):
        self.events.append(("generate", voice_id))
        if self.generation_error:
            error, self.generation_error = self.generation_error, None
            raise error
        row = self.row(voice_id)
        candidate = {
            "candidate_id": f"{voice_id}-take-{len(row['candidates']) + 1}",
            "reference_audio": f"/fixtures/{voice_id}-{len(row['candidates']) + 1}.wav",
            "reference_text": row["reference_text"],
            "description_applied": False,
            "creation": {"requested_description": row["prompt"]},
        }
        row["candidates"].append(candidate)
        return deepcopy(candidate)

    def update_prompt(self, voice_id, **values):
        self.events.append(("edit", voice_id, dict(values)))
        self.row(voice_id).update(values)

    def accept(self, voice_id, candidate_id):
        self.events.append(("accept", voice_id, candidate_id))
        row = self.row(voice_id)
        row["selected"] = next(candidate for candidate in row["candidates"]
                               if candidate["candidate_id"] == candidate_id)
        for candidate in row["candidates"]:
            candidate["selected"] = candidate["candidate_id"] == candidate_id
        return deepcopy(row["selected"])

    def ensure_ready(self, voice_id):
        self.events.append(("ensure_ready", voice_id))
        row = self.row(voice_id)
        if row["selected"]:
            return deepcopy(row["selected"])
        candidate = row["candidates"][-1] if row["candidates"] else self.generate(voice_id)
        return self.accept(voice_id, candidate["candidate_id"])

    def regenerate(self, voice_id, *, voice_designer=None):
        if voice_designer is not None:
            self.events.append(("designer", voice_designer))
        self.events.append(("regenerate", voice_id))
        candidate = self.generate(voice_id)
        return self.accept(voice_id, candidate["candidate_id"])

    def generate_missing(self):
        return [self.ensure_ready(row["voice_id"]) for row in self.data
                if row["selected"] is None]


class SharedSessionDouble(SessionDouble):
    """Expose two roles for one reference, plus a separate same-named character."""

    def __init__(self):
        super().__init__()
        canonical = self.row("v1")
        canonical["label"] = "Mara"
        canonical["canonical_voice_id"] = "v1"
        narrator = self.row("vn")
        narrator["label"] = "Mara (narration)"
        narrator["canonical_voice_id"] = "v1"
        other = dict(deepcopy(canonical), voice_id="v2", canonical_voice_id="v2")
        self.data = [narrator, canonical, other]
        for voice_id in ("v1", "v2"):
            self.ensure_ready(voice_id)
        self.events.clear()
        self.close = Mock()

    def canonical(self, voice_id):
        return self.row(voice_id).get("canonical_voice_id", voice_id)

    def rows(self):
        rows = super().rows()
        for row in rows:
            canonical = self.row(self.canonical(row["voice_id"]))
            for key in ("selected", "candidates", "prompt", "reference_text"):
                row[key] = deepcopy(canonical[key])
        return rows

    def ensure_ready(self, voice_id):
        return super().ensure_ready(self.canonical(voice_id))

    def update_prompt(self, voice_id, **values):
        self.events.append(("edit", voice_id, dict(values)))
        self.row(self.canonical(voice_id)).update(values)

    def regenerate(self, voice_id, *, voice_designer=None):
        if voice_designer is not None:
            self.events.append(("designer", voice_designer))
        self.events.append(("regenerate", voice_id))
        canonical = self.canonical(voice_id)
        candidate = self.generate(canonical)
        return self.accept(canonical, candidate["candidate_id"])


class PlayerDouble:
    def __init__(self):
        self.played = []
        self.playing = False
        self.stop_count = 0

    def start(self, path):
        self.played.append(path)
        self.playing = True

    def stop(self):
        self.playing = False
        self.stop_count += 1

    def poll(self):
        return self.playing


class CastingTerminalTests(unittest.TestCase):
    def setUp(self):
        self.session = SessionDouble()
        self.player = PlayerDouble()
        self.output = []
        self.prompts = []

    def playback(self, player, path):
        player.start(path)
        # The terminal helper consumes a stop key (or natural completion).
        # Returning here must redisplay the only menu, with no extra input.
        player.stop()

    def run_ui(self, *answers, playback_fn=None):
        pending = iter(answers)

        def answer(prompt):
            self.prompts.append(prompt)
            try:
                value = next(pending)
            except StopIteration:
                raise EOFError from None
            if isinstance(value, BaseException):
                raise value
            return value

        run_casting(self.session, self.player, input_fn=answer, output=self.output.append,
                    playback_fn=playback_fn or self.playback)

    def ready(self):
        for row in self.session.rows():
            self.session.ensure_ready(row["voice_id"])
        self.session.events.clear()

    def test_startup_prepares_book_script_before_generation_and_g_shows_it(self):
        script = "The river rose around the bridge while we waited for the others."

        def prepare():
            self.session.events.append(("prepare_samples",))
            row = self.session.row("v1")
            row["reference_text"] = script
            row["sample_source"] = {"kind": "book_excerpt", "chapter_title": "Chapter 4"}

        self.session.prepare_samples = prepare
        self.run_ui("g1", "", "", "q")
        self.assertEqual(self.session.events[0], ("prepare_samples",))
        self.assertEqual(self.session.row("v1")["candidates"][0]["reference_text"], script)
        self.assertEqual(self.session.row("v1")["selected"]["reference_text"], script)
        self.assertIn("Sample script: " + script, self.output)
        self.assertIn("From: Chapter 4", self.output)

    def test_fallback_script_is_explained_without_an_extra_menu_step(self):
        self.ready()
        self.session.row("v1")["sample_source"] = {"kind": "fallback"}
        self.run_ui("g1", "", "", "q")
        self.assertTrue(any("Fallback script:" in line for line in self.output))
        self.assertEqual(len(self.prompts), 4)

    def test_number_plays_immediately_then_returns_to_menu_without_extra_steps(self):
        self.run_ui("1", "2", "Q")
        self.assertEqual(self.player.played, ["/fixtures/v1-1.wav", "/fixtures/vn-1.wav"])
        self.assertEqual(len(self.prompts), 3)
        self.assertFalse(self.player.playing)
        self.assertTrue(all(row["selected"] is not None for row in self.session.data))
        menus = [line for line in self.output if "Voice casting" in line]
        self.assertGreaterEqual(len(menus), 3)
        self.assertNotIn("[a]", "\n".join(self.output).lower())
        self.assertNotIn("needs acceptance", "\n".join(self.output).lower())

    def test_generate_edits_both_fields_once_and_automatically_selects_and_plays(self):
        self.run_ui("G1", "A warm, older voice.", "The river is quiet.", "q")
        row = self.session.row("v1")
        self.assertEqual(len(row["candidates"]), 2)
        self.assertEqual(self.player.played, ["/fixtures/v1-2.wav"])
        self.assertEqual(row["candidates"][0]["reference_text"], "Hello there.")
        self.assertEqual(row["candidates"][0]["creation"]["requested_description"], "A gentle voice.")
        self.assertEqual(row["selected"]["reference_text"], "The river is quiet.")
        self.assertEqual(row["selected"]["creation"]["requested_description"], "A warm, older voice.")
        self.assertEqual(row["selected"]["candidate_id"], "v1-take-2")
        self.assertEqual([event for event in self.session.events if event[0] == "edit"], [
            ("edit", "v1", {"prompt": "A warm, older voice.", "reference_text": "The river is quiet."}),
        ])
        self.assertEqual([event for event in self.session.events if event[0] == "regenerate"],
                         [("regenerate", "v1")])
        self.assertEqual(len(self.prompts), 4)
        self.assertFalse(self.player.playing)

    def test_plain_g_asks_for_character_and_blank_edits_keep_current_text(self):
        self.run_ui("g", "2", "", "", "q")
        row = self.session.row("vn")
        self.assertEqual(row["selected"]["candidate_id"], "vn-take-2")
        self.assertEqual(row["selected"]["reference_text"], "Hello there.")
        self.assertEqual(self.player.played, ["/fixtures/vn-2.wav"])
        self.assertFalse(any(event[0] == "edit" for event in self.session.events))
        self.assertEqual(len(self.prompts), 5)

    def test_plain_g_defaults_to_last_played_character(self):
        self.run_ui("2", "G", "", "", "q")
        self.assertEqual(self.player.played, ["/fixtures/vn-1.wav", "/fixtures/vn-2.wav"])
        self.assertEqual([event for event in self.session.events if event[0] == "regenerate"],
                         [("regenerate", "vn")])
        self.assertEqual(self.session.row("v1")["selected"]["candidate_id"], "v1-take-1")
        self.assertEqual(len(self.prompts), 5)

    def test_g_with_spaced_number_edits_only_provided_field(self):
        self.run_ui("G 1", "", "A new sample.", "q")
        self.assertEqual([event for event in self.session.events if event[0] == "edit"], [
            ("edit", "v1", {"reference_text": "A new sample."}),
        ])
        self.assertEqual(self.session.row("v1")["selected"]["creation"]["requested_description"],
                         "A gentle voice.")
        self.assertEqual(self.player.played, ["/fixtures/v1-2.wav"])

    def test_gg_selects_gemini_inside_menu_and_edits_selects_and_plays(self):
        self.ready()
        original_config = deepcopy(self.session.config)
        self.run_ui("GG 1", "A bright voice.", "A different sample.", "q")
        self.assertEqual([event for event in self.session.events if event[0] == "designer"],
                         [("designer", "gemini")])
        self.assertEqual(self.session.row("v1")["selected"]["reference_text"], "A different sample.")
        self.assertEqual(self.session.row("v1")["selected"]["creation"]["requested_description"],
                         "A bright voice.")
        self.assertEqual(self.player.played, ["/fixtures/v1-2.wav"])
        self.assertEqual(self.session.config, original_config)
        self.assertEqual(len(self.prompts), 4)
        self.assertTrue(any("GG [number]" in line and "Gemini" in line for line in self.output))

    def test_compact_gg_then_g_choose_gemini_then_local(self):
        self.ready()
        self.run_ui("gg1", "", "", "G 1", "", "", "q")
        self.assertEqual([event for event in self.session.events if event[0] == "designer"],
                         [("designer", "gemini"), ("designer", "local")])
        self.assertEqual(self.player.played, ["/fixtures/v1-2.wav", "/fixtures/v1-3.wav"])

    def test_plain_gg_defaults_to_last_played_character(self):
        self.ready()
        self.run_ui("2", "GG", "", "", "q")
        self.assertEqual([event for event in self.session.events if event[0] == "designer"],
                         [("designer", "gemini")])
        self.assertEqual(self.player.played, ["/fixtures/vn-1.wav", "/fixtures/vn-2.wav"])
        self.assertEqual(len(self.prompts), 5)

    def test_invalid_gg_numbers_do_not_edit_or_generate(self):
        self.ready()
        self.run_ui("GG 0", "GG 99", "GGG 1", "1", "q")
        self.assertFalse(any(event[0] in {"edit", "regenerate", "designer"}
                             for event in self.session.events))
        self.assertEqual(self.player.played, ["/fixtures/v1-1.wav"])

    def test_failed_gg_keeps_existing_voice_and_returns_to_menu(self):
        self.ready()
        self.session.generation_error = RuntimeError("Gemini API request failed")
        self.run_ui("GG 1", "", "", "1", "q")
        self.assertTrue(any("Gemini API request failed" in line for line in self.output))
        self.assertEqual(self.session.row("v1")["selected"]["candidate_id"], "v1-take-1")
        self.assertEqual(self.player.played, ["/fixtures/v1-1.wav"])

    def test_shared_role_plays_immediately_and_g_updates_only_its_character(self):
        self.session = SharedSessionDouble()
        unrelated = deepcopy(self.session.row("v2"))
        self.run_ui("1", "G", "A clear voice.", "", "q")
        self.assertEqual(self.player.played, ["/fixtures/v1-1.wav", "/fixtures/v1-2.wav"])
        self.assertEqual([event for event in self.session.events if event[0] == "regenerate"],
                         [("regenerate", "vn")])
        rows = {row["voice_id"]: row for row in self.session.rows()}
        self.assertEqual(rows["vn"]["selected"], rows["v1"]["selected"])
        self.assertEqual(rows["v1"]["selected"]["candidate_id"], "v1-take-2")
        self.assertEqual(rows["vn"]["prompt"], "A clear voice.")
        self.assertEqual(self.session.row("v2"), unrelated)
        self.assertTrue(any("Mara (narration) — shares Mara" in line for line in self.output))
        self.assertTrue(any("Regeneration updates every linked role" in line
                            for line in self.output))
        self.assertEqual(len(self.prompts), 5)
        self.assertFalse(any("accept" in prompt.lower() or "confirm" in prompt.lower()
                             for prompt in self.prompts))

    def test_plain_g_without_last_character_can_cancel_with_blank(self):
        self.run_ui("g", "", "1", "q")
        self.assertFalse(any(event[0] == "regenerate" for event in self.session.events))
        self.assertEqual(self.player.played, ["/fixtures/v1-1.wav"])
        self.assertEqual(len(self.prompts), 4)

    def test_failed_regeneration_preserves_current_voice_and_menu_remains_available(self):
        self.ready()
        self.session.generation_error = RuntimeError("The transcript did not match.")
        self.run_ui("G1", "", "", "1", "q")
        self.assertTrue(any("The transcript did not match." in line for line in self.output))
        self.assertEqual(self.session.row("v1")["selected"]["candidate_id"], "v1-take-1")
        self.assertEqual(self.player.played, ["/fixtures/v1-1.wav"])
        self.assertFalse(self.player.playing)

    def test_invalid_character_numbers_and_commands_leave_menu_usable(self):
        self.run_ui("0", "99", "G 99", "unknown", "1", "q")
        self.assertEqual(self.player.played, ["/fixtures/v1-1.wav"])
        self.assertFalse(any(event[0] == "regenerate" for event in self.session.events))
        self.assertTrue(any("voice number from the menu" in line.lower()
                            for line in self.output))
        self.assertEqual(len(self.prompts), 6)

    def test_eof_stops_playback_and_keeps_generated_selected_voice(self):
        self.run_ui("1")
        self.assertFalse(self.player.playing)
        self.assertEqual(self.session.row("v1")["selected"]["candidate_id"], "v1-take-1")
        self.assertTrue(any("Casting closed" in line for line in self.output))

    def test_interrupt_stops_playback_and_keeps_current_voice(self):
        def interrupted(player, path):
            player.start(path)
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_ui("1", playback_fn=interrupted)
        self.assertFalse(self.player.playing)
        self.assertEqual(self.session.row("v1")["selected"]["candidate_id"], "v1-take-1")

    def test_failed_playback_returns_to_menu_and_can_play_another_voice(self):
        def failing_once(player, path):
            if path.endswith("v1-1.wav"):
                raise ValueError("Audio playback failed")
            self.playback(player, path)

        self.run_ui("1", "2", "q", playback_fn=failing_once)
        self.assertTrue(any("Audio playback failed" in line for line in self.output))
        self.assertEqual(self.player.played, ["/fixtures/vn-1.wav"])

    def test_metadata_cannot_write_terminal_escape_sequences(self):
        self.session.row("v1")["characters"] = ["Mara\x1b[2J"]
        self.run_ui("q")
        self.assertNotIn("\x1b", "\n".join(self.output))

    def test_startup_generates_and_selects_all_missing_samples_before_first_input(self):
        def first_input(prompt):
            self.assertEqual([event for event in self.session.events if event[0] == "generate"],
                             [("generate", "v1"), ("generate", "vn")])
            self.assertTrue(all(len(row["candidates"]) == 1 for row in self.session.data))
            self.assertTrue(all(row["selected"] is not None for row in self.session.data))
            return "q"

        run_casting(self.session, self.player, input_fn=first_input, output=self.output.append,
                    playback_fn=self.playback)

    def test_external_verification_outage_stops_preparing_more_voices_but_keeps_menu(self):
        from audiobook_renderer.workflow import VerificationUnavailable
        self.session.generation_error = VerificationUnavailable("JEV confirmation unavailable; saved audio retained.")
        self.run_ui("q")
        self.assertEqual([event for event in self.session.events if event[0] == "generate"],
                         [("generate", "v1")])
        self.assertTrue(any("Voice preparation paused" in line for line in self.output))
        self.assertTrue(any("Voice casting" in line for line in self.output))
        self.assertIsNone(self.session.row("vn")["selected"])

    def test_startup_reuses_active_voices_and_selects_saved_pending_sample_without_generation(self):
        self.session.generate("v1")
        self.session.accept("v1", "v1-take-1")
        self.session.generate("vn")
        self.session.events.clear()
        with patch.object(self.session, "generate", side_effect=AssertionError("Loaded model")):
            self.run_ui("q")
        self.assertEqual(self.session.row("v1")["selected"]["candidate_id"], "v1-take-1")
        self.assertEqual(self.session.row("vn")["selected"]["candidate_id"], "vn-take-1")
        self.assertFalse(any(event[0] == "generate" for event in self.session.events))

    def test_all_twenty_six_pending_samples_become_active_on_startup(self):
        template = self.session.data[0]
        self.session.data = [dict(deepcopy(template), voice_id=f"voice-{index}",
                                  characters=[f"Character {index}"])
                             for index in range(26)]
        for row in self.session.data:
            self.session.generate(row["voice_id"])
        self.session.events.clear()
        with patch.object(self.session, "generate", side_effect=AssertionError("Loaded model")):
            self.run_ui("q")
        self.assertEqual(sum(row["selected"] is not None for row in self.session.data), 26)
        self.assertEqual(sum(event[0] == "accept" for event in self.session.events), 26)

    def test_repeated_launch_preserves_active_voices_without_regeneration(self):
        self.run_ui("q")
        initial = deepcopy(self.session.data)
        self.session.events.clear()
        self.run_ui("q")
        self.assertEqual(self.session.data, initial)
        self.assertFalse(any(event[0] in {"generate", "accept", "regenerate"}
                             for event in self.session.events))

    def test_failed_startup_voice_does_not_prevent_playback_or_regeneration(self):
        self.session.generation_error = RuntimeError("The transcript did not match.")
        self.run_ui("2", "G1", "", "", "q")
        self.assertTrue(any("The transcript did not match." in line for line in self.output))
        self.assertEqual(self.player.played, ["/fixtures/vn-1.wav", "/fixtures/v1-1.wav"])
        self.assertTrue(all(row["selected"] is not None for row in self.session.data))

    def test_number_for_voice_with_no_sample_does_not_start_playback(self):
        self.session.generation_error = RuntimeError("The transcript did not match.")
        self.run_ui("1", "2", "q")
        self.assertEqual(self.player.played, ["/fixtures/vn-1.wav"])
        self.assertIsNone(self.session.row("v1")["selected"])
        self.assertTrue(any("sample" in line.lower() and "G" in line for line in self.output))


class CastingCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config_path = self.root / "renderer.toml"
        self.base_config = (
            '[engine]\ncheckpoint="./tts-model"\n'
            '[verification]\nmodel="./asr-model"\n'
            '[voices]\nseries_id="test-series"\n'
        )
        self.config_path.write_text(self.base_config)
        self.bundle = self.root / "bundle"
        export_bundle(prepare_document(valid_document()), self.bundle)

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["cast", str(self.bundle), "--config", str(self.config_path), *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def publish_fixture(self):
        # Playback is mocked; the bytes are only an immutable asset-store fixture.
        sample = self.root / "fixture.wav"
        sample.write_bytes(b"test-only voice asset")
        store = VoiceStore(self.root / "voices", "test-series")
        lock = store.lock()
        try:
            return store.publish("v1", valid_document()["cast"]["voices"][1], sample,
                                 "Hello there.", {"engine": "fixture", "model": "fixture"},
                                 {"passed": True}, {"description_applied": False})
        finally:
            lock.rmdir()

    def test_list_never_initializes_tts_or_verifier_and_creates_no_store(self):
        with patch("audiobook_renderer.engines.create_engine") as engine, \
                patch("audiobook_renderer.verification.create_verifier") as verifier:
            code, output, error = self.run_cli("--action", "list")
        self.assertEqual(code, 0, error)
        self.assertEqual({row["voice_id"] for row in json.loads(output)}, {"vn", "v1"})
        engine.assert_not_called()
        verifier.assert_not_called()
        self.assertFalse((self.root / "voices").exists())

    def test_play_existing_asset_never_initializes_models(self):
        saved = self.publish_fixture()
        with patch("audiobook_renderer.engines.create_engine") as engine, \
                patch("audiobook_renderer.verification.create_verifier") as verifier, \
                patch("audiobook_renderer.audition.AudioPlayer") as player:
            code, output, error = self.run_cli("--action", "play", "--voice-id", "v1")
        self.assertEqual(code, 0, error)
        engine.assert_not_called()
        verifier.assert_not_called()
        player.return_value.start.assert_called_once_with(saved["reference_audio"])
        player.return_value.wait.assert_called_once_with()
        player.return_value.stop.assert_called_once_with()
        self.assertIn("legacy", output)

    def test_generate_action_immediately_selects_replacement_without_accept_step(self):
        session = SessionDouble()
        session.ensure_ready("v1")
        session.events.clear()
        session.close = Mock()
        with patch("audiobook_renderer.casting.CastingSession", return_value=session):
            code, output, error = self.run_cli("--action", "generate", "--voice-id", "v1")
        self.assertEqual(code, 0, error)
        self.assertEqual(session.row("v1")["selected"]["candidate_id"], "v1-take-2")
        self.assertEqual([event for event in session.events if event[0] == "regenerate"],
                         [("regenerate", "v1")])
        self.assertNotIn("accept it", output)
        session.close.assert_called_once_with()

    def test_generate_missing_action_activates_pending_and_new_samples(self):
        session = SessionDouble()
        session.generate("v1")
        session.events.clear()
        session.close = Mock()
        with patch("audiobook_renderer.casting.CastingSession", return_value=session):
            code, output, error = self.run_cli("--action", "generate-missing")
        self.assertEqual(code, 0, error)
        self.assertTrue(all(row["selected"] is not None for row in session.data))
        self.assertEqual([event for event in session.events if event[0] == "generate"],
                         [("generate", "vn")])
        self.assertNotIn("awaiting acceptance", output)
        session.close.assert_called_once_with()

    def test_regenerate_all_replaces_each_canonical_voice_once_and_selects_it(self):
        session = SharedSessionDouble()
        originals = {voice_id: deepcopy(session.row(voice_id)["selected"])
                     for voice_id in ("v1", "v2")}
        with patch("audiobook_renderer.casting.CastingSession", return_value=session), \
                patch("audiobook_renderer.audition.AudioPlayer") as player:
            code, output, error = self.run_cli("--action", "regenerate-all")
        self.assertEqual(code, 0, error)
        self.assertEqual([event for event in session.events if event[0] == "regenerate"],
                         [("regenerate", "vn"), ("regenerate", "v2")])
        # The alias appears before its canonical row. The separate character
        # has the same display name, but must still get its own new reference.
        rows = {row["voice_id"]: row for row in session.rows()}
        self.assertEqual(rows["vn"]["selected"], rows["v1"]["selected"])
        self.assertEqual(rows["v1"]["selected"]["candidate_id"], "v1-take-2")
        self.assertEqual(rows["v2"]["selected"]["candidate_id"], "v2-take-2")
        for voice_id in ("v1", "v2"):
            self.assertEqual(rows[voice_id]["candidates"][0]["reference_audio"],
                             originals[voice_id]["reference_audio"])
        self.assertIn("Regenerated 2 shared character voices.", output)
        self.assertIn("WARNING: regenerate-all", error)
        self.assertIn("This run targets 2 distinct voices.", error)
        # Group only canonical IDs: the different Mara must never be presented
        # as a linked role, even though its display name matches exactly.
        self.assertEqual([line for line in error.splitlines() if line.startswith("  ")],
                         ["  Mara (narration); Mara"])
        self.assertNotIn("WARNING:", output)
        self.assertNotIn("accept", output.lower())
        player.return_value.start.assert_not_called()
        player.return_value.stop.assert_called_once_with()
        session.close.assert_called_once_with()

    def test_regenerate_all_flushes_scope_warning_before_first_generation_without_prompt(self):
        class FlushTrackingStream(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flushed = []

            def flush(self):
                self.flushed.append(self.getvalue())
                super().flush()

        session = SharedSessionDouble()
        regenerate = session.regenerate
        stdout, stderr = io.StringIO(), FlushTrackingStream()

        def check_warning_then_regenerate(voice_id):
            self.assertEqual(len(stderr.flushed), 1)
            warning = stderr.flushed[0]
            self.assertIn("shared series voice library", warning)
            self.assertIn("Future renders of every book or role sharing these voice IDs", warning)
            self.assertIn("Previous samples are preserved", warning)
            self.assertIn("existing audiobook audio is not rewritten", warning)
            self.assertIn("  Mara (narration); Mara", warning)
            return regenerate(voice_id)

        session.regenerate = check_warning_then_regenerate
        with patch("audiobook_renderer.casting.CastingSession", return_value=session), \
                patch("builtins.input") as user_input, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["cast", str(self.bundle), "--config", str(self.config_path),
                         "--action", "regenerate-all"])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(len([event for event in session.events if event[0] == "regenerate"]), 2)
        user_input.assert_not_called()

    def test_regenerate_all_warns_about_series_scope_without_local_aliases(self):
        session = SessionDouble()
        session.close = Mock()
        with patch("audiobook_renderer.casting.CastingSession", return_value=session):
            code, output, error = self.run_cli("--action", "regenerate-all")
        self.assertEqual(code, 0, error)
        self.assertIn("WARNING: regenerate-all", error)
        self.assertIn("shared series voice library", error)
        self.assertIn("Sharing may extend beyond this bundle.", error)
        self.assertNotIn("Linked roles in this book", error)
        self.assertNotIn("WARNING:", output)
        self.assertEqual([event for event in session.events if event[0] == "regenerate"],
                         [("regenerate", "v1"), ("regenerate", "vn")])

    def test_regenerate_all_continues_after_failure_without_replacing_failed_voice(self):
        session = SharedSessionDouble()
        original = deepcopy(session.row("v1"))
        session.generation_error = RuntimeError("Transcript mismatch.\x1b[2J")
        with patch("audiobook_renderer.casting.CastingSession", return_value=session), \
                patch("audiobook_renderer.audition.AudioPlayer") as player:
            code, output, error = self.run_cli("--action", "regenerate-all")
        self.assertEqual(code, 2)
        self.assertIn("1 voices could not be regenerated", error)
        self.assertIn("prior references are retained", error)
        self.assertEqual(session.row("v1"), original)
        self.assertEqual(session.row("v2")["selected"]["candidate_id"], "v2-take-2")
        self.assertEqual([event for event in session.events if event[0] == "regenerate"],
                         [("regenerate", "vn"), ("regenerate", "v2")])
        self.assertIn("Could not regenerate Mara (narration)", output)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("Regenerated 2", output)
        player.return_value.stop.assert_called_once_with()
        session.close.assert_called_once_with()

    def test_regenerate_all_interrupt_stops_batch_and_closes_resources(self):
        session = SharedSessionDouble()
        original = deepcopy(session.data)
        session.generation_error = KeyboardInterrupt()
        with patch("audiobook_renderer.casting.CastingSession", return_value=session), \
                patch("audiobook_renderer.audition.AudioPlayer") as player:
            code, _, _ = self.run_cli("--action", "regenerate-all")
        self.assertEqual(code, 130)
        self.assertEqual(session.data, original)
        self.assertEqual([event for event in session.events if event[0] == "regenerate"],
                         [("regenerate", "vn")])
        player.return_value.stop.assert_called_once_with()
        session.close.assert_called_once_with()

    def test_invalid_action_arguments_fail_before_session_or_model(self):
        cases = [
            (["--action", "generate"], "--voice-id"),
            (["--action", "play"], "--voice-id"),
            (["--action", "accept", "--voice-id", "v1"], "--candidate-id"),
            (["--action", "list", "--candidate-id", "abc"], "--candidate-id"),
            (["--action", "edit", "--voice-id", "v1"], "--prompt-file"),
            (["--action", "list", "--prompt-file", "missing.txt"], "--prompt-file"),
            (["--action", "interactive", "--voice-id", "v1"], "--voice-id"),
            (["--action", "generate-missing", "--voice-id", "v1"], "--voice-id"),
            (["--action", "regenerate-all", "--voice-id", "v1"], "--voice-id"),
            (["--action", "regenerate-all", "--candidate-id", "abc"], "--candidate-id"),
            (["--action", "regenerate-all", "--prompt-file", "missing.txt"], "--prompt-file"),
        ]
        for args, message in cases:
            with self.subTest(args=args), patch("audiobook_renderer.casting.CastingSession") as session, \
                    patch("epub_audiobook.audition_scripts.ensure_bundle_scripts",
                          side_effect=AssertionError("Invalid arguments must not prepare audition scripts")):
                code, _, error = self.run_cli(*args)
                self.assertEqual(code, 2)
                self.assertIn(message, error)
                session.assert_not_called()

    def test_unknown_voice_fails_without_loading_models(self):
        with patch("audiobook_renderer.engines.create_engine") as engine, \
                patch("audiobook_renderer.verification.create_verifier") as verifier:
            code, _, error = self.run_cli("--action", "play", "--voice-id", "unknown")
        self.assertEqual(code, 2)
        self.assertIn("Voice ID is not in the bundle cast", error)
        engine.assert_not_called()
        verifier.assert_not_called()

    def test_unknown_candidate_is_not_played(self):
        self.publish_fixture()
        with patch("audiobook_renderer.audition.AudioPlayer") as player:
            code, _, error = self.run_cli(
                "--action", "play", "--voice-id", "v1", "--candidate-id", "f" * 32,
            )
        self.assertEqual(code, 2)
        self.assertIn("No matching voice sample", error)
        player.return_value.start.assert_not_called()

    def test_player_config_rejects_shell_string_and_invalid_argv(self):
        for player in ('"afplay sample.wav"', "[]", '[""]', '["afplay", 5]',
                       '["afplay", ""]', '["afplay", "\\u0000"]'):
            with self.subTest(player=player):
                self.config_path.write_text(self.base_config + f"[casting]\nplayer={player}\n")
                with self.assertRaisesRegex(ValueError, "casting.player"):
                    load_config(self.config_path)

    def test_player_configuration_comes_only_from_file(self):
        self.config_path.write_text(
            self.base_config + '[casting]\nplayer=["custom-player", "--quiet"]\n')
        with patch.dict("os.environ", {"AUDIOBOOK_PLAYER": "wrong-player", "CASTING_PLAYER": "wrong-player"}):
            config = load_config(self.config_path)
        self.assertEqual(config["casting"]["player"], ["custom-player", "--quiet"])

    def test_cli_interrupt_closes_session_and_player(self):
        with patch("audiobook_renderer.casting.CastingSession") as session, \
                patch("audiobook_renderer.audition.AudioPlayer") as player, \
                patch("audiobook_renderer.audition.run_casting", side_effect=KeyboardInterrupt):
            session.return_value.rows.return_value = []
            code, _, _ = self.run_cli()
        self.assertEqual(code, 130)
        player.return_value.stop.assert_called_once_with()
        session.return_value.close.assert_called_once_with()


class AudioPlayerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sample = Path(self.temp.name) / "a $(touch unwanted); voice.wav"
        self.sample.write_bytes(b"fixture")

    def process(self, *, status=None):
        process = Mock()
        process.poll.return_value = status
        process.returncode = 0 if status is None else status
        process.wait.return_value = process.returncode
        return process

    def test_playback_passes_literal_argv_and_never_uses_a_shell(self):
        process = self.process()
        player = AudioPlayer(["afplay", "--rate", "1"])
        with patch("audiobook_renderer.audition.shutil.which", return_value="/usr/bin/afplay"), \
                patch("audiobook_renderer.audition.subprocess.Popen", return_value=process) as popen:
            player.start(self.sample)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["/usr/bin/afplay", "--rate", "1", str(self.sample.resolve())])
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        errors = kwargs["stderr"]
        player.stop()
        self.assertTrue(errors.closed)
        self.assertIsNone(player.process)

    def test_starting_new_playback_stops_old_process_first(self):
        first, second = self.process(), self.process()
        player = AudioPlayer(["afplay"])
        events = []
        first.terminate.side_effect = lambda: events.append("stop-first")

        def launch(*args, **kwargs):
            events.append("start")
            return first if len(events) == 1 else second

        with patch("audiobook_renderer.audition.shutil.which", return_value="/usr/bin/afplay"), \
                patch("audiobook_renderer.audition.subprocess.Popen", side_effect=launch):
            player.start(self.sample)
            player.start(self.sample)
        self.assertEqual(events, ["start", "stop-first", "start"])
        first.wait.assert_called_once_with(timeout=2)
        self.assertIs(player.process, second)
        player.stop()

    def test_unresponsive_process_is_killed_and_cleanup_is_idempotent(self):
        process = self.process()
        process.wait.side_effect = [subprocess.TimeoutExpired("afplay", 2), 0]
        player = AudioPlayer(["afplay"])
        with patch("audiobook_renderer.audition.shutil.which", return_value="/usr/bin/afplay"), \
                patch("audiobook_renderer.audition.subprocess.Popen", return_value=process):
            player.start(self.sample)
        errors = player.errors
        player.stop()
        player.stop()
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)
        self.assertTrue(errors.closed)
        self.assertIsNone(player.process)
        self.assertIsNone(player.errors)

    def test_missing_player_fails_with_configuration_help(self):
        player = AudioPlayer(["missing-player"])
        with patch("audiobook_renderer.audition.shutil.which", return_value=None), \
                patch("audiobook_renderer.audition.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "casting.player"):
                player.start(self.sample)
        popen.assert_not_called()
        self.assertIsNone(player.errors)

    def test_process_launch_error_closes_error_stream(self):
        errors = io.BytesIO()
        player = AudioPlayer(["afplay"])
        with patch("audiobook_renderer.audition.shutil.which", return_value="/usr/bin/afplay"), \
                patch("audiobook_renderer.audition.tempfile.TemporaryFile", return_value=errors), \
                patch("audiobook_renderer.audition.subprocess.Popen", side_effect=OSError("launch failed")):
            with self.assertRaisesRegex(OSError, "launch failed"):
                player.start(self.sample)
        self.assertTrue(errors.closed)
        self.assertIsNone(player.process)
        self.assertIsNone(player.errors)

    def test_playback_failure_reports_sanitized_error_and_releases_process(self):
        process = self.process(status=3)
        player = AudioPlayer(["afplay"])
        with patch("audiobook_renderer.audition.shutil.which", return_value="/usr/bin/afplay"), \
                patch("audiobook_renderer.audition.subprocess.Popen", return_value=process):
            player.start(self.sample)
        errors = player.errors
        errors.write(b"bad audio \x1b[2J")
        with self.assertRaisesRegex(ValueError, "Audio playback failed") as raised:
            player.poll()
        self.assertNotIn("\x1b", str(raised.exception))
        self.assertTrue(errors.closed)
        self.assertIsNone(player.process)
        self.assertFalse(player.poll())
