"""Real pronunciation decisions, saved voices, audio caches, and terminal flow."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from epub_tts_narrator.pronunciations import load_pronunciations, save_candidates, set_pronunciation
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.engines import prepare_synthesis
from epub_tts_narrator.renderer.pronunciation_review import (
    PronunciationReviewSession, _sample_bounds, run_pronunciation_review,
)
from shared_fixtures import renderer_config
from test_renderer_workflow import FixtureEngine, write_tone


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"
HAS_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


class PreviewEngine(FixtureEngine):
    capabilities = dict(FixtureEngine.capabilities, structured_performance=True)

    def __init__(self):
        super().__init__({})
        self.calls = []
        self.closed = False

    def create_voice(self, *args, **kwargs):
        raise AssertionError("Pronunciation review must never create voices")

    def synthesize(self, text, voice, path, *, seed, direction="", performance=None, traits=""):
        prepared, compiled, metadata = prepare_synthesis(text, direction, performance=performance, traits=traits)
        self.calls.append({"text": text, "compiled": compiled, "voice": deepcopy(voice),
                           "seed": seed, "performance": deepcopy(performance), "direction": direction})
        self._write(compiled, path, seed)
        return metadata

    def close(self):
        self.closed = True


class PronunciationReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(EXAMPLE, self.root / "book")
        self.bundle = load_bundle(self.root / "book")
        self.config = renderer_config(self.root, series_id="pronunciation-review")
        self.config["engine"]["type"] = "fixture"
        self.config["engine"]["options"] = {"api_key": "DO_NOT_ECHO"}
        self.engine = PreviewEngine()
        self.factory = Mock(return_value=self.engine)
        self.resolved = resolve_book_config(self.bundle, self.config)
        self.store = voice_store(self.bundle, self.resolved)
        self.reference = self.root / "reference.wav"
        write_tone(self.reference)
        for profile in self.bundle["cast"]["voices"]:
            if self.store.load(profile["id"]) is None:
                self.store.publish(profile["id"], profile, self.reference, "An existing character voice.",
                                   self.engine.identity, {"passed": True}, {})
        set_pronunciation(self.bundle["directory"], self.bundle, "Mara", status="pending")
        self.output = []
        self.proposal = Mock(return_value={"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"})
        self.session = PronunciationReviewSession(self.bundle, self.config, engine_factory=self.factory,
                                                 store=self.store, notify=self.output.append, proposal_fn=self.proposal)
        self.addCleanup(self.session.close)

    def args(self, action="review", **kwargs):
        return SimpleNamespace(action=action, word=None, say_as=None, phonemes=None, **kwargs)

    def command(self, action="list", *, word=None, say_as=None, phonemes=None, **kwargs):
        args = SimpleNamespace(action=action, word=word, say_as=say_as, phonemes=phonemes)
        return run_pronunciation_review(self.bundle, self.config, args, engine_factory=self.factory,
                                        store=self.store, output=self.output.append,
                                        proposal_fn=kwargs.pop("proposal_fn", self.proposal), **kwargs)

    def candidate(self, *, proposal=None, kind="name"):
        entry = {key: value for key, value in load_pronunciations(self.bundle["directory"])[0].items()
                 if key not in {"origin_directory", "inherited", "proposal_origin_directory"}}
        entry.update(kind=kind, status="pending")
        entry.pop("pronunciation", None)
        if proposal is not None:
            entry["proposal"] = proposal
        save_candidates(self.bundle["directory"], self.bundle, [entry])

    def test_listing_and_authoritative_edits_never_initialize_an_engine(self):
        self.command()
        self.assertTrue(any("Mara" in line and "pending" in line for line in self.output))
        self.command("set", word="Mara", phonemes="M AA1 R AH0")
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["pronunciation"]["value"], "M AA1 R AH0")
        self.command("ignore", word="Mara")
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["status"], "ignored")
        self.command("reset", word="Mara")
        self.assertEqual(load_pronunciations(self.bundle["directory"]), [])
        self.factory.assert_not_called()
        self.proposal.assert_not_called()

    def test_saved_audition_draft_survives_restart_without_becoming_render_policy(self):
        correction = {"alphabet": "respelling", "value": "Mahrah"}
        self.session.edit("Mara", correction)
        restarted = PronunciationReviewSession(self.bundle, self.config)
        self.assertEqual(restarted.rows()[0]["draft"], correction)
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["status"], "pending")
        self.factory.assert_not_called()

    def test_cli_decision_clears_older_preview_draft(self):
        self.session.edit("Mara", {"alphabet": "respelling", "value": "Mahrah"})
        self.command("set", word="Mara", phonemes="M EH1 R AH0")
        self.assertNotIn("draft", self.session.rows()[0])
        self.assertEqual(self.session.rows()[0]["pronunciation"]["value"], "M EH1 R AH0")

    def test_default_is_not_confirmed_as_a_correction_without_explicit_ignore(self):
        with self.assertRaisesRegex(ValueError, "keep the default"):
            self.session.confirm(self.session.rows()[0])
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["status"], "pending")

    def test_invalid_action_arguments_fail_without_models_or_registry_edits(self):
        before = Path(self.bundle["directory"], "pronunciations.json").read_bytes()
        for args in (dict(action="set"), dict(action="set", word="Mara"),
                     dict(action="set", word="Mara", say_as="Mahrah", phonemes="M AA1 R AH0"),
                     dict(action="ignore", word="Mara", say_as="Mahrah"),
                     dict(action="set", word="Mara", say_as="<whisper>bad</whisper>")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.command(**args)
        self.assertEqual(Path(self.bundle["directory"], "pronunciations.json").read_bytes(), before)
        self.factory.assert_not_called()

    def test_missing_saved_voice_never_loads_model_or_creates_one(self):
        empty = voice_store(self.bundle, dict(self.resolved, voices=dict(self.resolved["voices"], directory=str(self.root / "empty-voices"))))
        session = PronunciationReviewSession(self.bundle, self.config, engine_factory=self.factory, store=empty)
        with self.assertRaisesRegex(ValueError, "existing saved voice"):
            session.audition("Mara")
        self.factory.assert_not_called()

    def test_sample_context_is_exact_bounded_and_never_cuts_the_target_word(self):
        text = "Before " * 100 + "Qarm " + "after " * 100
        first = text.index("Qarm")
        start, end = _sample_bounds(text, first, first + 4)
        self.assertLessEqual(end - start, 360)
        self.assertLessEqual(start, first)
        self.assertGreaterEqual(end, first + 4)
        self.assertTrue(text[start:end].startswith("Before"))
        self.assertFalse(text[end - 1].isspace())

    def test_review_quits_without_loading_engine_and_stops_player(self):
        player = SimpleNamespace(stop=Mock())
        self.command("review", input_fn=lambda _: "q", player=player)
        self.factory.assert_not_called()
        player.stop.assert_called_once()

    def test_readable_conversion_failure_keeps_the_previous_choice_and_menu_open(self):
        original = {"alphabet": "respelling", "value": "Mahrah"}
        set_pronunciation(self.bundle["directory"], self.bundle, "Mara", original)
        before = Path(self.bundle["directory"], "pronunciations.json").read_bytes()
        commands = iter(["e 1 Mairah", "q"])
        self.proposal.side_effect = ProviderError("proposal service unavailable")
        self.command("review", input_fn=lambda _: next(commands), player=SimpleNamespace(stop=Mock()))
        self.assertEqual(Path(self.bundle["directory"], "pronunciations.json").read_bytes(), before)
        self.assertNotIn("draft", self.session.rows()[0])
        self.assertTrue(any("Action stopped:" in line for line in self.output))
        self.factory.assert_not_called()

    def test_confirmed_respelling_is_preserved_without_proactive_conversion(self):
        original = {"alphabet": "respelling", "value": "Mahrah"}
        set_pronunciation(self.bundle["directory"], self.bundle, "Mara", original)
        self.assertEqual(self.session.selection(self.session.rows()[0]), original)
        self.proposal.assert_not_called()

    def test_missing_proposal_provider_never_falls_back_to_confirming_readable_guess(self):
        before = Path(self.bundle["directory"], "pronunciations.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "configured analysis model"):
            self.command("set", word="Mara", say_as="Mahrah", proposal_fn=None,
                         player=SimpleNamespace(stop=Mock()), playback_fn=Mock())
        self.assertEqual(Path(self.bundle["directory"], "pronunciations.json").read_bytes(), before)
        self.factory.assert_not_called()

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_pending_name_proposal_plays_exact_phones_and_only_then_confirms(self):
        proposed = self.proposal.return_value
        self.candidate(proposal=proposed)
        commands = iter(["l", "c 1", "1", "c 1", "q"])
        self.command("review", input_fn=lambda _: next(commands), player=SimpleNamespace(stop=Mock()), playback_fn=Mock())
        self.assertTrue(any("guessed; not verified" in line for line in self.output))
        self.assertTrue(any("Play this exact pronunciation" in line for line in self.output))
        self.assertEqual(len(self.engine.calls), 1)
        self.assertIn("<|phoneme_start|>M AA1 R AH0<|phoneme_end|>", self.engine.calls[0]["compiled"])
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["pronunciation"], proposed)
        self.proposal.assert_not_called()

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_pending_name_without_proposal_is_guessed_once_on_number_play(self):
        self.candidate()
        commands = iter(["1", "1", "q"])
        self.command("review", input_fn=lambda _: next(commands), player=SimpleNamespace(stop=Mock()), playback_fn=Mock())
        self.proposal.assert_called_once_with("Mara", None)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.session.rows()[0]["draft"], self.proposal.return_value)
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["status"], "pending")

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_set_readable_correction_stages_and_plays_cmu_then_confirmation_survives_restart(self):
        self.command("set", word="Mara", say_as="Mahrah", player=SimpleNamespace(stop=Mock()), playback_fn=Mock())
        self.proposal.assert_called_once_with("Mara", "Mahrah")
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["status"], "pending")
        self.assertIn("<|phoneme_start|>", self.engine.calls[0]["compiled"])
        restarted = PronunciationReviewSession(self.bundle, self.config, engine_factory=self.factory, store=self.store)
        self.addCleanup(restarted.close)
        restarted.confirm(restarted.rows()[0])
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["pronunciation"], self.proposal.return_value)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertNotIn("draft", restarted.rows()[0])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_play_readable_correction_keeps_cmu_draft_even_if_player_fails(self):
        with self.assertRaisesRegex(ValueError, "playback failed"):
            self.command("play", word="Mara", say_as="Mahrah", player=SimpleNamespace(stop=Mock()),
                         playback_fn=Mock(side_effect=ValueError("playback failed")))
        self.assertEqual(self.session.rows()[0]["draft"], self.proposal.return_value)
        with self.assertRaisesRegex(ValueError, "Play this exact pronunciation"):
            self.session.confirm(self.session.rows()[0])
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["status"], "pending")
        self.assertEqual(len(self.engine.calls), 1)

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_confirmation_rejects_unplayed_new_draft_and_new_voice_without_generating(self):
        proposed = self.proposal.return_value
        self.session.edit("Mara", proposed)
        preview = self.session.audition("Mara", proposed)
        with self.assertRaisesRegex(ValueError, "Play this exact pronunciation"):
            self.session.confirm(self.session.rows()[0])
        self.session.mark_played(preview)
        changed = {"alphabet": "cmu_arpabet", "value": "M EH1 R AH0"}
        self.session.edit("Mara", changed)
        with self.assertRaisesRegex(ValueError, "Play this exact pronunciation"):
            self.session.confirm(self.session.rows()[0])
        self.session.edit("Mara", proposed)
        selected = self.session._sample("Mara")
        profile = next(item for item in self.bundle["cast"]["voices"] if item["id"] == selected["voice_id"])
        candidate = self.store.publish_candidate(profile["id"], profile, self.reference, "A different reference transcript.",
                                                 self.engine.identity, {"passed": True}, {}, candidate_id="a" * 32)
        self.store.select_candidate(profile["id"], candidate["candidate_id"], selected["voice"]["asset_hash"])
        with self.assertRaisesRegex(ValueError, "Play this exact pronunciation"):
            self.session.confirm(self.session.rows()[0])
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["status"], "pending")

    def test_sample_checks_each_selected_voice_once_per_request_not_once_per_occurrence(self):
        row = self.bundle["chapters"][0]["rows"][1]
        self.bundle["chapters"][0]["rows"].extend([deepcopy(row) for _ in range(10)])
        with patch.object(self.store, "load", wraps=self.store.load) as load:
            self.session._sample("Mara")
            self.assertEqual(load.call_count, 1)
            self.session._sample("Mara")
            self.assertEqual(load.call_count, 2)

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_multiword_overlap_in_one_occurrence_does_not_poison_other_choices(self):
        row = self.bundle["chapters"][0]["rows"][1]
        text = "Mara Vale stood. Mara waited."
        row["text"] = text
        row["end"] = row["start"] + len(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": 0, "end": len("Mara Vale"), "original": "Mara Vale",
             "replacement": "Mahra Vale", "direction": "Read the name smoothly.", "reason": "Blend the name."}]}
        preview = self.session.audition("Mara", {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"})
        self.assertEqual(preview["purpose"], "unverified_pronunciation_listening_preview")
        self.assertIn("M AA1 R AH0", self.engine.calls[-1]["compiled"])
        self.assertEqual(len(row["speech_normalization"]["edits"]), 1)  # canonical row untouched
        from epub_tts_narrator.renderer.pronunciation_runtime import merge_row_pronunciations
        entries = {"Mara": {"word": "Mara", "status": "confirmed",
                            "pronunciation": {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"}}}
        with self.assertRaisesRegex(ValueError, "multiword"):
            merge_row_pronunciations(row, None, entries)  # production mapping guard stays strict

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_all_overlap_word_previews_exact_word_with_the_saved_voice(self):
        row = self.bundle["chapters"][0]["rows"][1]
        text = "My name is Mara Vale here."
        row["text"] = text
        row["end"] = row["start"] + len(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": len("My name is "), "end": len("My name is Mara Vale"),
             "original": "Mara Vale", "replacement": "Mahra Vale",
             "direction": "Read the name smoothly.", "reason": "Blend the name."}]}
        before = deepcopy(row["speech_normalization"])
        preview = self.session.audition("Mara", {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"})
        self.assertEqual(preview["request"]["word"], "Mara")
        self.assertEqual(self.engine.calls[-1]["voice"]["asset_hash"],
                         self.store.load(row["voice_id"])["asset_hash"])
        self.assertIn("Mara", self.engine.calls[-1]["text"])
        self.assertIn("M AA1 R AH0", self.engine.calls[-1]["compiled"])
        self.assertEqual(row["speech_normalization"], before)  # canonical row untouched

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_excerpt_preview_omits_only_target_overlapping_analyzer_edits(self):
        row = self.bundle["chapters"][0]["rows"][1]
        text = "My name is Mara Vale here. Zorble came."
        row["text"] = text
        row["end"] = row["start"] + len(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": len("My name is "), "end": len("My name is Mara Vale"),
             "original": "Mara Vale", "replacement": "Mahra Vale",
             "direction": "Read the name smoothly.", "reason": "Blend the name."},
            {"start": text.index("Zorble"), "end": text.index("Zorble") + len("Zorble"),
             "original": "Zorble", "replacement": "Zorbul",
             "direction": "Read it plainly.", "reason": "Approved spelling."}]}
        from epub_tts_narrator.spoken_text import build_spoken
        row["spoken_text"] = build_spoken(row)[0]  # analyzer-saved derived form must not stale the copy
        self.session.audition("Mara", {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"})
        synthesized = self.engine.calls[-1]["text"]
        self.assertIn("Zorbul", synthesized)      # unrelated approved edit preserved
        self.assertIn("Mara Vale", synthesized)   # target-overlapping edit omitted on the copy
        self.assertNotIn("Mahra", synthesized)

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_isolated_preview_constructs_word_control_without_source_mapping(self):
        row = self.bundle["chapters"][0]["rows"][1]
        text = "My name is Mara Vale here."
        row["text"] = text
        row["end"] = row["start"] + len(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": len("My name is "), "end": len("My name is Mara Vale"),
             "original": "Mara Vale", "replacement": "Mahra Vale",
             "direction": "Read the name smoothly.", "reason": "Blend the name."}]}
        preview = self.session.audition_isolated("Mara", {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"})
        self.assertEqual(preview["purpose"], "unverified_pronunciation_isolated_preview")
        self.assertEqual(preview["request"]["text"], "Mara")
        self.assertEqual(self.engine.calls[-1]["text"], "Mara")
        self.assertIn("M AA1 R AH0", self.engine.calls[-1]["compiled"])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_preview_uses_original_voice_exact_context_and_real_compiler_then_caches_only_flac(self):
        reference_hash = file_hash(self.reference)
        baseline = self.session.audition("Mara")
        corrected = self.session.audition("Mara", {"alphabet": "respelling", "value": "Mahrah"})
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(self.engine.calls[0]["text"], self.engine.calls[1]["text"])
        self.assertIn("Mara", self.engine.calls[0]["compiled"])
        self.assertIn("Mahrah", self.engine.calls[1]["compiled"])
        self.assertEqual(self.engine.calls[0]["voice"]["asset_hash"], self.engine.calls[1]["voice"]["asset_hash"])
        self.assertEqual(file_hash(self.reference), reference_hash)
        self.assertNotEqual(baseline["path"], corrected["path"])
        self.assertTrue(Path(corrected["path"]).is_file())
        self.assertEqual(list(self.session.directory.rglob("*.wav")), [])
        self.assertEqual(self.session.audition("Mara", {"alphabet": "respelling", "value": "Mahrah"}), corrected)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertFalse(self.engine.voice_calls)
        self.assertEqual(corrected["purpose"], "unverified_pronunciation_listening_preview")
        self.assertNotIn("DO_NOT_ECHO", Path(corrected["path"]).with_name("preview.json").read_text())

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_interrupted_compression_reuses_completed_speech_on_resume(self):
        with patch("epub_tts_narrator.renderer.chapter_executor._flac_master", side_effect=RuntimeError("encoder interrupted")):
            with self.assertRaisesRegex(RuntimeError, "encoder interrupted"):
                self.session.audition("Mara")
        self.assertEqual(len(self.engine.calls), 1)
        state = read_json(next(self.session.directory.rglob("preview.json")))
        self.assertEqual(state["status"], "generated")
        resumed = self.session.audition("Mara")
        self.assertEqual(resumed["status"], "ready")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(list(self.session.directory.rglob("*.wav")), [])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_corrected_preview_sanitizes_bracketed_delivery_like_normal_rendering(self):
        for chapter in self.bundle["chapters"]:
            for row in chapter["rows"]:
                row["delivery"] = {"direction": "Emphasize [Mara] with awe.", "evidence": []}
        self.session.audition("Mara", {"alphabet": "respelling", "value": "Mahrah"})
        self.assertEqual(self.engine.calls[0]["direction"], "Emphasize (Mara) with awe.")
        self.assertIn("Mahrah", self.engine.calls[0]["compiled"])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_interrupted_synthesis_discards_only_its_uncheckpointed_partial_file(self):
        def interrupted(text, voice, path, **kwargs):
            Path(path).write_bytes(b"partial encoder output")
            raise RuntimeError("generation interrupted")
        with patch.object(self.engine, "synthesize", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "generation interrupted"):
                self.session.audition("Mara")
        pending = read_json(next(self.session.directory.rglob("preview.json")))
        self.assertEqual(pending["status"], "pending")
        self.assertNotIn("raw_sha256", pending)
        self.assertEqual(self.session.audition("Mara")["status"], "ready")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.engine.calls[0]["seed"], pending["take_seed"])
        self.assertFalse(list(self.session.directory.rglob("*.wav")))

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_changed_preview_does_not_silently_regenerate_or_change_voice(self):
        preview = self.session.audition("Mara")
        Path(preview["path"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "preview changed"):
            self.session.audition("Mara")
        self.assertEqual(len(self.engine.calls), 1)

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_number_immediately_plays_edit_auditions_and_only_confirm_changes_registry(self):
        commands = iter(["1", "e 1 Mahrah", "c 1", "b 1", "q"])
        player = SimpleNamespace(stop=Mock())
        played = []
        self.command("review", input_fn=lambda _: next(commands), player=player,
                     playback_fn=lambda _, path: played.append(path))
        self.assertEqual(len(played), 3)
        self.assertEqual(played[0], played[2])
        self.assertNotEqual(played[0], played[1])
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["pronunciation"], self.proposal.return_value)
        self.proposal.assert_called_once_with("Mara", "Mahrah")
        self.assertIn("<|phoneme_start|>M AA1 R AH0<|phoneme_end|>", self.engine.calls[1]["compiled"])
        self.assertTrue(self.engine.closed)
        player.stop.assert_called_once()

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_play_action_previews_unsaved_override_without_changing_registry(self):
        player = SimpleNamespace(stop=Mock())
        playback = Mock()
        before = Path(self.bundle["directory"], "pronunciations.json").read_bytes()
        self.command("play", word="Mara", phonemes="M AA1 R AH0", player=player, playback_fn=playback)
        playback.assert_called_once()
        self.assertIn("<|phoneme_start|>M AA1 R AH0<|phoneme_end|>", self.engine.calls[0]["compiled"])
        self.assertEqual(Path(self.bundle["directory"], "pronunciations.json").read_bytes(), before)

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_optional_phoneme_menu_edits_audition_before_explicit_confirmation(self):
        commands = iter(["p 1 M AA1 R AH0", "c 1", "q"])
        played = []
        self.command("review", input_fn=lambda _: next(commands), player=SimpleNamespace(stop=Mock()),
                     playback_fn=lambda _, path: played.append(path))
        self.assertEqual(len(played), 1)
        self.assertEqual(load_pronunciations(self.bundle["directory"])[0]["pronunciation"],
                         {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"})
        self.assertIn("<|phoneme_start|>", self.engine.calls[0]["compiled"])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_silent_preview_is_audited_and_next_explicit_play_uses_a_new_take(self):
        seeds = []
        def silence(text, voice, path, **kwargs):
            seeds.append(kwargs["seed"])
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 44100)
        with patch.object(self.engine, "synthesize", side_effect=silence) as synthesize:
            with self.assertRaisesRegex(ValueError, "silent.*Play again"):
                self.session.audition("Mara")
            synthesize.assert_called_once()
        state_path = next(self.session.directory.rglob("preview.json"))
        state = read_json(state_path)
        self.assertEqual(state["status"], "rejected")
        self.assertEqual(state["take"], 1)
        self.assertEqual(state["rejections"][0]["raw_sha256"], file_hash(state_path.with_name("raw.wav")))
        self.assertEqual(state["rejections"][0]["normalized_sha256"], file_hash(state_path.with_name("normalized.wav")))
        self.assertIn("silent", state["rejections"][0]["reason"])
        self.assertFalse(list(self.session.directory.rglob("*.flac")))
        resumed = self.session.audition("Mara")
        self.assertEqual(resumed["status"], "ready")
        self.assertEqual(resumed["take"], 2)
        self.assertEqual(self.engine.calls[0]["seed"], seeds[0] + 1)
        self.assertEqual(resumed["rejections"], state["rejections"])
        self.assertEqual(self.session.audition("Mara"), resumed)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(list(self.session.directory.rglob("*.wav")), [])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_modified_rejected_audio_is_retained_instead_of_overwritten_on_retry(self):
        with patch("epub_tts_narrator.renderer.pronunciation_review._preview_audio", side_effect=ValueError("invalid preview")):
            with self.assertRaisesRegex(ValueError, "Play again"):
                self.session.audition("Mara")
        state_path = next(self.session.directory.rglob("preview.json"))
        raw = state_path.with_name("raw.wav")
        original_raw = raw.read_bytes()
        normalized = state_path.with_name("normalized.wav")
        normalized.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed.*retained"):
            self.session.audition("Mara")
        self.assertEqual(raw.read_bytes(), original_raw)
        self.assertEqual(normalized.read_bytes(), b"changed")
        self.assertEqual(len(self.engine.calls), 1)


if __name__ == "__main__":
    unittest.main()
