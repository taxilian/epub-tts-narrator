"""Listening queue: real registry, review-state sidecar, preview cache, and FFmpeg.

Only playback completion, gap keys, and menu input are faked; synthesis runs
through the real preview compiler and cache, and acceptance writes the real
book-local pronunciation registry.
"""

from copy import deepcopy
from pathlib import Path
import json
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.pronunciations import load_pronunciations, set_pronunciation
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.pronunciation_review import PronunciationReviewSession
from epub_tts_narrator.renderer.pronunciation_review_queue import run_pronunciation_queue
from epub_tts_narrator.renderer.pronunciation_review_state import ReviewState
from epub_tts_narrator.series import link_analysis
from shared_fixtures import renderer_config
from test_renderer_pronunciation_review import EXAMPLE, PreviewEngine
from test_renderer_workflow import write_tone
from test_series import fixture, propose


HAS_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
EXTRA_WORDS = ["Qarm", "Xylo", "Vandire", "Kethra", "Ollo", "Brant", "Sella", "Durn", "Pell", "Wick"]
CMU_QARM = {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"}
CMU_XYLO = {"alphabet": "cmu_arpabet", "value": "Z AY1 L OW0"}


def scripted(commands):
    queue = iter(commands)

    def input_fn(prompt=""):
        try:
            return next(queue)
        except StopIteration:
            raise EOFError from None

    return input_fn


class PronunciationReviewQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(EXAMPLE, self.root / "book")
        directory = self.root / "book"
        source_path = directory / "source.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        addition = " " + " ".join(EXTRA_WORDS) + "."
        source["blocks"][1]["text"] += addition
        source_path.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        script_path = directory / "scripts/chapter-0001.jsonl"
        script_rows = [json.loads(line) for line in
                       script_path.read_text(encoding="utf-8").splitlines()]
        script_rows[1]["text"] += addition
        script_rows[1]["end"] = len(source["blocks"][1]["text"])
        script_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in script_rows),
                               encoding="utf-8")
        from epub_tts_narrator.renderer.common import file_hash
        manifest = read_json(directory / "manifest.json")
        manifest["source_fingerprint"] = fingerprint(source)
        manifest["sha256"]["source.json"] = file_hash(source_path)
        manifest["sha256"]["scripts/chapter-0001.jsonl"] = file_hash(script_path)
        write_json(directory / "manifest.json", manifest)
        self.bundle = load_bundle(directory)
        self.config = renderer_config(self.root, series_id="pronunciation-queue")
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
        for word in ("Mara", "Lena", *EXTRA_WORDS):
            set_pronunciation(self.bundle["directory"], self.bundle, word, status="pending")
        self.words = [row["word"] for row in load_pronunciations(self.bundle["directory"])]
        self.output = []
        self.session = PronunciationReviewSession(self.bundle, self.config, engine_factory=self.factory,
                                                  store=self.store, notify=self.output.append)
        self.addCleanup(self.session.close)

    def queue(self, commands, *, playback=None, wait=None, interactive=True, session=None):
        inputs = scripted(commands)
        redrawn, played = [], []

        def playback_fn(player, path):
            played.append(path)
            if playback is not None:
                return playback(len(played), path)
            return True

        waiter = Mock(return_value=wait)
        with patch("epub_tts_narrator.renderer.terminal.wait_key", waiter):
            result = run_pronunciation_queue(
                session or self.session, SimpleNamespace(stop=Mock()), playback_fn,
                input_fn=inputs, output=self.output.append, interactive=interactive,
                redraw=redrawn.append, gap_seconds=0)
        return result, self.output, redrawn, played, waiter

    def rows(self):
        return load_pronunciations(self.bundle["directory"])

    def row(self, word):
        return next(row for row in self.rows() if row["word"] == word)

    def state_path(self):
        return self.session.directory / "review-state.json"

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_uninterrupted_pairs_autoaccept_every_pending_word_and_persist_bound_state(self):
        self.session.edit("Qarm", CMU_QARM)
        self.session.edit("Xylo", CMU_XYLO)
        result, lines, redrawn, played, waiter = self.queue(["s"])
        self.assertEqual(result, 0)
        self.assertEqual(self.row("Qarm")["status"], "confirmed")
        self.assertEqual(self.row("Qarm")["pronunciation"], CMU_QARM)
        self.assertEqual(self.row("Xylo")["pronunciation"], CMU_XYLO)
        for word in self.words:
            if word not in {"Qarm", "Xylo"}:
                self.assertEqual(self.row(word)["status"], "ignored", word)
        self.assertEqual(len(played), 2 * len(self.words))
        self.assertEqual(waiter.call_count, 2 * len(self.words))
        self.assertTrue(any("Listening queue complete" in line for line in lines))
        state = read_json(self.state_path())
        self.assertEqual(state["format"], "epub-tts-pronunciation-review")
        self.assertIsNone(state["resume_word"])
        for word in self.words:
            entry = state["entries"][word]
            self.assertTrue(entry["accepted"])
            self.assertEqual(set(entry["receipt"]), {"request_sha256", "audio_sha256"})
        self.assertEqual(state["entries"]["Qarm"]["choice"], CMU_QARM)
        drawn = "".join(redrawn)
        for word in self.words:
            self.assertIn(word, drawn)
        purposes = {read_json(path)["purpose"] for path in self.session.directory.glob("*/preview.json")}
        self.assertEqual(purposes, {"unverified_pronunciation_isolated_preview"})

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_interrupted_play_never_autoaccepts_and_returns_to_last_five_words(self):
        def playback(count, path):
            return count != 11  # first play of the sixth word stops early

        result, lines, redrawn, played, waiter = self.queue(["s"], playback=playback)
        self.assertEqual(result, 0)
        accepted = self.words[:5]
        for word in accepted:
            self.assertEqual(self.row(word)["status"], "ignored", word)
        for word in self.words[5:]:
            self.assertEqual(self.row(word)["status"], "pending", word)
        self.assertEqual(len(played), 11)
        state = read_json(self.state_path())
        self.assertEqual(state["resume_word"], self.words[1])  # five words: indexes 1..5
        self.assertTrue(any("Interrupted at word 6/12" in line for line in lines))
        window = [line for line in lines if re.match(r"^  \d\. \[\d+/12\] ", line)]
        self.assertEqual(len(window), 5)
        self.assertTrue(any(f"[2/12] {self.words[1]}" in line for line in window))
        self.assertTrue(any("Enter resumes the queue at [2/12]" in line for line in lines))
        self.assertNotIn(self.words[5], state["entries"])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_key_during_gap_interrupts_before_any_acceptance(self):
        result, lines, redrawn, played, waiter = self.queue(["s"], wait="key")
        self.assertEqual(result, 0)
        for word in self.words:
            self.assertEqual(self.row(word)["status"], "pending", word)
        self.assertEqual(len(played), 1)
        self.assertEqual(waiter.call_count, 1)
        state = read_json(self.state_path())
        self.assertEqual(state["resume_word"], self.words[0])
        self.assertEqual(state["entries"], {})

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_failed_playback_stops_queue_and_keeps_unheard_words_pending(self):
        def playback(count, path):
            if count == 3:
                raise ValueError("playback failed")
            return True

        result, lines, redrawn, played, waiter = self.queue(["s"], playback=playback)
        self.assertEqual(result, 0)
        self.assertEqual(self.row(self.words[0])["status"], "ignored")  # its pair completed first
        for word in self.words[1:]:
            self.assertEqual(self.row(word)["status"], "pending", word)
        self.assertEqual(len(played), 3)
        self.assertTrue(any("Action stopped: playback failed" in line for line in lines))
        self.assertNotIn(self.words[1], read_json(self.state_path())["entries"])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_resume_reopens_at_saved_position_and_finishes_remaining_words(self):
        def interrupt_once(count, path):
            return count != 11

        self.queue(["s"], playback=interrupt_once)
        result, lines, redrawn, played, waiter = self.queue([""])  # blank Enter resumes
        self.assertEqual(result, 0)
        for word in self.words:
            self.assertEqual(self.row(word)["status"], "ignored", word)
        accepted_before_interrupt = 5
        self.assertEqual(len(played), 2 * (len(self.words) - accepted_before_interrupt))
        self.assertTrue(any("Listening queue complete" in line for line in lines))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_explicit_start_number_skips_earlier_pending_words(self):
        result, lines, redrawn, played, waiter = self.queue(["s 6"])
        self.assertEqual(result, 0)
        for word in self.words[:5]:
            self.assertEqual(self.row(word)["status"], "pending", word)
        for word in self.words[5:]:
            self.assertEqual(self.row(word)["status"], "ignored", word)
        self.assertEqual(len(played), 2 * (len(self.words) - 5))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_reopen_marks_sidecar_only_and_registry_stays_byte_identical(self):
        self.session.edit("Qarm", CMU_QARM)
        self.queue(["s"])
        self.assertEqual(self.row("Qarm")["status"], "confirmed")
        registry_bytes = Path(self.bundle["directory"], "pronunciations.json").read_bytes()
        first, last = self.words.index("Mara"), self.words.index("Qarm")
        result, lines, redrawn, played, waiter = self.queue([f"o {first + 1}-{last + 1}", "q"])
        self.assertEqual(result, 0)
        self.assertEqual(Path(self.bundle["directory"], "pronunciations.json").read_bytes(), registry_bytes)
        self.assertEqual(read_json(self.session.directory / "drafts.json")["entries"], {})
        state = read_json(self.state_path())
        for word in self.words[first:last + 1]:
            self.assertEqual(state["entries"][word]["reopened"], True, word)
            self.assertFalse(state["entries"][word]["accepted"], word)
        for word in self.words[:first] + self.words[last + 1:]:
            self.assertTrue(state["entries"][word]["accepted"], word)
            self.assertFalse(state["entries"][word]["reopened"], word)
        self.assertTrue(any("registry decisions are unchanged" in line for line in lines))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_reopened_words_are_reheard_and_reaccepted_without_losing_choices(self):
        self.session.edit("Qarm", CMU_QARM)
        self.queue(["s"])
        self.queue(["o all", "s"])
        self.assertEqual(self.row("Qarm")["status"], "confirmed")
        self.assertEqual(self.row("Qarm")["pronunciation"], CMU_QARM)
        state = read_json(self.state_path())
        self.assertTrue(all(entry["accepted"] and not entry["reopened"] for entry in state["entries"].values()))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_confirmed_row_with_new_draft_is_queued_and_reaccepted(self):
        self.session.edit("Qarm", CMU_QARM)
        self.queue(["s"])
        self.assertEqual(self.row("Qarm")["pronunciation"], CMU_QARM)
        self.session.edit("Qarm", CMU_XYLO)
        result, lines, redrawn, played, waiter = self.queue(["s"])
        self.assertEqual(result, 0)
        self.assertEqual(self.row("Qarm")["pronunciation"], CMU_XYLO)
        self.assertEqual(self.row("Qarm")["status"], "confirmed")
        state = read_json(self.state_path())
        self.assertEqual(state["entries"]["Qarm"]["choice"], CMU_XYLO)
        self.assertTrue(state["entries"]["Qarm"]["accepted"])

    def test_invalid_start_argument_rejects_without_starting_queue(self):
        result, lines, redrawn, played, waiter = self.queue(["s abc", "q"])
        self.assertEqual(result, 0)
        self.assertEqual(played, [])
        self.assertTrue(any("Action stopped: Use S, or S N with an entry number" in line for line in lines))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_enter_resume_returns_to_menu_after_player_io_failure(self):
        state = ReviewState(self.session.directory, self.bundle["source"]["book"]["id"])
        state.resume_word = self.words[0]
        state.save()

        def failing_player(count, path):
            raise OSError("player unavailable")

        result, lines, redrawn, played, waiter = self.queue(["", "q"], playback=failing_player)

        self.assertEqual(result, 0)
        self.assertTrue(any("Action stopped: player unavailable" in line for line in lines))
        self.assertEqual(read_json(self.state_path())["resume_word"], self.words[0])
        self.assertTrue(all(row["status"] == "pending" for row in self.rows()))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_resume_position_saved_before_expensive_preview_work_fails(self):
        target = self.words[3]
        original = self.session.audition_isolated

        def failing(word, pronunciation=None, *, cached_only=False):
            if word == target:
                raise ValueError("preview generation failed")
            return original(word, pronunciation, cached_only=cached_only)

        with patch.object(self.session, "audition_isolated", side_effect=failing):
            result, lines, redrawn, played, waiter = self.queue(["s"])
        self.assertEqual(result, 0)
        self.assertTrue(any("Action stopped: preview generation failed" in line for line in lines))
        state = read_json(self.state_path())
        self.assertEqual(state["resume_word"], target)
        self.assertNotIn(target, state["entries"])
        for word in self.words[:3]:
            self.assertTrue(state["entries"][word]["accepted"], word)

    def test_reopen_marks_inherited_decisions_and_never_touches_ancestor_registries(self):
        first, _ = self.publish_book("book-one")
        set_pronunciation(first, load_bundle(first), "Mara", CMU_QARM)
        ancestor_bytes = (first / "pronunciations.json").read_bytes()
        second, _ = self.publish_book("book-two", previous=first)
        bundle = load_bundle(second)
        # Analysis directories own their locations; a global config must not set them.
        config = renderer_config(None, series_id="pronunciation-queue-series")
        engine = PreviewEngine()
        store = voice_store(bundle, resolve_book_config(bundle, config))
        reference = self.root / "series-reference.wav"
        write_tone(reference)
        for profile in bundle["cast"]["voices"]:
            if store.load(profile["id"]) is None:
                store.publish(profile["id"], profile, reference, "An existing character voice.",
                              engine.identity, {"passed": True}, {})
        session = PronunciationReviewSession(bundle, config, engine_factory=Mock(return_value=engine),
                                             store=store)
        self.addCleanup(session.close)
        set_pronunciation(second, bundle, "Vale", status="pending")
        second_bytes = (second / "pronunciations.json").read_bytes()
        output = []
        result = run_pronunciation_queue(session, SimpleNamespace(stop=Mock()), Mock(),
                                         input_fn=scripted(["o all", "q"]), output=output.append,
                                         interactive=False)
        self.assertEqual(result, 0)
        self.assertEqual((second / "pronunciations.json").read_bytes(), second_bytes)
        inherited = next(row for row in session.rows() if row["word"] == "Mara")
        self.assertTrue(inherited["inherited"])
        self.assertEqual(inherited["status"], "confirmed")
        state = ReviewState(session.directory, bundle["source"]["book"]["id"])
        self.assertTrue(state.queued(inherited))
        self.assertEqual((first / "pronunciations.json").read_bytes(), ancestor_bytes)
        result = run_pronunciation_queue(session, SimpleNamespace(stop=Mock()), Mock(return_value=True),
                                         input_fn=scripted(["s"]), output=output.append, interactive=True,
                                         redraw=lambda value: None, gap_seconds=0)
        self.assertEqual(result, 0)
        self.assertEqual((first / "pronunciations.json").read_bytes(), ancestor_bytes)
        local = load_pronunciations(second)
        mara = next(row for row in local if row["word"] == "Mara")
        self.assertEqual(mara["pronunciation"], CMU_QARM)
        self.assertFalse(mara["inherited"])

    def publish_book(self, name, previous=None):
        document = fixture(sequel=previous is not None)
        if previous is None:
            registry = fresh_registry(self.root / name)
        else:
            registry = continuation_registry(previous, document["source"])
            prior = json.loads((previous / "analysis.json").read_text(encoding="utf-8"))
            propose(document, prior["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, self.root / name, export_bundle,
                          continue_from=previous)
        return self.root / name, document

    @staticmethod
    def row_status(directory, word):
        return next(row for row in load_pronunciations(directory) if row["word"] == word)["status"]

    def test_non_interactive_session_refuses_walk_and_leaves_registry_unchanged(self):
        result, lines, redrawn, played, waiter = self.queue(["s", "q"], interactive=False)
        self.assertEqual(result, 0)
        self.assertEqual(played, [])
        self.assertTrue(any("interactive terminal" in line for line in lines))
        for word in self.words:
            self.assertEqual(self.row(word)["status"], "pending", word)

    def test_eof_closes_safely_and_keeps_queue_position(self):
        result, lines, redrawn, played, waiter = self.queue([])
        self.assertEqual(result, 0)
        self.assertTrue(any("closed" in line for line in lines))

    def test_compact_list_arranges_width_aware_columns_for_more_than_ten_entries(self):
        import shutil
        self.session.edit("Qarm", CMU_QARM)
        result, lines, redrawn, played, waiter = self.queue(["l", "q"])
        self.assertEqual(result, 0)
        columns = shutil.get_terminal_size((80, 24)).columns
        listed = [line for line in lines if re.match(r"^\s*\d+\*? \S+ — ", line)]
        joined = " ".join(listed)
        for word in self.words:
            self.assertIn(word, joined)
        for line in listed:
            self.assertLessEqual(len(line), columns)
        self.assertLess(len(listed), len(self.words))  # horizontal columns, not one row per entry
        self.assertRegex(joined, r"\b5\* Mara — pending")

    def test_acceptance_is_bound_to_choice_and_invalidated_by_draft_change(self):
        self.session.edit("Qarm", CMU_QARM)
        state = ReviewState(self.session.directory, self.bundle["source"]["book"]["id"])
        row = next(row for row in self.session.rows() if row["word"] == "Qarm")
        self.assertIsNone(state.accepted(row))
        state.record("Qarm", CMU_QARM, {"request_sha256": "a" * 64, "audio_sha256": "b" * 64},
                     accepted=True)
        reloaded = ReviewState(self.session.directory, self.bundle["source"]["book"]["id"])
        self.assertIsNotNone(reloaded.accepted(row))
        self.session.edit("Qarm", CMU_XYLO)
        changed = next(item for item in self.session.rows() if item["word"] == "Qarm")
        self.assertIsNone(reloaded.accepted(changed))
        reloaded.clear(["Qarm"])
        self.assertFalse(read_json(self.state_path())["entries"])

    def test_corrupted_or_foreign_review_state_fails_closed(self):
        state = ReviewState(self.session.directory, self.bundle["source"]["book"]["id"])
        state.record("Qarm", CMU_QARM, {"request_sha256": "a" * 64, "audio_sha256": "b" * 64},
                     accepted=True)
        value = read_json(self.state_path())
        value["entries"]["Qarm"]["choice"] = CMU_XYLO  # tamper without re-signing
        write_json(self.state_path(), value)
        with self.assertRaisesRegex(ValueError, "changed"):
            ReviewState(self.session.directory, self.bundle["source"]["book"]["id"])
        with self.assertRaisesRegex(ValueError, "different book"):
            ReviewState(self.session.directory, "some-other-book")

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
    def test_isolated_preview_has_distinct_cache_identity_and_hash_bound_receipt(self):
        from epub_tts_narrator.renderer.common import digest
        excerpt = self.session.audition("Qarm", CMU_QARM)
        isolated = self.session.audition_isolated("Qarm", CMU_QARM)
        self.assertNotEqual(Path(excerpt["path"]).parent, Path(isolated["path"]).parent)
        self.assertEqual(excerpt["purpose"], "unverified_pronunciation_listening_preview")
        self.assertEqual(isolated["purpose"], "unverified_pronunciation_isolated_preview")
        self.assertTrue(isolated["request"]["isolated"])
        self.assertEqual(isolated["request"]["text"], "Qarm")
        self.assertIn("Qarm", excerpt["request"]["text"])
        self.assertGreater(len(excerpt["request"]["text"]), len("Qarm"))
        self.assertIn("<|phoneme_start|>K W AO1 R M<|phoneme_end|>", self.engine.calls[-1]["compiled"])
        isolated_state = read_json(Path(isolated["path"]).with_name("preview.json"))
        self.assertNotIn("played", isolated_state)
        self.session.mark_played(isolated)
        isolated_state = read_json(Path(isolated["path"]).with_name("preview.json"))
        self.assertEqual(isolated_state["played"],
                         {"request_sha256": digest(isolated_state["request"]),
                          "audio_sha256": isolated_state["audio_sha256"]})
        excerpt_state = read_json(Path(excerpt["path"]).with_name("preview.json"))
        self.assertNotIn("played", excerpt_state)


if __name__ == "__main__":
    unittest.main()
