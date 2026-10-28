from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.name_script_workflow import prepare_name_script
from epub_tts_narrator.pronunciations import load_pronunciations
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.pronunciation_review import PronunciationReviewSession
from epub_tts_narrator.renderer.pronunciation_review_queue import run_pronunciation_queue
from epub_tts_narrator.renderer.pronunciation_review_state import ReviewState
from epub_tts_narrator.scoped_pronunciations import scope_for
from name_script_fixtures import NameClient, PHONES, book, requests
from shared_fixtures import renderer_config
from test_renderer_pronunciation_review import HAS_FFMPEG, PreviewEngine
from test_renderer_workflow import write_tone


class ScopedPronunciationReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = book(self.root / "book", two=True)
        prepare_name_script(self.bundle["directory"], {}, requests=requests(self.root, NameClient()))
        self.config = renderer_config(paragraph_synthesis=False)
        self.config["engine"]["type"] = "fixture"
        self.engine = PreviewEngine()
        self.store = voice_store(self.bundle, resolve_book_config(self.bundle, self.config))
        reference = self.root / "reference.wav"
        write_tone(reference)
        for profile in self.bundle["cast"]["voices"]:
            self.store.publish(profile["id"], profile, reference, "Existing reference.",
                               self.engine.identity, {"passed": True}, {})
        self.first = self.session("N1")
        self.second = self.session("N2")

    def session(self, key):
        session = PronunciationReviewSession(self.bundle, self.config, scope=scope_for(self.bundle, key),
            engine_factory=lambda config: self.engine, store=self.store, notify=lambda message: None,
            proposal_fn=lambda word, respelling=None: PHONES)
        self.addCleanup(session.close)
        return session

    def test_drafts_case_families_and_queue_resume_are_entity_isolated(self):
        self.assertNotEqual(self.first.rows("Ally")[0]["id"], self.second.rows("Ally")[0]["id"])
        self.first.edit("Ally", PHONES)
        self.assertEqual(self.first.rows("ALLY")[0]["draft"], PHONES)
        self.assertNotIn("draft", self.second.rows("Ally")[0])
        first = ReviewState(self.first.directory, self.bundle["source"]["book"]["id"])
        first.mark_reopened("Ally", PHONES)
        second = ReviewState(self.second.directory, self.bundle["source"]["book"]["id"])
        self.assertEqual(second.entries, {})
        self.second.ignore("Ally")
        self.assertEqual(self.first.rows("Ally")[0]["status"], "pending")
        self.assertEqual(self.second.rows("Ally")[0]["status"], "ignored")

    def test_explicit_special_word_stays_reviewable_even_when_dictionary_known(self):
        from epub_tts_narrator.scoped_pronunciations import set_choice
        set_choice(self.bundle["directory"], self.bundle, "softly", status="pending", scope={"class": "special"})
        session = PronunciationReviewSession(self.bundle, self.config, scope={"class": "special"})
        self.assertIn("softly", [row["word"] for row in session.rows()])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_special_preview_excludes_uncertainty_and_queue_cannot_confirm_it(self):
        from epub_tts_narrator.name_script_state import mention_id, read_state
        from epub_tts_narrator.name_script_workflow import resolve_mention
        from epub_tts_narrator.scoped_pronunciations import set_choice
        state = read_state(self.bundle["directory"])
        resolve_mention(self.bundle["directory"], mention_id(state, "b1", 0, 4), "unresolved")
        set_choice(self.bundle["directory"], self.bundle, "Ally", status="pending", scope={"class": "special"})
        session = PronunciationReviewSession(self.bundle, self.config, scope={"class": "special"},
            engine_factory=lambda config: self.engine, store=self.store, notify=lambda message: None)
        self.addCleanup(session.close)
        session.edit("Ally", PHONES)
        preview = session.audition("Ally", PHONES)
        controls = preview["request"]["options"]["performance"]["controls"]
        self.assertEqual([item["start"] for item in controls], [preview["request"]["text"].index("Ally with")])
        path = Path(self.bundle["directory"]) / "scoped-pronunciations.json"
        before = path.read_bytes()
        commands = iter(["s", "q"])
        with patch("epub_tts_narrator.renderer.terminal.wait_key", return_value=None):
            run_pronunciation_queue(session, object(), lambda player, path: True,
                input_fn=lambda prompt: next(commands), output=lambda message: None,
                interactive=True, redraw=lambda line: None, gap_seconds=0)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(session.rows("Ally")[0]["status"], "pending")
        self.assertIsNone(ReviewState(session.directory, self.bundle["source"]["book"]["id"]).accepted(session.rows("Ally")[0]))

    def test_special_preview_with_only_unresolved_or_named_occurrences_never_synthesizes(self):
        from epub_tts_narrator.name_script_state import mention_id, read_state
        from epub_tts_narrator.name_script_workflow import resolve_mention
        from name_script_fixtures import TEXT
        state = read_state(self.bundle["directory"])
        for start, word in ((TEXT.index("ally"), "ally"), (TEXT.index("Ally with"), "Ally")):
            resolve_mention(self.bundle["directory"], mention_id(state, "b1", start, start + len(word)), "unresolved")
        session = PronunciationReviewSession(self.bundle, self.config, scope={"class": "special"},
            engine_factory=lambda config: self.engine, store=self.store, notify=lambda message: None)
        self.addCleanup(session.close)
        for preview in (session.audition, session.audition_isolated):
            with self.assertRaises(ValueError):
                preview("Ally", PHONES)
        self.assertEqual(self.engine.calls, [])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_playback_receipt_for_other_entity_cannot_confirm_same_phones(self):
        self.first.edit("Ally", PHONES)
        self.second.edit("Ally", PHONES)
        preview = self.first.audition("Ally", PHONES)
        self.first.mark_played(preview)
        with self.assertRaises(ValueError):
            self.second.confirm(self.second.rows("Ally")[0])
        self.first.confirm(self.first.rows("Ally")[0])
        self.assertEqual(self.first.rows("Ally")[0]["status"], "confirmed")
        self.assertEqual(self.second.rows("Ally")[0]["status"], "pending")
        self.assertEqual(preview["request"]["scope"], self.first.scope)
        controls = preview["request"]["options"]["performance"]["controls"]
        self.assertEqual([item["text"] for item in controls], ["Ally"])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_queue_accepts_only_selected_entity_and_reopen_preserves_decisions(self):
        self.first.edit("Ally", PHONES)
        commands = iter(["s", "q"])
        with patch("epub_tts_narrator.renderer.terminal.wait_key", return_value=None):
            run_pronunciation_queue(self.first, object(), lambda player, path: True,
                input_fn=lambda prompt: next(commands), output=lambda message: None,
                interactive=True, redraw=lambda line: None, gap_seconds=0)
        self.assertEqual(self.first.rows("Ally")[0]["status"], "confirmed")
        self.assertEqual(self.second.rows("Ally")[0]["status"], "pending")
        path = Path(self.bundle["directory"]) / "scoped-pronunciations.json"
        before = path.read_bytes()
        commands = iter(["o all", "q"])
        run_pronunciation_queue(self.first, object(), lambda player, path: True,
            input_fn=lambda prompt: next(commands), output=lambda message: None,
            interactive=False, redraw=lambda line: None)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(load_pronunciations(self.bundle["directory"]), [])

    @unittest.skipUnless(HAS_FFMPEG, "FFmpeg and ffprobe required")
    def test_excerpt_centers_selected_entity_not_first_same_spelling_in_row(self):
        from epub_tts_narrator.name_script_state import mention_id, read_state
        from epub_tts_narrator.name_script_workflow import resolve_mention
        text = ("Ally softly left. " + "The road was quiet. " * 30 +
                "Ally waited beside the old gate while the others watched the winding road and wondered "
                "whether their friends would arrive before nightfall.")
        self.bundle = book(self.root / "long-book", text=text, two=True)
        prepare_name_script(self.bundle["directory"], {}, requests=requests(self.root, NameClient()))
        state = read_state(self.bundle["directory"])
        start = text.rindex("Ally")
        resolve_mention(self.bundle["directory"], mention_id(state, "b1", start, start + 4), "N2")
        for profile in self.bundle["cast"]["voices"]:
            self.store.publish(profile["id"], profile, self.root / "reference.wav", "Existing reference.",
                               self.engine.identity, {"passed": True}, {})
        session = self.session("N2")
        preview = session.audition("Ally", PHONES)
        self.assertTrue(preview["request"]["options"]["performance"]["controls"])
        self.assertIn("waited beside", preview["request"]["text"])
