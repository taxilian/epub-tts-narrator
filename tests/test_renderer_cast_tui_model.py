"""Casting screen logic: statuses, filters, details, and job-backed actions, without a terminal."""

import unittest

from epub_tts_narrator.renderer.cast_tui_model import (
    STATUS_FILTERS, CastController, candidate_rows, next_status_filter, voice_status,
)
from epub_tts_narrator.tui_jobs import FAILED, QUEUED, RUNNING
from cast_tui_fixtures import FakePlayer, FakeSession, cast_rows


class StatusTests(unittest.TestCase):
    def test_every_casting_state_has_a_distinct_status(self):
        statuses = {row["voice_id"]: voice_status(row)[0] for row in cast_rows()}
        self.assertEqual(statuses, {
            "v-ava": "accepted", "v-ben": "missing", "v-cole": "pending", "v-dee": "inherited",
            "v-eli": "failed", "v-fay": "shared", "v-ava-old": "derived",
        })

    def test_failed_status_carries_the_saved_error_and_jobs_override_saved_state(self):
        eli = next(row for row in cast_rows() if row["voice_id"] == "v-eli")
        self.assertEqual(voice_status(eli), ("failed", "verification failed"))
        self.assertEqual(voice_status(eli, {"v-eli": RUNNING})[0], "generating")
        self.assertEqual(voice_status(eli, {"v-eli": QUEUED})[0], "queued")

    def test_a_shared_voice_shows_as_busy_while_its_canonical_voice_generates(self):
        fay = next(row for row in cast_rows() if row["voice_id"] == "v-fay")
        self.assertEqual(voice_status(fay, {"v-ava": RUNNING})[0], "generating")

    def test_status_filters_cycle_through_every_choice_and_back(self):
        seen, current = [], None
        for _ in STATUS_FILTERS:
            current = next_status_filter(current)
            seen.append(current)
        self.assertEqual(seen[-1], None)
        self.assertEqual(set(seen), set(STATUS_FILTERS))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.player = FakePlayer()
        self.controller = CastController(self.session, self.player, config_path="config.toml",
                                         describe_error=str)
        self.addCleanup(self.controller.close)
        self.controller.refresh()

    def row(self, voice_id):
        return self.controller.row(voice_id)

    def settle(self):
        self.assertTrue(self.controller.jobs.wait_idle(5))

    def visible_ids(self):
        return [row["voice_id"] for row, _, _ in self.controller.visible()]

    def test_text_and_status_filters_combine_and_sorting_puts_attention_first(self):
        self.controller.filter.text = "ava"
        self.assertEqual(self.visible_ids(), ["v-ava", "v-ava-old"])
        self.controller.filter.text = ""
        self.controller.filter.status = "missing"
        self.assertEqual(self.visible_ids(), ["v-ben"])
        self.controller.filter.status = None
        self.controller.filter.by_status = True
        self.assertEqual(self.visible_ids()[:3], ["v-eli", "v-ben", "v-cole"])

    def test_details_name_every_editable_field_and_its_key(self):
        sections = {title: (value, key) for title, value, key in self.controller.details(self.row("v-ava"))}
        self.assertEqual(sections["Voice description"], ("A calm voice for Ava.", "e"))
        self.assertEqual(sections["Audition script"][1], "s")
        self.assertEqual(sections["Speech traits"], ("(none)", "t"))
        self.assertEqual(sections["Sharing"], ("Used by Fay", "d"))
        self.assertEqual(sections["Reference clips"], ("3 (default)", "c"))
        derived = {title: value for title, value, _ in self.controller.details(self.row("v-ava-old"))}
        self.assertIn("Clone direction", derived)
        self.assertEqual(derived["Derived from"], "v-ava")
        shared = {title: value for title, value, _ in self.controller.details(self.row("v-fay"))}
        self.assertEqual(shared["Sharing"], "Uses the voice of Ava")

    def test_takes_list_marks_the_accepted_one_and_counts_joined_clips(self):
        takes = candidate_rows(self.row("v-ava"))
        self.assertEqual([(item["candidate_id"], item["accepted"], item["clips"]) for item in takes],
                         [("c-ava-1", False, 1), ("c-ava-2", True, 3)])
        self.assertEqual(takes[0]["designer"], "gemini")
        inherited = candidate_rows(self.row("v-dee"))
        self.assertEqual((inherited[0]["designer"], inherited[0]["accepted"]), ("inherited", True))

    def test_play_uses_the_accepted_take_or_the_chosen_one(self):
        self.controller.play(self.row("v-ava"))
        self.controller.play(self.row("v-ava"), "c-ava-1")
        self.assertEqual(self.player.started, ["/voices/c-ava-2.wav", "/voices/c-ava-1.wav"])
        with self.assertRaisesRegex(ValueError, "No accepted take"):
            self.controller.play(self.row("v-ben"))
        self.controller.stop()
        self.assertIsNone(self.controller.playing)

    def test_comparison_alternates_between_the_marked_take_and_another(self):
        ava = self.row("v-ava")
        self.assertIsNone(self.controller.compare_partner(ava, "c-ava-2"))
        self.controller.mark_compare(ava, "c-ava-1")
        self.assertEqual(self.controller.compare_partner(ava, "c-ava-2"), "c-ava-1")
        self.assertIsNone(self.controller.compare_partner(ava, "c-ava-1"))
        self.assertIsNone(self.controller.compare_partner(self.row("v-cole"), "c-cole-1"))

    def test_actions_run_the_matching_session_call_in_the_background(self):
        ava = self.row("v-ava")
        self.controller.regenerate(ava, "gemini")
        self.controller.accept(ava, "c-ava-1")
        self.controller.save_field(ava, "speech_traits", "")
        self.controller.share(self.row("v-ben"), ava)
        self.controller.stop_sharing(self.row("v-fay"))
        self.controller.set_clips(ava, 2)
        self.controller.set_clips(ava, None)
        self.controller.refresh_prompt(ava)
        self.settle()
        saves = {"update_prompt", "set_reference_clips"}  # quick saves may run beside the queue
        self.assertEqual([call for call in self.session.calls if call[0] not in saves], [
            ("regenerate", ("v-ava",), {"voice_designer": "gemini"}),
            ("accept", ("v-ava", "c-ava-1"), {}),
            ("share_voice", ("v-ben", "v-ava"), {}),
            ("stop_sharing", ("v-fay",), {}),
            ("refresh_prompts", (), {"voice_ids": ["v-ava"], "config_path": "config.toml"}),
        ])
        self.assertEqual([call for call in self.session.calls if call[0] in saves], [
            ("update_prompt", ("v-ava",), {"speech_traits": ""}),
            ("set_reference_clips", ("v-ava", 2), {}),
            ("set_reference_clips", ("v-ava", None), {}),
        ])
        self.assertTrue(any(text.startswith("Done: Generate Ava with Gemini")
                            for _, _, text in self.controller.messages))

    def test_the_accent_cue_is_shown_offered_from_the_description_and_saved(self):
        ava = self.row("v-ava")
        ava["accent_suggestion"] = "light Irish accent"
        sections = {title: (value, key) for title, value, key in self.controller.details(ava)}
        self.assertEqual(sections["Accent cue (said before every line)"],
                         ("(none); suggested: light Irish accent", "A"))
        self.assertEqual(self.controller.field_value(ava, "accent"), "light Irish accent")
        self.controller.save_field(ava, "accent", "light Irish accent")
        with self.assertRaisesRegex(ValueError, "at most 80"):
            self.controller.save_field(ava, "accent", "x" * 81)
        self.settle()
        self.assertEqual(self.session.calls, [("update_prompt", ("v-ava",), {"accent": "light Irish accent"})])

    def test_invalid_edits_fail_before_anything_is_queued(self):
        ava = self.row("v-ava")
        for field, value in (("prompt", ""), ("reference_text", "x" * 1001), ("speech_traits", "[shouting]"),
                             ("nickname", "x")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.controller.save_field(ava, field, value)
        self.assertEqual(self.controller.jobs.jobs(), [])
        with self.assertRaisesRegex(ValueError, "clone direction"):
            self.controller.refresh_prompt(self.row("v-ava-old"))
        with self.assertRaisesRegex(ValueError, "different voice"):
            self.controller.share(ava, ava)

    def test_a_failed_job_is_listed_and_retried_in_place(self):
        self.session.fail["regenerate"] = ValueError("Gemini quota exceeded")
        self.controller.regenerate(self.row("v-eli"), "gemini")
        self.controller.set_clips(self.row("v-ava"), 1)
        self.settle()
        [failure] = self.controller.failures()
        self.assertEqual((failure.state, failure.error), (FAILED, "Gemini quota exceeded"))
        self.assertTrue(any(severity == "error" and "quota" in text for _, severity, text in self.controller.messages))
        self.assertEqual(self.session.calls[-1][0], "set_reference_clips")
        self.controller.retry(failure.job_id)
        self.settle()
        self.assertEqual(self.controller.failures(), [])
        self.assertEqual(self.session.calls[-1], ("regenerate", ("v-eli",), {"voice_designer": "gemini"}))

    def test_startup_prepares_only_voices_without_an_accepted_take(self):
        self.controller.startup()
        self.settle()
        self.assertEqual(self.session.calls[0][0], "prepare_samples")
        self.assertEqual(self.session.calls[1], ("ensure_ready_many", (["v-ben", "v-cole"],), {}))

    def test_regeneration_warns_about_shared_inherited_and_accepted_voices(self):
        self.assertIn("Fay", self.controller.regenerate_warning(self.row("v-ava")))
        self.assertIn("inherited", self.controller.regenerate_warning(self.row("v-dee")))
        self.assertEqual(self.controller.regenerate_warning(self.row("v-ben")), "")

    def test_the_session_s_notices_reach_the_screen_log(self):
        self.session.notify("Selected Gemini voice ID: abc")
        self.assertEqual(self.controller.messages[-1][1:], ("info", "Selected Gemini voice ID: abc"))


if __name__ == "__main__":
    unittest.main()


class ParallelAndImmediateTests(unittest.TestCase):
    """Gemini voices generate up to three at once; saving never waits for another voice's generation."""

    def controller(self, session):
        controller = CastController(session, FakePlayer(), describe_error=str)
        self.addCleanup(controller.close)
        controller.refresh()
        return controller

    def test_three_gemini_generations_run_at_once_through_one_pool(self):
        import threading
        from unittest.mock import patch
        gate, lock, running = threading.Event(), threading.Lock(), {"now": 0, "most": 0}
        opened = []

        class Pool:
            def __init__(self, session, limit, designer):
                opened.append((limit, designer))

            def open(self):
                pass

            def regenerate(self, voice_id):
                with lock:
                    running["now"] += 1
                    running["most"] = max(running["most"], running["now"])
                gate.wait(5)
                with lock:
                    running["now"] -= 1
                return voice_id

            def close(self):
                opened.append("closed")
        session = FakeSession(config={"casting": {"reference_clips": 3, "gemini_concurrency": 4},
                                      "voice_design": {"type": "gemini"}})
        controller = self.controller(session)
        with patch("epub_tts_narrator.renderer.casting_batch.GenerationPool", Pool):
            for voice_id in ("v-ava", "v-ben", "v-cole", "v-eli"):
                controller.regenerate(controller.row(voice_id), "gemini")
            controller.regenerate(controller.row("v-dee"), "local")  # local design stays in line, alone
            for _ in range(100):
                if running["now"] == 3:
                    break
                threading.Event().wait(0.02)
            self.assertEqual(running["now"], 3)
            gate.set()
            self.assertTrue(controller.jobs.wait_idle(5))
        self.assertEqual(running["most"], 3)
        self.assertEqual(opened, [(3, "gemini"), "closed"])  # one pool for the run of Gemini voices
        self.assertEqual([call[0] for call in session.calls], ["regenerate"])  # only the local one

    def test_a_save_runs_at_once_unless_its_own_voice_is_busy(self):
        import threading
        gate = threading.Event()
        session = FakeSession()
        session.regenerate = lambda voice_id, voice_designer=None: gate.wait(5)
        controller = self.controller(session)
        controller.regenerate(controller.row("v-ava"), "local")
        other = controller.save_field(controller.row("v-ben"), "speech_traits", "")
        own = controller.save_field(controller.row("v-ava"), "speech_traits", "")
        for _ in range(100):
            if other.state == "done":
                break
            threading.Event().wait(0.02)
        self.assertEqual(other.state, "done")  # another voice's save did not wait
        self.assertEqual(own.state, QUEUED)    # this voice's save follows its generation
        gate.set()
        self.assertTrue(controller.jobs.wait_idle(5))
        self.assertEqual(own.state, "done")
