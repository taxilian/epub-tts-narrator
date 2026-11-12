"""Pipeline lifecycle: mode identity, warm cache reuse, and shared take dedup.

Locks end-to-end lifecycle behavior the lane/scheduler unit tests do not
cover: switching scheduler modes over one book consumes exactly the same
accepted caches with zero new generation, and two chapters presenting the
identical performance request share one take lock, one retry series, and
one accepted take instead of racing over the same attempt files.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.engines import DRAMA3_MODEL_REVISION
from epub_tts_narrator.renderer.interrupts import RenderInterrupted, RenderInterrupts
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import (
    ChaptersRenderFailed, RenderBlocked, VerificationUnavailable, make_render_plan, render,
)
from audition_fixtures import prepare_bundle
from shared_fixtures import FFMPEG, FFPROBE, renderer_config, timed_transcript
from test_chapter_export import chapter_document
from test_renderer_workflow import EXAMPLE, FixtureEngine, FixtureVerifier, pcm_key

WATCHDOG = 30


def deadline_loop(seconds, condition):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


class FirstAttemptFailsVerifier(FixtureVerifier):
    """Rejects the marker text exactly once, then verifies normally."""

    def __init__(self, fixture_transcripts, fail_text):
        super().__init__(fixture_transcripts)
        self.fail_text = fail_text
        self.remaining_failures = 1

    def transcribe(self, path):
        self.calls.append(str(path))
        if self.remaining_failures and self.fixture_transcripts[pcm_key(path)] == self.fail_text:
            self.remaining_failures -= 1
            return timed_transcript("Deliberately wrong recognized words for the first attempt.")
        return FixtureVerifier.transcribe(self, path)


class ThreadRecordingEngine(FixtureEngine):
    """Records the thread identity of every synthesis call."""

    def __init__(self, fixture_transcripts):
        super().__init__(fixture_transcripts)
        self.lock = threading.Lock()
        self.model_threads = []

    def synthesize(self, text, voice, path, *, seed, direction=""):
        with self.lock:
            self.model_threads.append(threading.get_ident())
        return FixtureEngine.synthesize(self, text, voice, path, seed=seed, direction=direction)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class PipelineLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = renderer_config(self.root, series_id="pipeline-lifecycle",
                                      checkpoint="unused-tts", asr_model="unused-asr",
                                      max_attempts=3, paragraph_synthesis=False)
        self.config["render_pool"] = {
            "workers": [{"name": "local", "type": "local", "concurrency": 1}],
            "model_revision": DRAMA3_MODEL_REVISION, "scheduler": "legacy",
            "chapter_concurrency": 2,
        }
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = FixtureVerifier(self.fixtures)

    def pool_for(self, scheduler, engine=None):
        config = deepcopy(self.config)
        config["render_pool"]["scheduler"] = scheduler
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        pool._workers[0]._engine = engine or self.engine
        return pool

    def publish_voices(self, pool):
        store = VoiceStore(self.config["voices"]["directory"], "pipeline-lifecycle")
        for profile in self.bundle["cast"]["voices"]:
            reference = self.root / (profile["id"] + ".wav")
            self.engine._write("Reference words.", reference, 17)
            store.publish(profile["id"], profile, reference, "Reference words.",
                          pool.voice_binding_identity, {"passed": True}, {})

    def render_chapters(self, pool, verifier=None):
        return render(self.bundle, self.config, pool, verifier or self.verifier,
                      notify=lambda _: None, chapters_only=True)

    def accepted_records(self):
        return sorted((self.root / "work" / "clips").glob("*/accepted.json"))

    def clear_chapter_checkpoints(self):
        for path in (self.root / "work" / "completed-chapters").glob("*.json"):
            path.unlink()

    def test_scheduler_mode_switch_reuses_accepted_caches_without_new_generation(self):
        legacy = self.pool_for("legacy")
        pipeline = self.pool_for("pipeline")
        self.assertEqual(pipeline.identity, legacy.identity)
        self.publish_voices(legacy)
        self.render_chapters(legacy)
        first_calls = len(self.engine.speech_calls)
        self.assertGreater(first_calls, 0)

        def accepted_snapshot():
            return {path.parent.name: read_json(path) for path in self.accepted_records()}

        def audio_snapshot(records):
            return {name: file_hash(self.root / "work" / "clips" / name / record["audio_file"])
                    for name, record in records.items()}

        def cache_snapshot():
            return {path.name: read_json(path)
                    for path in (self.root / "work" / "speech-cache").glob("*.json")}

        before = accepted_snapshot()
        audio = audio_snapshot(before)
        indices = cache_snapshot()

        # Clip-level reuse, not the chapter checkpoint: drop completed-chapter
        # state so every chunk must resolve through the accepted speech cache.
        self.clear_chapter_checkpoints()
        self.render_chapters(pipeline)
        self.assertEqual(len(self.engine.speech_calls), first_calls)
        self.clear_chapter_checkpoints()
        self.render_chapters(legacy)
        self.assertEqual(len(self.engine.speech_calls), first_calls)

        after = accepted_snapshot()
        self.assertEqual(after, before)
        self.assertEqual(audio_snapshot(after), audio)
        self.assertEqual(cache_snapshot(), indices)

    def test_identical_chapter_text_shares_one_take_retry_series_and_acceptance(self):
        marker = self.bundle["chapters"][0]["rows"][0]
        first = {"entry": deepcopy(self.bundle["chapters"][0]["entry"]), "rows": [marker]}
        first["entry"].update(title="Marker chapter one",
                              path="scripts/chapter-0001.jsonl")
        second = {"entry": deepcopy(self.bundle["chapters"][0]["entry"]),
                  "rows": [deepcopy(marker)]}
        second["entry"].update(title="Marker chapter two",
                               path="scripts/chapter-0002.jsonl")
        self.bundle["chapters"] = [first, second]
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        verifier = FirstAttemptFailsVerifier(self.fixtures, marker["text"])
        self.render_chapters(pool, verifier)

        marker_calls = [call for call in self.engine.speech_calls
                        if call["text"] == marker["text"]]
        self.assertEqual(len(marker_calls), 2)
        self.assertEqual(len({call["seed"] for call in marker_calls}), 2)
        # The second chapter waited on the shared take lock and reused the
        # accepted take: no extra generation, and the failed first attempt is
        # never re-run. The accepted audio may be re-transcribed once per
        # cache-hit validation, so only distinct files and counts bind.
        self.assertEqual(verifier.remaining_failures, 0)
        accepted = self.accepted_records()
        self.assertEqual(len(accepted), 1)
        attempts = sorted(accepted[0].parent.glob("attempt-*.json"))
        self.assertEqual(len(attempts), 2)
        records = [read_json(path) for path in attempts]
        recorded_names = {record["attempt"]: record["audio_file"] for record in records}
        transcribed = [Path(path).name for path in verifier.calls]
        self.assertEqual(set(transcribed), set(recorded_names.values()))
        self.assertEqual(transcribed.count(recorded_names[1]), 1)
        self.assertGreaterEqual(transcribed.count(recorded_names[2]), 1)
        self.assertIs(read_json(attempts[0])["report"]["passed"], False)
        self.assertIs(read_json(accepted[0])["report"]["passed"], True)

    def test_failed_chapter_continues_beyond_limit_and_resume_reuses_completed_chapters(self):
        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The first harbor remained silent.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("The second boat crossed the harbor.", "narration", "p"),
            ("CHAPTER THREE", "heading", "p"),
            ("The third boat crossed the harbor.", "narration", "p"),
            ("CHAPTER FOUR", "heading", "p"),
            ("The fourth boat crossed the harbor.", "narration", "p"),
            ("CHAPTER FIVE", "heading", "p"),
            ("The fifth boat crossed the harbor.", "narration", "p"),
        ])
        export_bundle(document, self.root / "continued-bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "continued-bundle"))
        self.config["render"]["max_attempts"] = 2
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        self.verifier.wrong_texts.add("The first harbor remained silent.")

        with self.assertRaises(RenderBlocked):
            render(self.bundle, self.config, pool, self.verifier, notify=lambda _: None)

        checkpoints = {
            read_json(path)["chapter_number"]: path
            for path in (self.root / "work" / "completed-chapters").glob("*.json")
        }
        self.assertEqual(set(checkpoints), {2, 3, 4, 5})
        completed_audio = {
            number: file_hash(read_json(path)["audio_file"])
            for number, path in checkpoints.items()
        }
        self.assertFalse((self.root / "output").exists())
        status = read_json(self.root / "work" / "status.json")
        self.assertEqual(status["status"], "blocked")
        self.assertEqual([item["number"] for item in status["failed_chapters"]], [1])
        self.assertEqual(status["failed_chapters"][0]["error_type"], "SpeechVerificationFailed")
        self.assertIn("after 2 attempts", status["failed_chapters"][0]["error"])
        failed_chunk = next(
            chunk for chunk in make_render_plan(self.bundle, self.config, multi_speaker=False)["chapters"][0]["chunks"]
            if chunk["spoken_text"] == "The first harbor remained silent.")
        linkage = status["failed_chapters"][0]
        self.assertEqual(linkage["segment_ids"], [failed_chunk["segment"]["id"]])
        self.assertEqual(linkage["chunk_id"], failed_chunk["chunk_id"])
        failed_attempts = sorted(
            (self.root / "work" / "clips" / linkage["request_hash"]).glob("attempt-*.json"))
        self.assertEqual(len(failed_attempts), 2)

        self.verifier.wrong_texts.clear()
        before = len(self.engine.speech_calls)
        result = render(self.bundle, self.config, pool, self.verifier, notify=lambda _: None)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.engine.speech_calls) - before, 1)
        resumed = {
            read_json(path)["chapter_number"]: path
            for path in (self.root / "work" / "completed-chapters").glob("*.json")
        }
        self.assertEqual(set(resumed), {1, 2, 3, 4, 5})
        for number, checksum in completed_audio.items():
            self.assertEqual(file_hash(read_json(resumed[number])["audio_file"]), checksum)
        self.assertTrue((self.root / "output" / result["audio"]["audiobook"]["path"]).is_file())

    def test_permanent_chapter_generation_failure_continues_remaining_chapters(self):
        class PermanentFailure(RuntimeError):
            retryable = False

        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The first boat crossed the harbor.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("The second boat crossed the harbor.", "narration", "p"),
            ("CHAPTER THREE", "heading", "p"),
            ("The third boat crossed the harbor.", "narration", "p"),
            ("CHAPTER FOUR", "heading", "p"),
            ("The fourth boat crossed the harbor.", "narration", "p"),
        ])
        export_bundle(document, self.root / "blocked-bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "blocked-bundle"))
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        original = self.engine.synthesize

        def generate(*args, **kwargs):
            if args[0] == "The second boat crossed the harbor.":
                raise PermanentFailure("Fixture permanent generation error")
            return original(*args, **kwargs)

        with patch.object(self.engine, "synthesize", side_effect=generate):
            with self.assertRaises(ChaptersRenderFailed) as raised:
                render(self.bundle, self.config, pool, self.verifier, notify=lambda _: None)

        self.assertEqual([chapter["number"] for chapter in raised.exception.failed_chapters], [2])
        self.assertIn("Chapter 2 (RenderBlocked): Fixture permanent generation error",
                      str(raised.exception))
        checkpoints = {
            read_json(path)["chapter_number"]
            for path in (self.root / "work" / "completed-chapters").glob("*.json")
        }
        self.assertEqual(checkpoints, {1, 3, 4})
        status = read_json(self.root / "work" / "status.json")
        self.assertEqual(status["status"], "blocked")
        self.assertEqual([item["number"] for item in status["failed_chapters"]], [2])
        self.assertEqual(status["failed_chapters"][0]["error_type"], "RenderBlocked")
        failed_chunk = next(
            chunk for chunk in make_render_plan(self.bundle, self.config, multi_speaker=False)["chapters"][1]["chunks"]
            if chunk["spoken_text"] == "The second boat crossed the harbor.")
        linkage = status["failed_chapters"][0]
        self.assertEqual(linkage["segment_ids"], [failed_chunk["segment"]["id"]])
        self.assertEqual(linkage["chunk_id"], failed_chunk["chunk_id"])
        failed_attempts = sorted(
            (self.root / "work" / "clips" / linkage["request_hash"]).glob("attempt-*.json"))
        self.assertEqual(len(failed_attempts), 1)
        self.assertFalse((self.root / "output").exists())

    def _four_chapter_bundle(self, name):
        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The first boat crossed the harbor.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("The second boat crossed the harbor.", "narration", "p"),
            ("CHAPTER THREE", "heading", "p"),
            ("The third boat crossed the harbor.", "narration", "p"),
            ("CHAPTER FOUR", "heading", "p"),
            ("The fourth boat crossed the harbor.", "narration", "p"),
        ])
        export_bundle(document, self.root / name)
        self.bundle = prepare_bundle(load_bundle(self.root / name))

    def _render_in_shell(self, scheduler, on_event):
        """Render with a run shell's board; ``on_event`` sees each run event as it happens."""
        from epub_tts_narrator.renderer.progress import shell_board
        from epub_tts_narrator.run_events import EventBus

        class PermanentFailure(RuntimeError):
            retryable = False

        pool = self.pool_for(scheduler)
        self.publish_voices(pool)
        original = self.engine.synthesize
        self.broken = True

        def generate(*args, **kwargs):
            if self.broken and args[0] == "The second boat crossed the harbor.":
                raise PermanentFailure("Fixture permanent generation error")
            return original(*args, **kwargs)

        bus = EventBus()
        bus.subscribe(on_event)
        with patch.object(self.engine, "synthesize", side_effect=generate), shell_board(bus) as board:
            return render(self.bundle, self.config, pool, self.verifier, notify=board.log)

    def _retry_in_place(self, scheduler):
        self._four_chapter_bundle("retry-bundle")
        events = []

        lines = []

        def on_event(event):
            events.append(event.kind)
            if event.kind == "log":
                lines.append(event.text)
            if event.kind == "failure":
                self.broken = False  # the person fixed the cause, then pressed r
                self.assertTrue(event.data["retry"]())

        result = self._render_in_shell(scheduler, on_event)
        self.assertEqual(result["status"], "complete")
        retried = next(index for index, line in enumerate(lines) if line.startswith("Retrying chapter 2 / 4 first"))
        if scheduler == "legacy":
            # A sequential engine renders the retried chapter next, ahead of the chapters still waiting.
            later = next(index for index, line in enumerate(lines) if line.startswith("Processing chapter 3 / 4"))
            self.assertLess(retried, later)
        self.assertEqual(events.count("failure"), 1)
        self.assertEqual(read_json(self.root / "work" / "status.json")["status"], "complete")
        self.assertTrue((self.root / "output" / result["audio"]["audiobook"]["path"]).is_file())

    def test_a_shell_retries_a_failed_chapter_while_the_pipeline_continues(self):
        self._retry_in_place("pipeline")

    def test_the_sequential_scheduler_retries_after_its_main_pass(self):
        self._retry_in_place("legacy")

    def test_a_person_accepts_a_failed_take_and_the_chapter_continues_from_it(self):
        from epub_tts_narrator.renderer.progress import shell_board
        from epub_tts_narrator.run_events import EventBus
        self._four_chapter_bundle("listened-bundle")
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        self.verifier.wrong_texts = {"The second boat crossed the harbor."}
        seen = []

        def on_event(event):
            if event.kind == "failure":
                repair = event.data["repair"]
                takes = repair.takes()
                seen.append([(row["attempt"], row["passed"], row["eligible"]) for row in takes])
                outcome = repair.accept(takes[0]["attempt"])
                seen.append(outcome["basis"])
                self.assertTrue(event.data["retry"]())

        bus = EventBus()
        bus.subscribe(on_event)
        with shell_board(bus) as board:
            result = render(self.bundle, self.config, pool, self.verifier, notify=board.log)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(seen[0][0], (1, False, True))
        self.assertTrue(all(not passed for _, passed, _ in seen[0]))
        self.assertEqual(seen[1], "manual_take_selection")
        self.assertTrue((self.root / "output" / result["audio"]["audiobook"]["path"]).is_file())

    def test_node_activity_shows_which_block_holds_each_speech_and_verify_slot(self):
        from epub_tts_narrator.renderer.node_activity import ACTIVITY
        self._four_chapter_bundle("nodes-bundle")
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        seen = {"speech": [], "verify": []}
        original_synthesize, original_transcribe = self.engine.synthesize, self.verifier.transcribe

        def jobs(role):
            return [(node["name"], item["chapter"], item["block"]) for node in ACTIVITY.snapshot()
                    if node["role"] == role for item in node["jobs"]]

        def synthesize(*args, **kwargs):
            seen["speech"].extend(jobs("speech"))
            return original_synthesize(*args, **kwargs)

        def transcribe(path):
            seen["verify"].extend(jobs("verify"))
            return original_transcribe(path)

        with (patch.object(self.engine, "synthesize", side_effect=synthesize),
              patch.object(self.verifier, "transcribe", side_effect=transcribe)):
            render(self.bundle, self.config, pool, self.verifier, notify=lambda _: None)
        self.assertIn(("local", 2, 2), seen["speech"])
        self.assertTrue(all(name == "local" and chapter and block for name, chapter, block in seen["verify"]))
        self.assertTrue(seen["verify"])
        self.assertEqual({(node["name"], node["role"], node["slots"]) for node in ACTIVITY.snapshot()},
                         {("local", "speech", 1), ("local", "verify", 1)})
        pool.close()
        self.assertEqual(ACTIVITY.snapshot(), [])

    def test_changing_a_finished_chapter_s_take_rebuilds_it_before_the_book_is_assembled(self):
        from epub_tts_narrator.renderer.progress import shell_board
        from epub_tts_narrator.run_events import EventBus
        self._four_chapter_bundle("rebuild-bundle")
        self.verifier = FirstAttemptFailsVerifier(self.fixtures, "The second boat crossed the harbor.")
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        original = self.engine.synthesize
        broken = {"The fourth boat crossed the harbor."}

        class PermanentFailure(RuntimeError):
            retryable = False

        def generate(*args, **kwargs):
            if args[0] in broken:
                raise PermanentFailure("Fixture permanent generation error")
            return original(*args, **kwargs)

        state, lines = {}, []

        def on_event(event):
            if event.kind == "log":
                lines.append(event.text)
            elif event.kind == "rejected":
                state["review"] = event.data["review"]
            elif event.kind == "rebuild":
                state["rebuild"] = event.data["handler"]
            elif event.kind == "failure":
                state["retry"] = event.data["retry"]
            elif event.kind == "hold" and event.data["release"] is not None and "done" not in state:
                # While chapter 4 waits for a retry: a person prefers chapter 2's rejected take.
                state["done"] = True
                takes = state["review"].takes()
                self.assertEqual([(row["attempt"], row["passed"], row["accepted"]) for row in takes],
                                 [(1, False, False), (2, True, True)])
                self.assertEqual(state["review"].accept(1)["basis"], "manual_take_selection")
                self.assertTrue(state["rebuild"](2))
                broken.clear()
                self.assertTrue(state["retry"]())

        bus = EventBus()
        bus.subscribe(on_event)
        with patch.object(self.engine, "synthesize", side_effect=generate), shell_board(bus) as board:
            result = render(self.bundle, self.config, pool, self.verifier, notify=board.log)
        self.assertEqual(result["status"], "complete")
        # Chapter 2 was assembled again, with the take the person chose (no new generation).
        self.assertEqual(sum(line.startswith("Assembling chapter 2 / 4") for line in lines), 2)
        self.assertEqual([(row["attempt"], row["accepted"]) for row in state["review"].takes()], [(1, True), (2, False)])
        self.assertEqual(state["review"].used()["attempt"], 1)

    def test_a_held_render_reports_its_failures_when_the_person_finishes(self):
        self._four_chapter_bundle("held-bundle")
        holds = []

        def on_event(event):
            if event.kind == "hold" and event.data["release"] is not None:
                holds.append(event.data["failed"])
                event.data["release"]()

        with self.assertRaises(ChaptersRenderFailed) as raised:
            self._render_in_shell("pipeline", on_event)
        self.assertEqual(holds, [1])
        self.assertEqual([chapter["number"] for chapter in raised.exception.failed_chapters], [2])

    def test_cache_integrity_failure_keeps_failed_request_linkage(self):
        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The first boat crossed the harbor.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("The second boat crossed the harbor.", "narration", "p"),
            ("CHAPTER THREE", "heading", "p"),
            ("The third boat crossed the harbor.", "narration", "p"),
        ])
        export_bundle(document, self.root / "cache-bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "cache-bundle"))
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        self.render_chapters(pool)

        # Tamper with one accepted take's audio: its cache lookup raises
        # ValueError inside execute_take, which keeps the failed request.
        index = read_json(sorted((self.root / "work" / "speech-cache").glob("*.json"))[0])
        accepted = read_json(self.root / "work" / "clips" / index["request_hash"] / "accepted.json")
        (self.root / "work" / "clips" / index["request_hash"] / accepted["audio_file"]).write_bytes(b"tampered")
        self.clear_chapter_checkpoints()

        with self.assertRaises(ChaptersRenderFailed) as raised:
            self.render_chapters(pool)

        self.assertEqual(len(raised.exception.failed_chapters), 1)
        failure = raised.exception.failed_chapters[0]
        self.assertEqual(failure["error_type"], "RenderBlocked")
        self.assertIn("audio changed", failure["error"])
        plan = make_render_plan(self.bundle, self.config, multi_speaker=False)
        known_chunks = {chunk["chunk_id"]: chunk for chapter in plan["chapters"]
                        for chunk in chapter["chunks"]}
        failed_chunk = known_chunks[failure["chunk_id"]]
        self.assertEqual(failure["segment_ids"], [failed_chunk["segment"]["id"]])
        self.assertIsInstance(failure["request_hash"], str)
        checkpoints = {
            read_json(path)["chapter_number"]
            for path in (self.root / "work" / "completed-chapters").glob("*.json")
        }
        self.assertEqual(checkpoints, {1, 2, 3} - {failure["number"]})
        status = read_json(self.root / "work" / "status.json")
        self.assertEqual(status["status"], "blocked")
        self.assertEqual(status["failed_chapters"][0]["chunk_id"], failure["chunk_id"])

    def test_failure_linkage_invents_no_ids_without_a_request(self):
        from epub_tts_narrator.renderer.workflow import _failure_linkage
        self.assertEqual(_failure_linkage(RenderBlocked("chapter-level block")), {})
        self.assertEqual(_failure_linkage(RenderBlocked("bad cache", request={"kind": "speech"})), {})

    def test_verification_outage_stops_book_admission(self):
        import epub_tts_narrator.renderer.workflow as workflow

        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The first boat crossed the harbor.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("The second boat crossed the harbor.", "narration", "p"),
            ("CHAPTER THREE", "heading", "p"),
            ("The third boat crossed the harbor.", "narration", "p"),
            ("CHAPTER FOUR", "heading", "p"),
            ("The fourth boat crossed the harbor.", "narration", "p"),
        ])
        export_bundle(document, self.root / "outage-bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "outage-bundle"))
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        messages = []
        original_validation = workflow.validate_audio

        def validate(*args, **kwargs):
            report = original_validation(*args, **kwargs)
            return {**report, "passed": False, "reasons": ["text_mismatch"],
                    "text_judge": {"decision": "unavailable", "error_code": "fixture_timeout"}}

        with patch.object(workflow, "validate_audio", side_effect=validate):
            with self.assertRaises(VerificationUnavailable):
                render(self.bundle, self.config, pool, self.verifier, notify=messages.append)

        # An outage is run-fatal: only the two admitted chapters generated one
        # take each, and pending chapters were never admitted.
        self.assertEqual(len(self.engine.speech_calls), 2)
        self.assertFalse(any("Processing chapter 3" in message for message in messages))
        self.assertFalse(list((self.root / "work" / "completed-chapters").glob("*.json")))
        status = read_json(self.root / "work" / "status.json")
        self.assertEqual(status["status"], "blocked")
        self.assertEqual(status["error_type"], "VerificationUnavailable")
        self.assertNotIn("failed_chapters", status)
        self.assertFalse((self.root / "output").exists())

    def test_interrupt_after_chapter_failure_stops_admission_and_publishes_no_partial_book(self):
        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The first harbor remained silent.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("The second boat crossed the harbor.", "narration", "p"),
            ("CHAPTER THREE", "heading", "p"),
            ("The third boat crossed the harbor.", "narration", "p"),
            ("CHAPTER FOUR", "heading", "p"),
            ("The fourth boat crossed the harbor.", "narration", "p"),
            ("CHAPTER FIVE", "heading", "p"),
            ("The fifth boat crossed the harbor.", "narration", "p"),
        ])
        export_bundle(document, self.root / "interrupted-bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "interrupted-bundle"))
        self.config["render"]["max_attempts"] = 2
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        self.verifier.wrong_texts.add("The first harbor remained silent.")
        original = self.engine.synthesize
        with RenderInterrupts() as control:
            def generate(*args, **kwargs):
                result = original(*args, **kwargs)
                if args[0] == "The second boat crossed the harbor.":
                    control.requested = True
                return result

            with patch.object(self.engine, "synthesize", side_effect=generate):
                with self.assertRaises(RenderInterrupted):
                    render(self.bundle, self.config, pool, self.verifier, notify=lambda _: None)

        # Chapter 1 was recorded as failed; the interruption during chapter 2
        # then stopped admission, so no chapter reached a checkpoint and no
        # partial audiobook was published.
        self.assertFalse(list((self.root / "work" / "completed-chapters").glob("*.json")))
        status = read_json(self.root / "work" / "status.json")
        self.assertEqual(status["status"], "interrupted")
        self.assertFalse((self.root / "output").exists())

        self.verifier.wrong_texts.clear()
        result = render(self.bundle, self.config, pool, self.verifier, notify=lambda _: None)
        self.assertEqual(result["status"], "complete")
        resumed = {
            read_json(path)["chapter_number"]
            for path in (self.root / "work" / "completed-chapters").glob("*.json")
        }
        self.assertEqual(resumed, {1, 2, 3, 4, 5})
        self.assertTrue((self.root / "output" / result["audio"]["audiobook"]["path"]).is_file())

    def test_pipeline_local_workers_share_one_persistent_model_thread(self):
        self.config["render_pool"]["workers"] = [
            {"name": "local", "type": "local", "concurrency": 2}]
        self.config["render_pool"]["scheduler"] = "pipeline"
        pool = RenderPool(deepcopy(self.config))
        self.addCleanup(pool.close)
        engine = ThreadRecordingEngine(self.fixtures)
        pool._workers[0]._engine = engine
        caller = ThreadPoolExecutor(max_workers=4)
        self.addCleanup(caller.shutdown)
        caller_threads = []

        def submit(index):
            caller_threads.append(threading.get_ident())
            return pool.synthesize(f"shared words {index}", {"asset_hash": "fixture"},
                                   str(self.root / f"t{index}.wav"), seed=index)

        futures = [caller.submit(submit, index) for index in range(4)]
        for future in futures:
            future.result(WATCHDOG)
        self.assertEqual(len(engine.model_threads), 4)
        # Every lane worker delegated to the one persistent LocalRuntime
        # thread; no caller thread ran a model call itself.
        self.assertEqual(len(set(engine.model_threads)), 1)
        self.assertGreaterEqual(len(set(caller_threads)), 2)
        self.assertNotIn(engine.model_threads[0], set(caller_threads))


if __name__ == "__main__":
    unittest.main()
