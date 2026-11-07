"""History echo: detection, the recovery ladder, and its engine and transport paths."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import importlib.util
import re
import tempfile
import threading
import unittest

from epub_tts_narrator.renderer import continuity
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.engines import Drama3MLXEngine, EngineError
from epub_tts_narrator.renderer.history_echo import (
    LADDER, apply_recovery, archive, candidate_echo, detect, next_mode, plain_text, rejected_pick, role_tagged)
from epub_tts_narrator.renderer.workflow import _render_chapter_chunks
from shared_fixtures import renderer_config
import test_renderer_continuity as fixtures
from test_renderer_paragraphs import passages
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


def tagged(text):
    return f"<|speaker:0|> {text}"


@contextmanager
def heard(*texts):
    """A chapter transaction scope whose history turns have these texts."""
    takes = [{"identity": {"request_hash": f"h{index}"}, "marker": {"text": text}}
             for index, text in enumerate(texts)]
    token = continuity._CURRENT.set({"history": True, "takes": takes})
    try:
        yield
    finally:
        continuity._CURRENT.reset(token)


class DetectionTests(unittest.TestCase):
    def test_a_take_speaking_the_previous_line_is_an_echo(self):
        evidence = detect("Sal muttered quietly to Upgrade, who was nibbling at her thumbnail.",
                          " I know it works, but I can't help but feel nervous.",
                          [("pinned", tagged("Quest asked as the doors slid open.")),
                           ("latest", tagged("[Quiet, uneasy mutter] I know it works, but I can’t help but feel nervous."))])
        self.assertEqual(evidence["heard_request_hash"], "latest")
        self.assertEqual((evidence["heard_share"], evidence["expected_share"]), (1.0, 0.0))

    def test_a_partial_echo_inside_a_repair_join_is_still_an_echo(self):
        self.assertIsNotNone(detect(
            "Sal muttered quietly to Upgrade, who was nibbling at her thumbnail.",
            " I know it works, but I can't help but feel nervous, who was nibbling at her thumbnail.",
            [("latest", tagged("I know it works, but I can’t help but feel nervous."))]))

    def test_a_line_that_really_repeats_earlier_words_is_not_an_echo(self):
        self.assertIsNone(detect("No. No.", " No. No.", [("latest", tagged("No."))]))
        self.assertIsNone(detect("Alex moved the decanter away from her as he looked at Upgrade.",
                                 " Alex moved the decanter away from her as he looked at upgrade.",
                                 [("latest", tagged("Alex looked at Upgrade and the decanter."))]))

    def test_an_ordinary_misreading_is_not_an_echo(self):
        self.assertIsNone(detect("Barry muttered.", " Mary muttered.",
                                 [("latest", tagged("Man, that was cathartic. Thank you."))]))
        self.assertIsNone(detect("Barry muttered.", " Mary.", [("latest", tagged("Mary."))]))

    def test_plain_text_drops_speaker_tags_and_controls(self):
        self.assertEqual(plain_text("<|speaker:0|> [laugh] Okay, [pause] I did. <|speaker:0|> Fine."),
                         "Okay, I did. Fine.")


class LadderTests(unittest.TestCase):
    def record(self, applied=None, transcript=" Second words.", passed=False, turns=2):
        metadata = {"history_turns": turns}
        if applied is not None:
            metadata["history_recovery"] = {"version": 1, "requested": applied, "applied": applied}
        return {"report": {"passed": passed, "reasons": ["text_mismatch"], "transcript": transcript},
                "engine_metadata": metadata}

    def test_each_echo_moves_one_rung_past_the_rung_that_echoed(self):
        turns = [("latest", tagged("Second words."))]
        self.assertIsNone(next_mode([], "Sal nodded.", turns))
        self.assertEqual(next_mode([self.record()], "Sal nodded.", turns), "role_tag")
        self.assertEqual(next_mode([self.record(), self.record("role_tag")], "Sal nodded.", turns), "streaming")
        # A rung skipped by the engine counts as the rung it applied.
        self.assertEqual(next_mode([self.record(), self.record("none", turns=0)], "Sal nodded.", turns), "none")
        self.assertEqual(next_mode([self.record(), self.record("role_tag"), self.record("streaming")],
                                   "Sal nodded.", turns), "none")

    def test_an_ordinary_failure_keeps_the_reached_rung_and_never_returns_to_full_history(self):
        turns = [("latest", tagged("Second words."))]
        misread = self.record("role_tag", transcript=" Sal nodder.")
        self.assertEqual(next_mode([self.record(), misread], "Sal nodded.", turns), "role_tag")
        self.assertIsNone(next_mode([self.record(transcript=" Sal nodder.")], "Sal nodded.", turns))

    def test_saved_attempts_without_evidence_are_reassessed_from_their_transcripts(self):
        # Takes recorded before echo detection existed still move the ladder.
        old = self.record()
        self.assertNotIn("history_echo", old["report"])
        self.assertEqual(next_mode([old, old, old], "Sal nodded.", [("latest", tagged("Second words."))]), "role_tag")

    def test_role_tag_leads_the_line_and_merges_with_a_direction(self):
        self.assertEqual(role_tagged(tagged("Sal nodded."), "narration"), tagged("[narration] Sal nodded."))
        self.assertEqual(role_tagged(tagged("[Quiet, uneasy mutter] I know."), "dialogue"),
                         tagged("[dialogue; Quiet, uneasy mutter] I know."))
        self.assertEqual(role_tagged(tagged("Heading"), "heading"), tagged("[narration] Heading"))

    def test_rungs_an_engine_or_request_cannot_use_are_skipped_in_order(self):
        history = [(tagged("Second words."), [[1]])]
        both = {"directing", "streaming"}
        self.assertEqual(apply_recovery(None, tagged("Sal."), history, both, kind="narration"),
                         (None, tagged("Sal."), history, "directing"))
        self.assertEqual(apply_recovery("streaming", tagged("Sal."), history, both, kind="narration")[0::3],
                         ("streaming", "streaming"))
        # No streaming support, a bracketed control, or several voices: no history next.
        self.assertEqual(apply_recovery("streaming", tagged("Sal."), history, {"directing"}, kind=None)[0], "none")
        self.assertEqual(apply_recovery("streaming", tagged("[calm] Sal."), history, both, kind=None)[0], "none")
        mixed = "<|speaker:0|> Sal. <|speaker:1|> Vee."
        self.assertEqual(apply_recovery("role_tag", mixed, history, both, kind=None)[0::2], ("none", []))
        self.assertEqual(apply_recovery("none", tagged("Sal."), history, both, kind=None)[2], [])
        # Without history there is nothing to recover from.
        self.assertIsNone(apply_recovery("none", tagged("Sal."), [], both, kind=None)[0])
        self.assertEqual(LADDER, ("role_tag", "streaming", "none"))


class PublishedChoiceTests(unittest.TestCase):
    def test_an_exhaustion_choice_of_an_echo_is_rejected_and_retired_without_deletion(self):
        record = {"engine_metadata": {"exhaustion_selection": {}},
                  "report": {"passed": True, "expected_text": "Sal nodded.",
                             "original_failed_report": {"passed": False, "transcript": " Second words."}}}
        with heard(tagged("Second words.")):
            evidence = rejected_pick(record)
        self.assertEqual(evidence["heard_request_hash"], "h0")
        ordinary = deepcopy(record)
        ordinary["report"]["original_failed_report"]["transcript"] = " Sal nodder."
        with heard(tagged("Second words.")):
            self.assertIsNone(rejected_pick(ordinary))
        self.assertIsNone(rejected_pick({**record, "engine_metadata": {}}, [("h", tagged("Second words."))]))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_json(folder / "accepted.json", record)
            write_json(folder / "exhaustion-selection.json", {"choice": 1})
            (folder / "clip-repair-parts").mkdir()
            moved = archive(folder, ["accepted.json", "exhaustion-selection.json", "clip-repair.json",
                                     "clip-repair-parts"], evidence)
            self.assertEqual(len(moved), 3)
            self.assertFalse((folder / "accepted.json").exists())
            log = next(folder.glob("history-echo-*.json"))
            self.assertEqual(read_json(log)["retired"], moved)
            self.assertEqual(read_json(folder / moved[0]), record)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ExhaustionCandidateTests(unittest.TestCase):
    def setUp(self):
        from test_renderer_speech_identity import reference, speech_request
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.request = speech_request(reference(), "Sal nodded.")

    def candidate(self, number, transcript):
        wav = self.directory / f"attempt-{number:03d}.wav"
        write_tone(wav, 200 + number * 30)
        write_json(self.directory / f"attempt-{number:03d}.json", {
            "request": deepcopy(self.request), "request_hash": digest(self.request), "attempt": number,
            "audio_file": wav.name, "audio_sha256": file_hash(wav), "engine_metadata": {},
            "report": {"passed": False, "reasons": ["text_mismatch"], "transcript": transcript}})

    def test_echoes_are_never_offered_and_echo_only_candidates_publish_nothing(self):
        from epub_tts_narrator.renderer.exhaustion_selection import select_retained
        from test_renderer_recovery_boundaries import FormatClient, valid_choice
        self.candidate(1, " Second words of the chapter.")
        self.candidate(2, " Sal nodder.")
        client = FormatClient([valid_choice])
        with heard(tagged("Second words of the chapter.")):
            self.assertIsNotNone(candidate_echo(read_json(self.directory / "attempt-001.json"), "Sal nodded."))
            _, record = select_retained(self.directory, self.request, client)
        self.assertEqual([row["attempt"] for row in client.calls[0]["candidates"]], [2])
        self.assertEqual(record["attempt"], 2)

    def test_a_pending_choice_that_offered_an_echo_is_retired_and_asked_again(self):
        from epub_tts_narrator.renderer.exhaustion_selection import select_retained
        from test_renderer_recovery_boundaries import FormatClient, valid_choice
        self.candidate(1, " Second words of the chapter.")
        self.candidate(2, " Sal nodder.")
        client = FormatClient([valid_choice])
        _, earlier = select_retained(self.directory, self.request, client)
        self.assertEqual(earlier["attempt"], 1)
        (self.directory / "accepted.json").unlink()
        with heard(tagged("Second words of the chapter.")):
            _, record = select_retained(self.directory, self.request, client)
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(len(list(self.directory.glob("exhaustion-selection.history-echo-*.json"))), 1)

    def test_only_echoes_left_is_a_clear_failure(self):
        from epub_tts_narrator.renderer.exhaustion_selection import select_retained
        from test_renderer_recovery_boundaries import FormatClient, valid_choice
        self.candidate(1, " Second words of the chapter.")
        with heard(tagged("Second words of the chapter.")), \
                self.assertRaisesRegex(ValueError, "history echo"):
            select_retained(self.directory, self.request, FormatClient([valid_choice]))


class EchoModel(fixtures.Model):
    """Fake Drama 3 that replays its latest history turn for chosen lines."""

    def __init__(self, echo_lines=(), *, role_tag_fixes=True):
        super().__init__()
        self.echo_lines = set(echo_lines)
        self.role_tag_fixes = role_tag_fixes

    def generate(self, text, **options):
        import numpy as np
        index = len(self.calls) + 1
        self.calls.append((text, deepcopy(options)))
        spoken = re.sub(r"<[^>]*>|\[[^]]*\]", "", text).strip()
        previous = options.get("previous_takes") or []
        tag_fixes = self.role_tag_fixes and re.search(r"\[(narration|dialogue)\b", text)
        if previous and spoken in self.echo_lines and not tag_fixes:
            spoken = re.sub(r"<[^>]*>|\[[^]]*\]", "", previous[-1][0]).strip()
        frequency = 200 + 20 * index
        self.transcripts[frequency] = spoken
        audio = (0.15 * np.sin(2 * np.pi * frequency * np.arange(44100) / 44100)).astype(np.float32)
        yield type("Segment", (), {"text": text, "codes": np.full((10, 20), index, dtype=np.int64),
                                   "audio": audio, "finish_reason": "eos"})()


class EchoEngine(Drama3MLXEngine):
    """The local adapter with its streaming layout served by the fake model."""

    streaming_echoes = False

    def _streaming_segment(self, model, text, voice, kept, seed):
        self.streaming_calls.append({"text": text, "kept": [turn for turn, _ in kept], "voice": voice["voice_id"]})
        history = kept if self.streaming_echoes else []
        options = {"previous_takes": history, "streaming": True}
        return next(model.generate(text, **options))


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class LadderWorkflowTests(unittest.TestCase):
    def setUp(self):
        import soundfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = renderer_config(self.root)
        self.engine = EchoEngine(self.root)
        self.engine.streaming_calls = []
        self.engine._identity = {"engine": "drama3_mlx", "model": "fixture", "revision": "one"}
        self.engine._soundfile = soundfile
        reference = self.root / "voice.wav"
        write_tone(reference)
        self.voices = {"v1": {"kind": "reference", "voice_id": "v1", "asset_hash": "a" * 64,
            "reference_audio": str(reference), "reference_sha256": file_hash(reference),
            "reference_text": "A fixed reference."}}
        self.chunks = passages(["First words of the chapter.", "Second words of the chapter.",
                                "Sal nodded.", "Fourth words of the chapter."])
        for chunk in self.chunks:
            chunk["segment"]["delivery"]["direction"] = ""
            chunk["segment"]["kind"] = "narration"

    def run_chapter(self, model):
        self.model = self.engine._model = model
        self.verifier = fixtures.Verifier(model)
        return _render_chapter_chunks({"number": 1, "chunks": self.chunks}, self.voices,
            self.engine, self.engine.identity, self.verifier, self.config, self.root / "work",
            notify=lambda message: None)

    def take_records(self, text):
        records = []
        for path in sorted((self.root / "work" / "clips").glob("*/attempt-*.json")):
            record = read_json(path)
            if record["request"]["chunk"]["spoken_text"] == text:
                records.append(record)
        return sorted(records, key=lambda record: record["attempt"])

    def accepted(self, text):
        for path in (self.root / "work" / "clips").glob("*/accepted.json"):
            record = read_json(path)
            if record["request"]["chunk"]["spoken_text"] == text:
                return record
        raise AssertionError(f"No accepted take for {text!r}")

    def test_role_tag_retry_keeps_full_history_and_becomes_ordinary_history(self):
        self.run_chapter(EchoModel({"Sal nodded."}))
        texts = [text for text, options in self.model.calls]
        self.assertEqual(texts, [tagged("First words of the chapter."), tagged("Second words of the chapter."),
                                 tagged("Sal nodded."), tagged("[narration] Sal nodded."),
                                 tagged("Fourth words of the chapter.")])
        echo, retry = self.model.calls[2][1]["previous_takes"], self.model.calls[3][1]["previous_takes"]
        self.assertEqual([turn for turn, _ in retry], [turn for turn, _ in echo])
        failed = self.take_records("Sal nodded.")[0]
        self.assertEqual(failed["report"]["history_echo"]["heard_share"], 1.0)
        accepted = self.accepted("Sal nodded.")
        self.assertEqual(accepted["engine_metadata"]["history_recovery"],
                         {"version": 1, "requested": "role_tag", "applied": "role_tag"})
        # The tagged take is a real take of the line, so later lines hear it.
        self.assertEqual(self.model.calls[4][1]["previous_takes"][-1][0], tagged("[narration] Sal nodded."))
        self.assertNotIn("history_recovery", self.accepted("Fourth words of the chapter.")["engine_metadata"])

    def test_streaming_layout_retry_hears_the_same_history(self):
        self.run_chapter(EchoModel({"Sal nodded."}, role_tag_fixes=False))
        self.assertEqual(len(self.engine.streaming_calls), 1)
        self.assertEqual(self.engine.streaming_calls[0]["kept"],
                         [tagged("First words of the chapter."), tagged("Second words of the chapter.")])
        accepted = self.accepted("Sal nodded.")
        self.assertEqual(accepted["engine_metadata"]["history_recovery"]["applied"], "streaming")
        self.assertEqual(accepted["engine_metadata"]["history_layout"], "streaming")
        self.assertEqual([record["engine_metadata"].get("history_recovery", {}).get("applied")
                          for record in self.take_records("Sal nodded.")], [None, "role_tag", "streaming"])

    def test_no_history_is_the_last_rung_and_gets_attempts_beyond_the_ordinary_limit(self):
        self.assertEqual(self.config["render"]["max_attempts"], 2)
        self.engine.streaming_echoes = True
        self.run_chapter(EchoModel({"Sal nodded."}, role_tag_fixes=False))
        records = self.take_records("Sal nodded.")
        self.assertEqual([record["engine_metadata"].get("history_recovery", {}).get("applied") for record in records],
                         [None, "role_tag", "streaming", "none"])
        self.assertEqual([bool(record["report"].get("history_echo")) for record in records], [True, True, True, False])
        self.assertEqual(self.model.calls[-2][1]["previous_takes"], [])
        self.assertTrue(self.accepted("Sal nodded.")["report"]["passed"])

    def test_an_ordinary_mismatch_retries_the_same_prompt(self):
        model = EchoModel()
        self.verifier_reject = None
        self.model = self.engine._model = model
        verifier = fixtures.Verifier(model)
        verifier.reject.add(260)
        _render_chapter_chunks({"number": 1, "chunks": self.chunks}, self.voices, self.engine,
            self.engine.identity, verifier, self.config, self.root / "work", notify=lambda message: None)
        records = self.take_records("Sal nodded.")
        self.assertEqual(len(records), 2)
        self.assertNotIn("history_echo", records[0]["report"])
        self.assertNotIn("history_recovery", records[1]["engine_metadata"])
        self.assertEqual(model.calls[2][0], model.calls[3][0])


class StreamingLayoutTests(unittest.TestCase):
    def test_rows_put_the_reference_first_and_end_on_an_open_voice_turn(self):
        import numpy as np
        from types import SimpleNamespace
        from epub_tts_narrator.renderer.drama3_layouts import streaming_rows, streaming_segments

        reference = SimpleNamespace(text="A fixed reference.", codes=np.ones((10, 3), dtype=np.int64))
        history = [(tagged("[calm] First words."), [[2] * 4] * 10), (tagged("Second words."), [[3] * 4] * 10)]
        segments = streaming_segments(tagged("Sal nodded."), reference, history)
        users = [segment for segment in segments[3:] if isinstance(segment, str) and "<|endofturn|>" in segment]
        self.assertEqual(users, ["A fixed reference.<|endofturn|>First words.",
                                 "First words.<|endofturn|>Second words.",
                                 "Second words.<|endofturn|>Sal nodded.", "Sal nodded.<|endofturn|>"])
        self.assertEqual(segments[1], "streaming speech continuation with one-chunk lookahead split by <|endofturn|>")
        self.assertEqual(segments[-1], "<|im_start|>assistant\n<|voice|>")

        class Tokenizer:
            def encode(self, text):
                return SimpleNamespace(ids=[ord(character) % 97 for character in text])

        config = SimpleNamespace(audio_decoder=SimpleNamespace(num_codebooks=10, vocab_size=1024),
                                 semantic_start_token_id=5000)
        try:
            rows = streaming_rows(tagged("Sal nodded."), reference, history, Tokenizer(), config)
        except ModuleNotFoundError:
            self.skipTest("pinned Drama 3 runtime is not installed")
        self.assertEqual(rows.shape[1], 11)
        audio = rows[rows[:, 0] >= 5000]
        self.assertEqual(len(audio), 3 + 4 + 4)
        self.assertTrue((audio[:, 0] == audio[:, 1] + 5000).all())


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class TransportLayoutTests(unittest.TestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.fish_http import FishHTTPEngine
        from epub_tts_narrator.renderer.server import make_server
        from test_fish_http import FakeEngine, wav_bytes

        class StreamingEngine(FakeEngine):
            history_layouts = frozenset({"directing", "streaming"})

            def synthesize_fish(self, text, voices, path, *, seed, history=(), layout="directing"):
                result = super().synthesize_fish(text, voices, path, seed=seed, history=history)
                self.calls[-1]["layout"] = layout
                return {**result, **({"history_layout": layout} if layout != "directing" else {})}

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engines = {"plain": FakeEngine(), "streaming": StreamingEngine()}
        self.clients = {}
        for name, engine in self.engines.items():
            config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "private-test-key",
                                 "max_request_bytes": 104857600, "max_pending_requests": 4},
                      "engine": {"type": "drama3_mlx"}}
            try:
                server = make_server(config, engine, notify=lambda message: None)
            except PermissionError:
                self.skipTest("localhost sockets are disabled by the sandbox")
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            thread.start()

            def stop(server=server, thread=thread):
                server.shutdown()
                thread.join()
                server.server_close()
            self.addCleanup(stop)
            self.clients[name] = FishHTTPEngine({"url": f"http://127.0.0.1:{server.server_port}/v1/tts",
                                                 "name": name, "api_key": "private-test-key"})
        path = self.root / "voice.wav"
        path.write_bytes(wav_bytes())
        self.voice = {"kind": "reference", "voice_id": "v1", "asset_hash": "a" * 64, "speaker": 0,
                      "reference_audio": str(path), "reference_text": "A reference."}

    def test_health_advertises_layouts_and_only_supporting_workers_get_streaming(self):
        for client in self.clients.values():
            client.check_continuity()
        self.assertEqual(self.clients["plain"].history_layouts, frozenset({"directing"}))
        self.assertEqual(self.clients["streaming"].history_layouts, frozenset({"directing", "streaming"}))
        history = [(tagged("Second words."), [[6] * 4] * 10)]
        metadata, _, _ = self.clients["streaming"]._generate_wire(
            tagged("Sal nodded."), [self.voice], history, self.root / "streamed.wav", seed=7, layout="streaming")
        self.assertEqual(self.engines["streaming"].calls[0]["layout"], "streaming")
        self.assertEqual(metadata["history_layout"], "streaming")
        ordinary, _, _ = self.clients["streaming"]._generate_wire(
            tagged("Sal nodded."), [self.voice], history, self.root / "plain.wav", seed=8)
        self.assertEqual(self.engines["streaming"].calls[1]["layout"], "directing")
        self.assertNotIn("history_layout", ordinary)
        with self.assertRaisesRegex(EngineError, "does not build"):
            self.clients["plain"]._generate_wire(tagged("Sal nodded."), [self.voice], history,
                                                 self.root / "refused.wav", seed=9, layout="streaming")
        self.assertEqual(self.engines["plain"].calls, [])

    def test_a_server_refuses_a_layout_its_engine_cannot_build(self):
        client = self.clients["plain"]
        client._history_layouts = frozenset({"directing", "streaming"})  # a stale or forged advertisement
        with self.assertRaisesRegex(EngineError, "400"):
            client._generate_wire(tagged("Sal nodded."), [self.voice], [(tagged("Earlier."), [[1] * 2] * 10)],
                                  self.root / "take.wav", seed=7, layout="streaming")
        self.assertEqual(self.engines["plain"].calls, [])


class ReportTests(unittest.TestCase):
    """The history-echoes command reads saved takes only and retires by renaming."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name) / "render-work"
        self.clips = self.work / "clips"
        self.clips.mkdir(parents=True)
        heard = self.clips / ("a" * 64)
        heard.mkdir()
        write_json(heard / f"{'c' * 64}.codes.json", {"text": tagged("[Quiet] I know it works.")})

    def choice(self, name, transcript, *, chapter="ch0059", exhaustion=True, takes=True):
        directory = self.clips / name
        directory.mkdir()
        continuity_takes = [{"request_hash": "a" * 64, "codes_sha256": "c" * 64}] if takes else []
        metadata = {"exhaustion_selection": {}} if exhaustion else {}
        record = {"request": {"chunk": {"spoken_text": "Sal muttered quietly.",
                                        "segment": {"id": f"{name[:4]}:0:21", "chapter_id": chapter,
                                                    "voice_id": "narrator"}},
                              "continuity": {"version": 3, "takes": continuity_takes}},
                  "engine_metadata": metadata,
                  "report": {"passed": True, "expected_text": "Sal muttered quietly.",
                             "original_failed_report": {"passed": False, "transcript": transcript}}}
        write_json(directory / "accepted.json", record)
        write_json(directory / "exhaustion-selection.json", {"choice": name})
        return directory

    def test_scan_finds_only_echo_choices_and_names_their_chapter(self):
        from epub_tts_narrator.renderer.history_echo_report import scan
        echo = self.choice("e" * 64, " I know it works.")
        self.choice("f" * 64, " Sal muttered quietly, mostly.")
        self.choice("d" * 64, " I know it works.", exhaustion=False)
        self.choice("b" * 64, " I know it works.", takes=False)
        before = {path: path.read_bytes() for path in self.clips.rglob("*.json")}
        report = scan(self.work, {"ch0059": 58})
        self.assertEqual(report["best_effort_choices"], 3)
        self.assertEqual([item["take_directory"] for item in report["echo_choices"]], [echo.name])
        finding = report["echo_choices"][0]
        self.assertEqual((finding["chapter_number"], finding["echoed_line"]), (58, "I know it works."))
        self.assertEqual({path: path.read_bytes() for path in self.clips.rglob("*.json")}, before)

    def test_retire_renames_echo_choices_under_the_render_lock(self):
        from epub_tts_narrator.renderer.history_echo_report import retire, scan
        from epub_tts_narrator.renderer.workflow import RenderBlocked
        echo = self.choice("e" * 64, " I know it works.")
        kept = self.choice("f" * 64, " Sal muttered quietly, mostly.")
        (self.work / ".render.lock").mkdir()
        with self.assertRaisesRegex(RenderBlocked, "may be running"):
            retire(self.work, {"ch0059": 58})
        self.assertTrue((echo / "accepted.json").exists())
        (self.work / ".render.lock").rmdir()
        report = retire(self.work, {"ch0059": 58})
        self.assertTrue(report["retired"])
        self.assertEqual(len(report["echo_choices"][0]["retired_files"]), 2)
        self.assertFalse((echo / "accepted.json").exists())
        self.assertEqual(len(list(echo.glob("accepted.history-echo-*.json"))), 1)
        self.assertTrue((kept / "accepted.json").exists())
        self.assertFalse((self.work / ".render.lock").exists())
        self.assertEqual(scan(self.work, {})["echo_choices"], [])


if __name__ == "__main__":
    unittest.main()
