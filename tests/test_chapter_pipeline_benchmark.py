"""Offline behavior tests for the chapter pipeline benchmark corpus freeze.

Covers deterministic equal-work selection, four balanced groups, frozen
canonical word counts, synthetic benchmark identity, private 0600 corpus
files, checksum/input-hash integrity, and fail-closed path safety.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_chapter_pipeline import main
from scripts.chapter_benchmark_corpus import (
    GROUP_COUNT, corpus_body, length_class, load_corpus, pack_voices, select_cases,
)
from scripts.chapter_benchmark_report import finalize, new_report
from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from test_renderer_workflow import FFMPEG, FFPROBE

FFMPEG_PATH = FFMPEG or "ffmpeg"
FFPROBE_PATH = FFPROBE or "ffprobe"


def plan_chunk(text, voice_id, index):
    row = {"id": f"s{index}", "block_id": f"b{index}", "chapter_id": "chapter-x",
           "start": 0, "end": len(text), "text": text, "spoken_text": text,
           "voice_id": voice_id, "role_id": "narrator", "delivery": {"direction": ""},
           "attribution": {"basis": "explicit"}, "kind": "narration",
           "performance": {}, "break_after": "paragraph"}
    return {"chunk_id": digest({"row": row})[:32], "segment": row, "spoken_text": text,
            "spoken_start": 0, "spoken_end": len(text),
            "source_ranges": [{"start": 0, "end": len(text)}], "silent_control": False}


def fixture_plan(words_per_chapter=((8, 24, 48, 8), (8, 24, 48, 8), (8, 24, 48, 8), (8, 24, 48, 8))):
    chapters, index = [], 0
    for number, words in enumerate(words_per_chapter, 1):
        chunks = [plan_chunk(" ".join(f"w{n}" for n in range(count)), "vn", index + position)
                  for position, count in enumerate(words)]
        index += len(words)
        chapters.append({"number": number, "id": f"chapter-{number:04d}", "title": f"Chapter {number}",
                         "chunks": chunks})
    return {"chapters": chapters, "voice_ids": ["vn"], "errors": []}


def fixture_voices():
    return {"vn": {"asset_hash": digest("vn-voice"), "reference_audio": "unused.wav",
                   "reference_text": "A saved reference."}}


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_group_balanced(self):
        plan = fixture_plan()
        before = deepcopy(plan)
        first = select_cases(plan["chapters"], fixture_voices(), [1, 2, 3, 4], total=8, seed=7)
        second = select_cases(plan["chapters"], fixture_voices(), [1, 2, 3, 4], total=8, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(plan, before)
        cases, warmup = first
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case["id"] for case in cases}), 8)
        self.assertEqual(len({case["chunk"]["chunk_id"] for case in cases}), 8)
        self.assertEqual([sum(case["group"] == group for case in cases) for group in range(GROUP_COUNT)],
                         [2, 2, 2, 2])
        for group in range(GROUP_COUNT):
            origins = {case["origin"] for case in cases if case["group"] == group}
            self.assertGreaterEqual(len(origins), 2, "groups must mix source chapters")

    def test_stratification_keeps_length_classes_balanced_across_groups(self):
        cases, _ = select_cases(fixture_plan()["chapters"], fixture_voices(), [1, 2, 3, 4], total=12, seed=3)
        self.assertEqual(length_class(" ".join(["w"] * 48)), "long")
        for group in range(GROUP_COUNT):
            members = [case for case in cases if case["group"] == group]
            classes = sorted(case["length_class"] for case in members)
            self.assertEqual(len(members), 3)
            self.assertEqual(classes, ["long", "medium", "short"])

    def test_insufficient_unique_candidates_fail_instead_of_repeating(self):
        plan = fixture_plan(((8, 24), (8, 24), (8, 24), (8, 24)))
        with self.assertRaisesRegex(ValueError, "long"):
            select_cases(plan["chapters"], fixture_voices(), [1, 2, 3, 4], total=12, seed=1)

    def test_duplicate_chapters_and_missing_chapters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "four"):
            select_cases(fixture_plan()["chapters"], fixture_voices(), [1, 1, 2, 3], total=8, seed=1)
        with self.assertRaisesRegex(ValueError, "outside"):
            select_cases(fixture_plan()["chapters"], fixture_voices(), [1, 2, 3, 9], total=8, seed=1)

    def test_warmup_cases_come_from_the_same_voices_but_never_the_groups(self):
        cases, warmup = select_cases(fixture_plan()["chapters"], fixture_voices(), [1, 2, 3, 4], total=8, seed=5)
        self.assertTrue(warmup)
        self.assertEqual(warmup[0]["voice_asset"], cases[0]["voice_asset"])
        group_chunks = {case["chunk"]["chunk_id"] for case in cases}
        self.assertFalse(group_chunks & {case["chunk"]["chunk_id"] for case in warmup})

    def test_canonical_word_counts_are_frozen_once_per_source_span(self):
        cases, _ = select_cases(fixture_plan()["chapters"], fixture_voices(), [1, 2, 3, 4], total=4, seed=2)
        for case in cases:
            self.assertEqual(case["words"], len(case["chunk"]["spoken_text"].split()))
        self.assertGreaterEqual(sum(case["words"] for case in cases), 4)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = (Path(self.temp.name) / "corpus").resolve()

    def freeze(self, total=8):
        bundle = {"manifest": {"source_fingerprint": "a" * 64, "provenance": {}},
                  "source": {"book": {"id": "book1", "title": "Fixture Book", "authors": ["A"]}},
                  "bundle_hash": digest("bundle")}
        config = {"render": {"max_attempts": 2, "paragraph_synthesis": False, "normalize_loudness": False,
                             "ffmpeg": FFMPEG_PATH, "ffprobe": FFPROBE_PATH}}
        cases, warmup = select_cases(fixture_plan()["chapters"], fixture_voices(), [1, 2, 3, 4], total=total, seed=11)
        reference = self.output / "refs" / "x.flac"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(b"fixture lossless reference bytes")
        packed = {voice_id: {"asset_hash": record["asset_hash"], "reference_audio": "refs/x.flac",
                             "reference_sha256": file_hash(reference),
                             "reference_text": record["reference_text"]}
                  for voice_id, record in fixture_voices().items()}
        return corpus_body(bundle, config, cases, warmup, packed, seed=11, sample="smoke")

    def test_body_binds_synthetic_identity_and_complete_input_hash(self):
        body = self.freeze()
        self.assertEqual(body["format"], "epub-tts-chapter-pipeline-benchmark")
        self.assertNotIn("Fixture Book", body["book"]["title"])
        self.assertNotEqual(body["book"]["id"], "book1")
        self.assertEqual(body["sample"], "smoke")
        self.assertEqual(body["total_source_words"],
                         sum(case["words"] for case in body["cases"]))
        self.assertEqual(body["requested_cases"], 8)
        again = self.freeze()
        self.assertEqual(body["input_hash"], again["input_hash"])
        changed = self.freeze(total=4)
        self.assertNotEqual(body["input_hash"], changed["input_hash"])

    @unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required for lossless FLAC packing")
    def test_pack_writes_portable_private_flac_references(self):
        from test_renderer_workflow import write_tone
        wav = Path(self.temp.name) / "reference.wav"
        write_tone(wav)
        voices = {"vn": {"asset_hash": digest("vn"), "reference_audio": str(wav),
                         "reference_text": "Reference words."}}
        self.output.mkdir(parents=True)
        packed = pack_voices(voices, self.output, FFMPEG_PATH)
        self.output.chmod(0o700)
        for path in self.output.rglob("*"):
            path.chmod(0o600 if path.is_file() else 0o700)
        record = packed["vn"]
        self.assertEqual(record["reference_text"], "Reference words.")
        self.assertFalse(Path(record["reference_audio"]).is_absolute())
        audio = self.output / record["reference_audio"]
        self.assertTrue(str(audio).startswith(str(self.output)))
        with audio.open("rb") as stream:
            self.assertEqual(stream.read(4), b"fLaC")
        self.assertTrue(all(os.stat(path).st_mode & 0o077 == 0 for path in self.output.rglob("*")))

    def test_load_verifies_checksums_input_hash_and_traversal_safety(self):
        body = self.freeze()
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "corpus.json"

        def signed(payload):
            write_json(path, dict(payload, corpus_sha256=digest({k: v for k, v in payload.items()})))

        signed(body)
        corpus, voices = load_corpus(path)
        self.assertEqual(voices["vn"]["reference_audio"], str(self.output / "refs" / "x.flac"))
        tampered = deepcopy(body)
        tampered["cases"][0]["id"] = "f" * 24
        signed(tampered)
        with self.assertRaisesRegex(ValueError, "input hash"):
            load_corpus(path)
        escaped = deepcopy(body)
        escaped["voices"]["vn"]["reference_audio"] = "../outside.flac"
        signed(escaped)
        with self.assertRaisesRegex(ValueError, "portable relative paths"):
            load_corpus(path)
        write_json(path, dict(escaped, corpus_sha256=digest("wrong")))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_corpus(path)


class ReportEligibilityTests(unittest.TestCase):
    def eligibility_report(self, *, resumed, fake_engine):
        from shared_fixtures import renderer_config
        corpus = {"corpus_sha256": "c" * 64, "input_hash": "i" * 64, "sample": None,
                  "total_source_words": 12, "cases": [], "warmup": []}
        capabilities = {"engine": {"engine": "fish_tts_pool"}, "renderer_fingerprint": "f" * 64,
                        "parallel_chapters": 2}
        return new_report(corpus=corpus, corpus_path="corpus.json", config=renderer_config(),
                          capabilities=capabilities, scheduler="legacy", output=Path("."),
                          resumed=resumed, warm_words=1, fake_engine=fake_engine)

    def test_fresh_real_is_eligible_and_fake_or_resumed_never_are(self):
        fresh = self.eligibility_report(resumed=False, fake_engine=False)
        self.assertTrue(fresh["speed_gate_eligible"])
        self.assertEqual(fresh["eligibility_reason"], "fresh")
        self.assertFalse(fresh["fake_engine"])
        self.assertEqual(fresh["engine_policy"], "configured_models")
        resumed = self.eligibility_report(resumed=True, fake_engine=False)
        self.assertFalse(resumed["speed_gate_eligible"])
        self.assertEqual(resumed["eligibility_reason"], "resumed_or_cached")
        fake = self.eligibility_report(resumed=False, fake_engine=True)
        self.assertTrue(fake["fake_engine"])
        self.assertFalse(fake["speed_gate_eligible"])
        self.assertEqual(fake["eligibility_reason"], "fake_engine")
        self.assertEqual(fake["engine_policy"], "fake_engine")
        # A fake run stays ineligible even when it also resumed.
        both = self.eligibility_report(resumed=True, fake_engine=True)
        self.assertEqual(both["eligibility_reason"], "fake_engine")
        self.assertFalse(both["speed_gate_eligible"])

    def test_only_complete_eligible_runs_compute_throughput(self):
        manifest = {"groups": [{"state": "complete"} for _ in range(4)]}
        fake = self.eligibility_report(resumed=False, fake_engine=True)
        fake["timing"]["active_pipeline_seconds"] = 2.0
        finalize(fake, manifest, status="complete")
        self.assertIsNone(fake["accepted_words_per_active_second"])
        real = self.eligibility_report(resumed=False, fake_engine=False)
        real["timing"]["active_pipeline_seconds"] = 2.0
        finalize(real, manifest, status="complete")
        self.assertGreater(real["accepted_words_per_active_second"], 0.0)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_main(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                main(list(args))
            finally:
                self._last_stderr = stderr.getvalue()
        return stdout.getvalue(), stderr.getvalue()

    def test_case_counts_must_split_across_exactly_four_groups(self):
        with self.assertRaisesRegex(SystemExit, "") as raised:
            self.run_main("prepare", "bundle", "--chapters", "1,2,3,4",
                          "--cases", "7", "--sample", "odd", "--output", str(self.root / "out"))
        self.assertNotEqual(raised.exception.code, 0)

    def test_small_samples_require_an_explicit_exploratory_name(self):
        with self.assertRaisesRegex(SystemExit, "") as raised:
            _, stderr = self.run_main("prepare", "bundle", "--chapters", "1,2,3,4",
                                      "--cases", "12", "--output", str(self.root / "out"))
        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("--sample", self._last_stderr)

    def test_bad_scheduler_choice_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_main("run", "corpus.json", "--scheduler", "turbo", "--output", str(self.root / "out"))


if __name__ == "__main__":
    unittest.main()
