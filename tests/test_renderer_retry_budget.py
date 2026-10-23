"""Retry allowances belong to a run; saved speech take IDs never rewind."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.retry_budget import (
    SpeechRetryBudget, generation_attempts, whole_strategy_allowance,
)


class SpeechRetryBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_same_request_across_replans_cannot_reset_allowance(self):
        budget = SpeechRetryBudget()
        self.assertEqual(budget.started(self.directory, "request"), 0)
        self.assertEqual(budget.claim(self.directory, "request", 3), 1)
        self.assertEqual(budget.claim(self.directory / ".", "request", 3), 2)
        self.assertEqual(budget.claim(self.directory / "unused" / "..", "request", 3), 3)
        self.assertIsNone(budget.claim(self.directory, "request", 3))
        self.assertEqual(budget.started(self.directory, "request"), 3)

    def test_new_run_has_fresh_budget_without_affecting_old_run(self):
        old, new = SpeechRetryBudget(), SpeechRetryBudget()
        self.assertEqual(old.claim(self.directory, "request", 1), 1)
        self.assertIsNone(old.claim(self.directory, "request", 1))
        self.assertEqual(new.claim(self.directory, "request", 1), 1)
        self.assertIsNone(old.claim(self.directory, "request", 1))

    def test_paragraph_retry_policy_preserves_local_fallback_unless_enabled(self):
        self.assertFalse(SpeechRetryBudget().retry_paragraphs)
        self.assertTrue(SpeechRetryBudget(retry_paragraphs=True).retry_paragraphs)
        for value in (1, "false", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SpeechRetryBudget(retry_paragraphs=value)

    def test_allowances_are_scoped_by_both_directory_and_request(self):
        budget = SpeechRetryBudget()
        for folder, request in ((self.directory, "one"), (self.directory, "two"),
                                (self.directory / "second", "one")):
            self.assertEqual(budget.claim(folder, request, 1), 1)
        self.assertIsNone(budget.claim(self.directory, "one", 1))

    def test_concurrent_duplicates_share_a_bounded_allowance(self):
        budget = SpeechRetryBudget()
        with ThreadPoolExecutor(max_workers=16) as executor:
            claims = list(executor.map(lambda _: budget.claim(self.directory, "request", 3), range(100)))
        self.assertEqual(sorted(value for value in claims if value is not None), [1, 2, 3])
        self.assertEqual(budget.started(self.directory, "request"), 3)

    def test_invalid_limits_do_not_claim_a_generation(self):
        budget = SpeechRetryBudget()
        for limit in (0, -1, True, 3.0, "3", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                budget.claim(self.directory, "request", limit)
        self.assertEqual(budget.started(self.directory, "request"), 0)

    def test_whole_strategy_reserves_a_real_shorter_part_opportunity(self):
        self.assertEqual(whole_strategy_allowance(12, 0), 12)
        self.assertEqual(whole_strategy_allowance(12, 2), 6)
        self.assertEqual(whole_strategy_allowance(12, 4), 6)
        self.assertEqual(whole_strategy_allowance(3, 2), 1)
        self.assertEqual(whole_strategy_allowance(3, 4), 1)

    def test_child_cannot_spend_the_remaining_siblings_reserved_starts(self):
        budget = SpeechRetryBudget()
        for number in range(1, 7):
            self.assertEqual(budget.claim(self.directory, "whole", 6,
                                          episode="parent", episode_maximum=12), number)
        for number in range(1, 6):
            self.assertEqual(budget.claim(self.directory / "first", "first", 12,
                                          episode="parent", episode_maximum=12,
                                          episode_reserve=1), number)
        self.assertIsNone(budget.claim(self.directory / "first", "first", 12,
                                       episode="parent", episode_maximum=12, episode_reserve=1))
        self.assertEqual(budget.episode_started("parent"), 11)
        self.assertEqual(budget.claim(self.directory / "second", "second", 12,
                                      episode="parent", episode_maximum=12), 1)
        self.assertEqual(budget.episode_started("parent"), 12)
        self.assertIsNone(budget.claim(self.directory / "second", "second", 12,
                                       episode="parent", episode_maximum=12))

    def test_inventory_includes_saved_orphan_and_high_ids_once_in_numeric_order(self):
        names = ["attempt-003.json", "attempt-003.wav", "attempt-002-raw.wav",
                 "attempt-012-recovery-raw.wav", "attempt-1000.wav",
                 "attempt-999-regenerated-0123456789abcdef.wav",
                 "attempt-1000000000.json"]
        for name in names:
            (self.directory / name).write_bytes(b"retained candidate")
        self.assertEqual(generation_attempts(self.directory), [2, 3, 12, 999, 1000, 1000000000])
        self.assertEqual({path.name for path in self.directory.iterdir()}, set(names))

    def test_inventory_ignores_unknown_noncanonical_and_reference_metadata_names(self):
        names = ["attempt-000.wav", "attempt-0001.wav", "attempt-01.wav", "attempt--01.wav",
                 "attempt-001-generation.json", "attempt-001-regenerated-nothex.wav",
                 "accepted.json", "unrelated.wav", "attempt-001.wav.backup", "attempt-١٠٠.wav"]
        for name in names:
            (self.directory / name).write_bytes(b"leave untouched")
        self.assertEqual(generation_attempts(self.directory), [])
        self.assertEqual({path.name for path in self.directory.iterdir()}, set(names))

    def test_inventory_rejects_recognized_symlinks_and_directories(self):
        target = self.directory / "target"
        target.write_bytes(b"do not read")
        path = self.directory / "attempt-012.wav"
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular, nonsymlink"):
            generation_attempts(self.directory)
        self.assertEqual(target.read_bytes(), b"do not read")
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "regular, nonsymlink"):
            generation_attempts(self.directory)

    def test_inventory_refuses_symlink_root_and_handles_missing_directory(self):
        missing = self.directory / "missing"
        self.assertEqual(generation_attempts(missing), [])
        self.assertFalse(missing.exists())
        alias = self.directory / "alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            generation_attempts(alias)
        missing.write_bytes(b"not a directory")
        with self.assertRaisesRegex(ValueError, "directory"):
            generation_attempts(missing)
