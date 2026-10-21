"""Failed paragraph decisions survive interruption without weakening verification."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.paragraph_recovery import (
    load_paragraph_fallbacks, record_paragraph_fallback,
)
from epub_tts_narrator.renderer.paragraphs import group_paragraphs


def request_fixture():
    members, cursor = [], 0
    for number, text in enumerate(["Private dialogue. ", "She answered. ", "More private dialogue."]):
        row = {"id": f"b1:{cursor}", "block_id": "b1", "chapter_id": "ch1",
               "start": cursor, "end": cursor + len(text), "text": text,
               "voice_id": f"voice-{number % 2}", "role_id": f"role-{number % 2}",
               "break_after": "paragraph" if number == 2 else "none"}
        members.append({"chunk_id": digest(row)[:32], "segment": row,
                        "spoken_text": text, "spoken_start": 0, "spoken_end": len(text),
                        "source_ranges": [{"start": cursor, "end": cursor + len(text)}],
                        "silent_control": False})
        cursor += len(text)
    plan = {"chapters": [{"id": "ch1", "chunks": members}]}
    grouped = group_paragraphs(plan)["chapters"][0]["chunks"][0]
    return {"kind": "paragraph_speech", "version": 1, "chunk": grouped,
            "engine": {"private_key": "must-never-be-copied"}}, plan


class ParagraphRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name) / "work"
        self.request, self.plan = request_fixture()

    def test_missing_directory_is_empty_and_does_not_create_files(self):
        self.assertEqual(load_paragraph_fallbacks(self.work), set())
        self.assertFalse(self.work.exists())

    def test_marker_preserves_original_ids_and_replans_without_regrouping(self):
        path = record_paragraph_fallback(self.work, self.request)
        expected = {turn["chunk_id"] for turn in self.request["chunk"]["paragraph_turns"]}
        self.assertEqual(load_paragraph_fallbacks(self.work), expected)
        self.assertEqual(group_paragraphs(self.plan, preserved_chunk_ids=expected), self.plan)
        record = read_json(path)
        self.assertEqual(path.name, digest(self.request) + ".json")
        self.assertEqual(record["request_hash"], digest(self.request))
        self.assertNotIn("Private dialogue", path.read_text())
        self.assertNotIn("must-never-be-copied", path.read_text())
        self.assertNotIn("voice-", path.read_text())

    def test_repeated_record_is_idempotent_and_immutable(self):
        path = record_paragraph_fallback(self.work, self.request)
        original = path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino
        self.assertEqual(record_paragraph_fallback(self.work, deepcopy(self.request)), path)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino), original)

    def test_changed_synthesis_request_has_separate_audit_marker(self):
        first = record_paragraph_fallback(self.work, self.request)
        changed = deepcopy(self.request)
        changed["base_seed"] = 22
        second = record_paragraph_fallback(self.work, changed)
        self.assertNotEqual(first, second)
        self.assertEqual(len(load_paragraph_fallbacks(self.work)), 3)

    def test_changed_source_ids_are_not_preserved(self):
        record_paragraph_fallback(self.work, self.request)
        changed = deepcopy(self.plan)
        for member in changed["chapters"][0]["chunks"]:
            member["chunk_id"] = digest({"changed": member["chunk_id"]})[:32]
        grouped = group_paragraphs(changed, preserved_chunk_ids=load_paragraph_fallbacks(self.work))
        self.assertEqual(len(grouped["chapters"][0]["chunks"]), 1)

    def test_tampered_marker_is_not_replaced(self):
        path = record_paragraph_fallback(self.work, self.request)
        record = read_json(path)
        record["request_hash"] = "0" * 64
        write_json(path, record)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            load_paragraph_fallbacks(self.work)
        with self.assertRaises(ValueError):
            record_paragraph_fallback(self.work, self.request)
        self.assertEqual(path.read_bytes(), before)

    def test_digest_detects_content_changes(self):
        path = record_paragraph_fallback(self.work, self.request)
        record = read_json(path)
        record["record_sha256"] = "0" * 64
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_paragraph_fallbacks(self.work)

    def test_wrong_filename_is_rejected(self):
        path = record_paragraph_fallback(self.work, self.request)
        path.rename(path.with_name("0" * 64 + ".json"))
        with self.assertRaises(ValueError):
            load_paragraph_fallbacks(self.work)

    def test_malformed_marker_shapes_are_rejected(self):
        path = record_paragraph_fallback(self.work, self.request)
        valid = read_json(path)
        malformed = [[], {}, {**valid, "version": True}, {**valid, "extra": "unexpected"},
                     {**valid, "member_chunk_ids": ["bad", "also-bad"]},
                     {**valid, "member_chunk_ids": valid["member_chunk_ids"][:1]},
                     {**valid, "member_chunk_ids": [valid["member_chunk_ids"][0]] * 2},
                     {**valid, "paragraph_chunk_id": "0" * 32}]
        for record in malformed:
            with self.subTest(record=record):
                write_json(path, record)
                with self.assertRaises(ValueError):
                    load_paragraph_fallbacks(self.work)

    def test_non_json_marker_is_rejected(self):
        path = record_paragraph_fallback(self.work, self.request)
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_paragraph_fallbacks(self.work)

    def test_symlink_marker_is_rejected_for_read_and_write(self):
        path = record_paragraph_fallback(self.work, self.request)
        target = self.work / "outside.json"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            load_paragraph_fallbacks(self.work)
        with self.assertRaises(ValueError):
            record_paragraph_fallback(self.work, self.request)

    def test_symlink_directory_and_work_are_rejected(self):
        self.work.mkdir()
        target = self.work / "target"
        target.mkdir()
        (self.work / "paragraph-fallbacks").symlink_to(target, target_is_directory=True)
        for work in [self.work, self.work / "paragraph-fallbacks"]:
            with self.subTest(work=work):
                with self.assertRaises(ValueError):
                    load_paragraph_fallbacks(work)
                with self.assertRaises(ValueError):
                    record_paragraph_fallback(work, self.request)

    def test_non_directory_state_is_rejected(self):
        self.work.mkdir()
        (self.work / "paragraph-fallbacks").write_text("not a directory")
        with self.assertRaises(ValueError):
            load_paragraph_fallbacks(self.work)

    def test_invalid_requests_are_rejected_before_writing(self):
        variants = [None, {}, {**self.request, "kind": "speech"}]
        for field, value in [("paragraph_turns", []), ("spoken_text", "rewritten"),
                             ("chunk_id", "0" * 32), ("silent_control", True)]:
            changed = deepcopy(self.request)
            changed["chunk"][field] = value
            variants.append(changed)
        for field, value in [("spoken_text", ""), ("chunk_id", "../bad"),
                             ("paragraph_turns", []), ("silent_control", True)]:
            changed = deepcopy(self.request)
            changed["chunk"]["paragraph_turns"][0][field] = value
            variants.append(changed)
        for request in variants:
            with self.subTest(request=request):
                with self.assertRaises(ValueError):
                    record_paragraph_fallback(self.work, request)
                self.assertFalse(self.work.exists())

    def test_omitted_reordered_and_cross_block_text_are_rejected(self):
        for mutation in ("omitted", "reordered", "cross-block", "row-text"):
            changed = deepcopy(self.request)
            chunk = changed["chunk"]
            if mutation == "omitted":
                chunk["paragraph_turns"][0]["spoken_text"] = "Private "
            elif mutation == "reordered":
                chunk["paragraph_turns"].reverse()
            elif mutation == "cross-block":
                chunk["paragraph_turns"][0]["segment"]["block_id"] = "another-block"
            else:
                chunk["segment"]["text"] = "changed"
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    record_paragraph_fallback(self.work, changed)


if __name__ == "__main__":
    unittest.main()
