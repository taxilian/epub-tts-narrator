"""Export numbering cannot change audio identity or bypass speech checks."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, write_json
from epub_tts_narrator.renderer.incremental_identity import (
    accepted_requests, completed_chunks, reanchor_export_sequences,
)
from test_renderer_paragraphs import passages, plan
from epub_tts_narrator.renderer.paragraphs import group_paragraphs


class ExportSequenceIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.chunks = passages(["Come inside. ", "The rain is getting worse."])
        for sequence, chunk in enumerate(self.chunks):
            chunk["segment"]["sequence"] = sequence

    def request(self, chunk):
        return {"kind": "speech", "chunk": deepcopy(chunk), "engine": {"engine": "fixture"},
                "voice_asset_hash": "original-voice", "base_seed": 1, "sample_rate": 44100}

    def save_request(self, request):
        key = digest(request)
        path = self.work / "clips" / key / "accepted.json"
        write_json(path, {"request": request, "request_hash": key, "report": {"passed": True}})
        return path

    def shifted(self):
        result = plan(deepcopy(self.chunks))
        for chunk in result["chapters"][0]["chunks"]:
            chunk["segment"]["sequence"] += 2
        return result

    def test_only_sequence_changes_reuse_the_original_request_identity(self):
        requests = [self.request(chunk) for chunk in self.chunks]
        for request in requests:
            self.save_request(request)
        current = self.shifted()
        untouched = deepcopy(current)
        result = reanchor_export_sequences(current, accepted_requests(self.work))
        self.assertEqual(result, plan(self.chunks))
        self.assertEqual(current, untouched)
        for chunk, original in zip(result["chapters"][0]["chunks"], requests):
            self.assertEqual(digest(self.request(chunk)), digest(original))

    def test_voice_source_direction_and_offset_changes_do_not_reuse_metadata(self):
        request = self.request(self.chunks[0])
        for field, value in (("voice_id", "different"), ("role_id", "other"),
                             ("text", "Other words"), ("start", 5),
                             ("delivery", {"direction": "shouting", "evidence": []})):
            with self.subTest(field=field):
                current = self.shifted()
                current["chapters"][0]["chunks"][0]["segment"][field] = value
                self.assertEqual(reanchor_export_sequences(current, [request]), current)

    def test_paragraph_turns_preserve_original_sequence_and_group_identity(self):
        original = group_paragraphs(plan(self.chunks))
        grouped = original["chapters"][0]["chunks"][0]
        request = {"kind": "paragraph_speech", "chunk": grouped}
        reanchored = reanchor_export_sequences(self.shifted(), [request])
        self.assertEqual(group_paragraphs(reanchored), original)

    def test_a_completed_chapter_anchor_wins_over_other_historical_numbering(self):
        alternate = deepcopy(self.chunks[0])
        alternate["segment"]["sequence"] = 91
        result = reanchor_export_sequences(self.shifted(), [self.request(alternate)],
                                           completed=self.chunks)
        self.assertEqual(result, plan(self.chunks))

    def test_changed_cached_request_cannot_supply_a_sequence_anchor(self):
        path = self.save_request(self.request(self.chunks[0]))
        request = self.request(self.chunks[0])
        request["chunk"]["segment"]["sequence"] = 98
        write_json(path, {"request": request, "request_hash": path.parent.name})
        with self.assertRaisesRegex(ValueError, "Invalid accepted synthesis request"):
            accepted_requests(self.work)

    def test_completed_silent_chunk_can_keep_its_sequence_without_accepted_audio(self):
        silent = deepcopy(self.chunks[0])
        silent.update(silent_control=True, spoken_text="")
        identity = {"clips": [dict(silent, audio_sha256=None)], "context": {"chapter_hash": "prior"}}
        cache_key = digest(identity)
        cached = {"format": "audiobook-renderer-chapter", "version": 1, "identity": identity}
        write_json(self.work / "chapters" / cache_key / "chapter.json",
                   dict(cached, record_sha256=digest(cached)))
        checkpoint = {"format": "audiobook-chapter-checkpoint", "version": 1,
                      "cache_key": cache_key, "context": identity["context"]}
        write_json(self.work / "completed-chapters" / "chapter-1.json",
                   dict(checkpoint, record_sha256=digest(checkpoint)))
        current = plan([deepcopy(silent)])
        current["chapters"][0]["chunks"][0]["segment"]["sequence"] = 4
        restored = reanchor_export_sequences(
            current, [], completed=completed_chunks(self.work, current["chapters"]))
        self.assertEqual(restored, plan([silent]))


if __name__ == "__main__":
    unittest.main()
