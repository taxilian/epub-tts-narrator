from copy import deepcopy
from pathlib import Path
import unittest

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.speech_cache import SpeechCache
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.spoken_text import effective_delivery
import test_renderer_split_delivery_recovery as fixtures


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe required")
class RecoveryProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SplitDeliveryRecoveryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        fixture.fail_directed = False
        segment = fixture.request["chunk"]["segment"]
        source = fixtures.TEXT.replace("Arrgh", "Arrrrgh")
        segment.update(text=source, start=0, end=len(source), spoken_text=fixtures.TEXT)
        segment["speech_normalization"] = {"version": 1, "edits": [{
            "start": 0, "end": 7, "original": "Arrrrgh", "replacement": "Arrgh",
            "direction": "Keep the strained cry", "reason": "Repeated-letter spelling"}]}
        fixture.direction = effective_delivery(segment)
        self.identity = generation_identity(fixture.request, fixture.voices)
        fixture.request["synthesis_identity"] = self.identity
        fixture.parent = fixture.work / "clips" / digest(fixture.request)
        self.cache = SpeechCache(fixture.work, {"voice": fixture.voice})

    def legacy_join(self):
        wav, accepted = self.fixture.recover()
        legacy = deepcopy(accepted)
        audit = legacy["engine_metadata"]["recovery"]
        audit.pop("child_acceptance_version", None)
        for row in audit["parts"]:
            row.pop("acceptance", None)
        write_json(wav.parent / "accepted.json", legacy)
        return wav, legacy

    def test_normalized_directed_join_retains_all_child_delivery_evidence(self):
        wav, accepted = self.fixture.recover()

        self.assertEqual(generation_identity(accepted["request"], self.fixture.voices,
                                             engine_metadata=accepted["engine_metadata"]), self.identity)
        self.cache.remember(self.identity, accepted)
        self.assertEqual(self.cache.find(self.identity), (wav, accepted))
        for row in accepted["engine_metadata"]["recovery"]["parts"]:
            self.assertNotIn("synthesis_request", row["acceptance"]["request"])

    def test_legacy_join_publishes_and_reloads_without_mutating_verified_files(self):
        wav, accepted = self.legacy_join()
        files = {path: path.read_bytes() for path in wav.parent.rglob("*") if path.is_file()}
        before = len(self.fixture.calls)

        self.cache.remember(self.identity, accepted)
        reloaded = SpeechCache(self.fixture.work, {"voice": self.fixture.voice})

        self.assertEqual(reloaded.find(self.identity), (wav, accepted))
        self.assertTrue(reloaded.matches(self.fixture.request, accepted))
        self.assertEqual(len(self.fixture.calls), before)
        for path, contents in files.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_legacy_join_rejects_changed_acceptance_audio_and_unsafe_paths(self):
        wav, accepted = self.legacy_join()
        original = deepcopy(accepted)
        row = accepted["engine_metadata"]["recovery"]["parts"][0]
        audio = wav.parent / row["audio_file"]
        child = audio.parent / "accepted.json"
        saved_audio, saved_child = audio.read_bytes(), child.read_bytes()
        for change in ("acceptance", "audio", "path", "symlink", "delivery"):
            with self.subTest(change=change):
                altered = deepcopy(original)
                if change == "acceptance":
                    value = read_json(child)
                    value["report"]["expected_text"] = "Missing original words."
                    write_json(child, value)
                elif change == "audio":
                    audio.write_bytes(b"changed audio")
                elif change == "path":
                    altered["engine_metadata"]["recovery"]["parts"][0]["audio_file"] = "../outside.wav"
                elif change == "symlink":
                    audio.unlink()
                    audio.symlink_to(wav)
                else:
                    value = read_json(child)
                    value["engine_metadata"].update(synthesis_text=row["text"].strip(),
                                                     direction="A different performance")
                    write_json(child, value)
                    altered["engine_metadata"]["recovery"]["parts"][0]["accepted_sha256"] = file_hash(child)
                write_json(wav.parent / "accepted.json", altered)
                try:
                    with self.assertRaises((ValueError, OSError)):
                        self.cache.remember(self.identity, altered)
                    self.assertFalse(list((self.fixture.work / "speech-cache").glob("*.json")))
                finally:
                    if audio.is_symlink():
                        audio.unlink()
                    audio.write_bytes(saved_audio)
                    child.write_bytes(saved_child)

    def test_versioned_audit_is_never_reconstructed_over_missing_evidence(self):
        _, accepted = self.fixture.recover()
        metadata = deepcopy(accepted["engine_metadata"])
        metadata["recovery"]["child_acceptance_version"] = 1
        metadata["recovery"]["parts"][0].pop("acceptance", None)

        with self.assertRaises(ValueError):
            self.cache.identity(accepted["request"], engine_metadata=metadata)

    def test_known_engine_join_requires_recorded_synthesis_text(self):
        wav, accepted = self.fixture.recover()
        metadata = deepcopy(accepted["engine_metadata"])
        row = metadata["recovery"]["parts"][0]
        del row["acceptance"]["engine_metadata"]["synthesis_text"]
        child = (wav.parent / row["audio_file"]).parent / "accepted.json"
        write_json(child, row["acceptance"])
        row["accepted_sha256"] = file_hash(child)

        with self.assertRaises(ValueError):
            generation_identity(accepted["request"], self.fixture.voices, engine_metadata=metadata)


if __name__ == "__main__":
    unittest.main()
