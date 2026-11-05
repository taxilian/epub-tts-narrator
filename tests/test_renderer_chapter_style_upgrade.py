"""Old chapters whose freeform directions were ignored can refresh selectively."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import engines, speech_identity, workflow
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.workflow import RenderBlocked
import test_renderer_incremental as incremental
from test_renderer_workflow import FixtureEngine, FixtureVerifier, FFMPEG, FFPROBE


class DirectedFixtureEngine(FixtureEngine):
    """Use the real preparation contract around deterministic offline audio."""

    identity = {
        "engine": "drama3_mlx", "model": "fishaudio/drama-3",
        "runtime": "Sogni-AI/drama3-mlx", "adapter_version": 1,
        "installed_version": "0.1.1", "installed_revision": "old-runtime",
        "checkpoint_files_sha256": {"model.safetensors": "a" * 64},
    }

    def synthesize(self, text, voice, path, *, seed, direction=""):
        metadata = super().synthesize(text, voice, path, seed=seed, direction=direction)
        prepared, delivery = engines._directed_text(text, direction)
        # Older production metadata did not consistently include this key.
        if delivery.get("applied_control") is None:
            delivery.pop("applied_control", None)
        return dict(metadata, synthesis_text=prepared, **delivery)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class ChapterStyleUpgradeTests(unittest.TestCase):
    # Reuse the two-chapter, two-voice fixture without inheriting its test cases.
    setUp = incremental.IncrementalRendererTests.setUp
    save_revision = incremental.IncrementalRendererTests.save_revision
    run_render = incremental.IncrementalRendererTests.run_render
    package = incremental.IncrementalRendererTests.package
    checkpoint = incremental.IncrementalRendererTests.checkpoint
    output_bytes = incremental.IncrementalRendererTests.output_bytes
    model_counts = incremental.IncrementalRendererTests.model_counts

    def legacy_book(self):
        self.document["segments"][1]["delivery"] = {
            "direction": "tremulous and shaken",
            "evidence": [{"block_id": "b2", "quote": "Let us leave"}],
        }
        self.save_revision()
        self.engine = DirectedFixtureEngine({})
        self.engine.identity = deepcopy(self.engine.identity)
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        directed_text = engines._directed_text
        verified_take = workflow.verified_take

        def old_directed_text(text, direction):
            prepared, metadata = directed_text(text, direction)
            if metadata["applied_control"] == "style_prefix":
                prepared = text
                metadata.update(direction_applied=False, applied_control=None)
            return prepared, metadata

        def old_verified_take(text, generate, directory, request, verifier, config, **kwargs):
            # These chapters predate the extra synthesis-input binding. Keep
            # their real old request shape, hash, accepted file, and index.
            if "synthesis_identity" in request:
                request = dict(request)
                request.pop("synthesis_identity")
                directory = directory.parent / digest(request)
            return verified_take(text, generate, directory, request, verifier, config, **kwargs)

        with patch.object(engines, "_directed_text", side_effect=old_directed_text), \
             patch.object(speech_identity, "_directed_text", side_effect=old_directed_text), \
             patch.object(workflow, "verified_take", side_effect=old_verified_take):
            self.run_render()
        accepted = {path: path.read_bytes()
                    for path in (self.work / "clips").glob("*/accepted.json")}
        self.assertTrue(accepted)
        for path in accepted:
            self.assertNotIn("synthesis_identity", read_json(path)["request"])
        return accepted

    def assert_selective_refresh(self, *, changed_runtime=False):
        old_accepted = self.legacy_book()
        counts = self.model_counts()
        first_checkpoint = self.checkpoint(1).read_bytes()
        other_checkpoint = self.checkpoint(2).read_bytes()
        if changed_runtime:
            self.engine.identity.update(adapter_version=2, installed_revision="new-runtime")

        result = self.run_render()

        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.model_counts(), (counts[0], counts[1] + 1, counts[2] + 1))
        self.assertEqual(self.engine.speech_calls[-1]["text"], self.document["segments"][1]["text"])
        self.assertEqual(self.engine.speech_calls[-1]["direction"], "tremulous and shaken")
        self.assertNotEqual(self.checkpoint(1).read_bytes(), first_checkpoint)
        self.assertEqual(self.checkpoint(2).read_bytes(), other_checkpoint)
        new_records = [read_json(path) for path in (self.work / "clips").glob("*/accepted.json")
                       if path not in old_accepted]
        self.assertEqual(len(new_records), 1)
        self.assertEqual(new_records[0]["engine_metadata"]["synthesis_text"],
                         "[tremulous and shaken] " + self.document["segments"][1]["text"])
        for path, contents in old_accepted.items():
            self.assertEqual(path.read_bytes(), contents)

        # The replacement checkpoint must be complete, rather than scheduling
        # the same upgrade again at every invocation.
        published = self.output_bytes()
        counts = self.model_counts()
        with patch.object(workflow, "verified_take", side_effect=AssertionError("No TTS/ASR")), \
             patch.object(workflow, "prepare_chapter", side_effect=AssertionError("No chapter encoding")), \
             patch.object(workflow, "package_audio", side_effect=AssertionError("No book encoding")):
            self.assertEqual(self.run_render(), result)
        self.assertEqual(self.model_counts(), counts)
        self.assertEqual(self.output_bytes(), published)

    def test_legacy_exact_request_hash_does_not_hide_changed_style_input(self):
        self.assert_selective_refresh()

    def test_compatible_runtime_change_refreshes_only_previously_ignored_style(self):
        self.assert_selective_refresh(changed_runtime=True)

    def test_corrupt_accepted_audio_still_blocks_instead_of_refreshing_it(self):
        old_accepted = self.legacy_book()
        styled = next(path for path in old_accepted
                      if read_json(path)["request"]["chunk"]["segment"]["delivery"]["direction"])
        record = read_json(styled)
        audio = styled.parent / record["audio_file"]
        audio.write_bytes(audio.read_bytes() + b"changed")
        changed_checksum = file_hash(audio)
        counts, published = self.model_counts(), self.output_bytes()

        with self.assertRaisesRegex(RenderBlocked, "[Cc]ached.*changed|[Aa]udio.*changed"):
            self.run_render()

        self.assertEqual(self.model_counts(), counts)
        self.assertEqual(file_hash(audio), changed_checksum)
        self.assertEqual(self.output_bytes(), published)

    def test_assemble_identifies_stale_style_and_preserves_published_book(self):
        self.legacy_book()
        counts, published = self.model_counts(), self.output_bytes()

        with patch.object(workflow, "verified_take", side_effect=AssertionError("No models in assemble")), \
             self.assertRaisesRegex(RenderBlocked, "speech inputs changed.*render"):
            self.package()

        self.assertEqual(self.model_counts(), counts)
        self.assertEqual(self.output_bytes(), published)

    def test_stale_style_cannot_hide_later_invalid_verification(self):
        self.legacy_book()
        checkpoint = read_json(self.checkpoint(1))
        checkpoint.pop("record_sha256")
        # The styled narration precedes the final dialogue in this chapter.
        # Keep outer integrity valid to exercise the per-clip verification gate.
        checkpoint["acceptance"][-1]["report"]["passed"] = False
        write_json(self.checkpoint(1), dict(checkpoint, record_sha256=digest(checkpoint)))
        counts, published = self.model_counts(), self.output_bytes()

        with self.assertRaisesRegex(RenderBlocked, "Invalid saved chapter.*verification"):
            self.run_render()

        self.assertEqual(self.model_counts(), counts)
        self.assertEqual(self.output_bytes(), published)


if __name__ == "__main__":
    unittest.main()
