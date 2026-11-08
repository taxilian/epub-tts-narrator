"""Descriptive audio names retain ownership without becoming paths."""

from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.audio_naming import (
    descriptive_audio_filename,
    speech_take_filename,
    temporary_audio_prefix,
    validate_audio_filename,
)


class AudioFilenameTests(unittest.TestCase):
    def test_new_name_identifies_source_owner_and_purpose(self):
        # Given source and owner metadata containing path syntax and Unicode.
        # When deriving an actual artifact basename.
        filename = descriptive_audio_filename(
            "reference", "voice/../Mara", "明の本", suffix=".wav")

        # Then the file stays in its destination and names all available roles.
        self.assertEqual(filename, "voice Mara-明の本-reference.wav")
        self.assertEqual(Path(filename).name, filename)
        self.assertNotEqual(filename, "reference.wav")

    def test_new_name_bounds_utf8_and_keeps_its_purpose(self):
        # Given context longer than a portable component and a multibyte suffix role.
        # When deriving the bounded name.
        filename = descriptive_audio_filename("recognition-crop", "界" * 200, suffix=".wav")

        # Then truncation cannot split Unicode or discard the artifact purpose.
        self.assertLessEqual(len(filename.encode("utf-8")), 240)
        self.assertTrue(filename.endswith("-recognition-crop.wav"))
        self.assertEqual(validate_audio_filename(filename, suffix=".wav"), filename)

    def test_unusable_context_still_produces_a_stable_purpose_bound_id(self):
        # Given no readable source identifier.
        # When deriving the name twice from the same opaque metadata.
        first = descriptive_audio_filename("reference", "///", suffix=".flac")
        second = descriptive_audio_filename("reference", "///", suffix=".flac")

        # Then the result is stable and descriptive rather than generic or hash-only.
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{16}-reference\.flac$")

    def test_take_name_uses_chunk_voice_and_attempt_role_without_book_text(self):
        # Given a bound speech request and a sentence that must stay out of paths.
        request = {"kind": "speech", "chunk": {"chunk_id": "chunk-42", "spoken_text": "Secret passage",
                                               "segment": {"voice_id": "voice-mara"}}}

        # When deriving normalized and raw names for one attempt.
        normalized = speech_take_filename(request, 3)
        raw = speech_take_filename(request, 3, role="raw")

        # Then stable content ownership and roles replace the generic basename.
        self.assertEqual(normalized, "attempt-003-chunk-42-voice-mara-take.wav")
        self.assertEqual(raw, "attempt-003-chunk-42-voice-mara-raw.wav")
        self.assertNotIn("Secret", normalized + raw)

    def test_voice_and_recovery_requests_use_their_available_owner_ids(self):
        # Given voice creation and recovery-only requests with different metadata.
        creation = {"kind": "create_voice", "voice_id": "voice-hero", "profile": {"name": "Hero"}}
        recovery = {"kind": "speech_recovery_part", "parent_request_hash": "a" * 64,
                    "part": {"start": 12, "end": 34}}

        # When deriving their new attempt names.
        created = speech_take_filename(creation, 1)
        recovered = speech_take_filename(recovery, 2)

        # Then each uses real available identifiers rather than guessed book text.
        self.assertEqual(created, "attempt-001-voice-hero-Hero-take.wav")
        self.assertEqual(recovered, "attempt-002-aaaaaaaaaaaaaaaa-part-012-034-take.wav")

    def test_explicit_legacy_recorded_name_remains_valid(self):
        # Given an already persisted generic artifact name.
        # When validating it for an existing record.
        filename = validate_audio_filename("reference.wav", suffix=".wav")

        # Then compatibility does not rewrite the established path.
        self.assertEqual(filename, "reference.wav")

    def test_malformed_recorded_names_are_refused_before_file_operations(self):
        # Given persisted paths that could escape, hide, or confuse the destination.
        invalid = ("../outside.wav", "/tmp/outside.wav", "..\\outside.wav", ".hidden.wav",
                   "bad\x00name.wav", "CON.wav", "trailing .wav", "sample.flac", "", None, 42)
        for value in invalid:
            with self.subTest(value=value):
                # When resolving the recorded name, then validation fails closed.
                with self.assertRaisesRegex(ValueError, "Recorded audio filename"):
                    validate_audio_filename(value, suffix=".wav")


class TemporaryStagingPrefixTests(unittest.TestCase):
    def test_prefix_names_its_destination_and_stage_and_fits_the_filesystem(self):
        # Given a real chapter destination and the download stage that stages it.
        # When deriving the hidden staging prefix and creating an actual temp file.
        prefix = temporary_audio_prefix("chapter-0042.flac", "download", suffix=".flac")
        with tempfile.TemporaryDirectory() as temporary:
            with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".flac", dir=temporary,
                                             delete=False) as handle:
                staged = Path(handle.name)

            # Then the staged name is hidden, keeps destination and stage, and
            # reserves room for the random suffix without exceeding the limit.
            name = staged.name
            self.assertTrue(name.startswith("."))
            self.assertIn("chapter-0042", name)
            self.assertIn("download", name)
            self.assertNotIn(".flac-", name)
            self.assertLessEqual(len(name.encode("utf-8")), 255)
        self.assertFalse(staged.exists())

    def test_long_unicode_destination_still_creates_within_the_filesystem_limit(self):
        # Given a destination whose own name is longer than any safe prefix budget.
        # When staging through an actual NamedTemporaryFile with the derived prefix.
        prefix = temporary_audio_prefix("界" * 200 + "-take.wav", "drama3-cuda", suffix=".wav")
        with tempfile.TemporaryDirectory() as temporary:
            with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", dir=temporary,
                                             delete=False) as handle:
                staged = Path(handle.name)
                handle.write(b"audio")

            # Then truncation never splits Unicode or drops the stage identity.
            self.assertLessEqual(len(staged.name.encode("utf-8")), 255)
            self.assertIn("drama3-cuda", staged.name)
            self.assertEqual(staged.read_bytes(), b"audio")
        self.assertFalse(staged.exists())

    def test_destination_path_and_control_syntax_never_reach_the_staged_name(self):
        # Given destination metadata containing separators and a control character.
        # When deriving the prefix, then only the sanitized final component remains.
        prefix = temporary_audio_prefix(Path("voices") / "../Mara\x07.wav", "qwen3-design",
                                        suffix=".wav")
        self.assertNotIn("/", prefix)
        self.assertNotIn("\x07", prefix)
        self.assertIn("Mara", prefix)
        self.assertTrue(prefix.startswith(".Mara"))

    def test_unusable_destinations_still_receive_their_stage_name(self):
        # Given an opaque destination that sanitizes away to nothing.
        # When deriving the prefix, then a stable bounded stem carries the stage.
        first = temporary_audio_prefix("///", "copy")
        second = temporary_audio_prefix("///", "copy")
        self.assertEqual(first, second)
        self.assertIn("copy", first)
        self.assertTrue(first.startswith("."))

    def test_stage_must_describe_itself(self):
        # Given a stage label with no visible characters.
        # When deriving the prefix, then it is refused instead of staging anonymously.
        with self.assertRaisesRegex(ValueError, "stage"):
            temporary_audio_prefix("chapter.wav", "//")


if __name__ == "__main__":
    unittest.main()
