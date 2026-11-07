"""Descriptive audio names retain ownership without becoming paths."""

from pathlib import Path
import unittest

from epub_tts_narrator.renderer.audio_naming import (
    descriptive_audio_filename,
    speech_take_filename,
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
        self.assertEqual(normalized, "chunk-42-voice-mara-attempt-003.wav")
        self.assertEqual(raw, "chunk-42-voice-mara-attempt-003-raw.wav")
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
        self.assertEqual(created, "voice-hero-Hero-attempt-001.wav")
        self.assertEqual(recovered, "aaaaaaaaaaaaaaaa-part-012-034-attempt-002.wav")

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


if __name__ == "__main__":
    unittest.main()
