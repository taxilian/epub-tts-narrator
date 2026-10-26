import copy
import hashlib
import struct
import unittest

from epub_tts_narrator.renderer.censorship import CensorshipAlignmentRequired, apply_masks, matches, plan_masks, policy_identity, validate_mask_plan, validate_policy


class CensorshipTests(unittest.TestCase):
    def setUp(self):
        self.policy = validate_policy({"enabled": True})
        self.audio = hashlib.sha256(b"original accepted audio").hexdigest()

    def report(self, text, *, start=0.1, step=0.3, duration=0.2):
        return {"word_timing_status": "available", "words": [{"word": word, "start": start + index * step, "end": start + index * step + duration, "probability": 0.99} for index, word in enumerate(text.split())]}

    def plan(self, text, report=None, **kwargs):
        return plan_masks(text, report if report is not None else self.report(text), self.policy, sample_rate=1000, frames=10000, audio_sha256=self.audio, **kwargs)

    def validate(self, audit, text, **kwargs):
        return validate_mask_plan(audit, text, self.policy, sample_rate=1000, frames=10000, audio_sha256=self.audio, **kwargs)

    def test_disabled_is_absent_from_identity_and_needs_no_timing(self):
        self.assertIsNone(policy_identity())
        self.assertEqual(matches("shit", {}), [])
        self.assertEqual(plan_masks("shit", None, {}, sample_rate=1000, frames=10, audio_sha256=self.audio)["masks"], [])

    def test_default_matches_whole_words_case_and_explicit_compounds_only(self):
        selected = matches("FUCK! Shit, mother-fucker, bullshit; Scunthorpe shitake and shittingly.", self.policy)
        self.assertEqual([item["normalized"] for item in selected], ["fuck", "shit", "motherfucker", "bullshit"])
        self.assertEqual(matches("shit", {"enabled": True, "words": ["heck"]}), [])
        self.assertEqual([item["normalized"] for item in matches("fucker’s", self.policy)], ["fuckers"])

    def test_policy_identity_is_stable_and_round_trips(self):
        a = policy_identity({"enabled": True, "words": ["SHIT", "fuck", "shit"]})
        b = policy_identity({"enabled": True, "words": ["fuck", "shit"]})
        self.assertEqual(a, b)
        self.assertEqual(policy_identity(a), a)

    def test_policy_rejects_malformed_or_excessive_controls(self):
        for policy in ({"enabled": 1}, {"words": "shit"}, {"words": ["holy shit"]}, {"words": ["*"]}, {"preserve_ms": -1}, {"preserve_ms": True}, {"fade_ms": 21}, {"padding_ms": float("nan")}, {"max_preserved_fraction": 0.6}, {"max_preserved_fraction": float("inf")}, {"version": True}, {"not_a_setting": 1}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_policy(policy)

    def test_no_target_does_not_require_an_asr_report(self):
        audit = self.plan("Nothing objectionable here.", report={})
        self.assertEqual(audit["masks"], [])
        self.validate(audit, "Nothing objectionable here.")

    def test_exact_transcript_maps_repeated_words_by_real_timing(self):
        text = "Shit happens, then shit happens again."
        audit = self.plan(text)
        self.assertEqual(len(audit["masks"]), 2)
        self.assertEqual([item["word_start_frame"] for item in audit["masks"]], [100, 1000])
        self.assertTrue(all(item["method"] == "asr_word_timestamps:exact_transcript" for item in audit["masks"]))
        self.validate(audit, text)

    def test_split_asr_compound_is_a_single_source_word_mask(self):
        audit = self.plan("That motherfucker ran away", self.report("That mother fucker ran away"))
        mask = audit["masks"][0]
        self.assertEqual(mask["target"], "motherfucker")
        self.assertEqual(mask["word_start_frame"], 400)
        self.assertEqual(mask["word_end_frame"], 900)
        self.validate(audit, "That motherfucker ran away")

    def test_joined_asr_word_cannot_guess_internal_target_boundary(self):
        with self.assertRaises(CensorshipAlignmentRequired):
            self.plan("Holy shit that hurt", self.report("Holyshit that hurt"))

    def test_nonexact_transcript_uses_unique_context_not_shifted_word_index(self):
        text = "Jasper suddenly said this shit is cold today"
        report = self.report("Jess said this shit is cold today")
        audit = self.plan(text, report)
        self.assertEqual(audit["masks"][0]["word_start_frame"], 1000)
        self.assertEqual(audit["masks"][0]["method"], "asr_word_timestamps:unique_context")
        self.validate(audit, text)

    def test_repeated_ambiguous_phrases_require_alignment(self):
        with self.assertRaises(CensorshipAlignmentRequired):
            self.plan("well shit it is well shit it is", self.report("well shit it is"))

    def test_target_only_in_unrelated_transcript_is_not_sufficient(self):
        with self.assertRaises(CensorshipAlignmentRequired):
            self.plan("What the shit happened here", self.report("A totally different shit sentence"))

    def test_missing_substituted_or_low_confidence_target_requires_alignment(self):
        text = "What the shit happened here"
        for report in (self.report("What the happened here"), self.report("What the ship happened here"), {"words": [], "word_timing_status": "unavailable"}):
            with self.subTest(report=report), self.assertRaises(CensorshipAlignmentRequired):
                self.plan(text, report)
        report = self.report(text)
        report["words"][2]["probability"] = 0.1
        with self.assertRaises(CensorshipAlignmentRequired):
            self.plan(text, report)

    def test_overlapping_zero_nonfinite_or_outside_timestamps_require_alignment(self):
        for field, value in (("start", -1), ("start", 0.4), ("end", 0.0), ("end", float("nan")), ("end", 11), ("probability", True)):
            report = self.report("shit happens")
            report["words"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(CensorshipAlignmentRequired):
                self.plan("shit happens", report)
        report = self.report("shit happens")
        report["words"][1]["start"] = 0.2
        with self.assertRaises(CensorshipAlignmentRequired):
            self.plan("shit happens", report)

    def test_explicit_source_proven_spelling_targets_are_bound_without_fuzzy_matching(self):
        text = "What the fahk happened here"
        targets = [{"start": 9, "end": 13, "text": "fahk", "normalized": "fuck"}]
        self.assertEqual(matches(text, self.policy), [])
        audit = self.plan(text, targets=targets)
        self.assertEqual(audit["masks"][0]["normalized"], "fuck")
        self.validate(audit, text, targets=targets)
        with self.assertRaises(ValueError):
            self.validate(audit, text)

    def test_explicit_target_cannot_match_inside_an_innocent_word(self):
        for target in ({"start": 0, "end": 4, "text": "shit", "normalized": "shit"}, {"start": 0, "end": 7, "text": "shitake", "normalized": "not-in-policy"}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.plan("shitake", targets=[target])

    def test_explicit_spelling_target_can_span_multiple_known_aligned_tokens(self):
        text = "The mother fah ker left"
        targets = [{"start": 4, "end": 18, "text": "mother fah ker", "normalized": "motherfucker"}]
        report = self.report(text)
        report["alignment_method"] = "mlx_whisper_attention_dtw"
        audit = self.plan(text, report, targets=targets)
        mask = audit["masks"][0]
        self.assertEqual(mask["word_start_frame"], 400)
        self.assertEqual(mask["word_end_frame"], 1200)
        self.assertEqual(mask["method"], "mlx_whisper_attention_dtw:exact_transcript")
        self.validate(audit, text, targets=targets)

    def test_pcm_keeps_short_onset_then_mutes_tail_without_changing_neighbors_or_duration(self):
        text = "Oh shit no"
        audit = self.plan(text)
        mask = audit["masks"][0]
        raw = struct.pack("<" + "h" * 10000, *([12000] * 10000))
        modified = apply_masks(raw, 0, audit["masks"])
        result = struct.unpack("<" + "h" * 10000, modified)
        self.assertEqual(len(modified), len(raw))
        self.assertEqual(result[:mask["mute_start_frame"]], (12000,) * mask["mute_start_frame"])
        self.assertEqual(result[mask["silence_start_frame"]:mask["word_end_frame"]], (0,) * (mask["word_end_frame"] - mask["silence_start_frame"]))
        self.assertEqual(result[mask["mute_end_frame"]:], (12000,) * (10000 - mask["mute_end_frame"]))
        self.assertLessEqual(mask["preserved_frames"], 45)
        self.assertLessEqual(mask["preserved_frames"], (mask["word_end_frame"] - mask["word_start_frame"]) * 0.2)
        fading = result[mask["mute_start_frame"]:mask["silence_start_frame"]]
        self.assertEqual(list(fading), sorted(fading, reverse=True))
        self.assertTrue(all(sample == 0 for sample in result[mask["silence_start_frame"]:mask["word_end_frame"]]))

    def test_pcm_streaming_and_stereo_match_whole_buffer_application(self):
        audit = self.plan("Oh shit no")
        raw = struct.pack("<" + "h" * 2000, *([12000, -8000] * 1000))
        whole = apply_masks(raw, 0, audit["masks"], channels=2)
        pieces = [apply_masks(raw[start * 4:min(1000, start + 71) * 4], start, audit["masks"], channels=2) for start in range(0, 1000, 71)]
        self.assertEqual(b"".join(pieces), whole)
        self.assertEqual(len(whole), len(raw))

    def test_too_short_target_is_fully_muted_instead_of_preserved(self):
        audit = self.plan("shit", self.report("shit", duration=0.02))
        mask = audit["masks"][0]
        self.assertEqual(mask["preserved_frames"], 0)
        self.assertEqual(mask["mute_start_frame"], mask["word_start_frame"])
        self.validate(audit, "shit")

    def test_target_shorter_than_alignment_resolution_requires_realignment(self):
        for duration in (0.008, 0.019, 0.0199):
            with self.subTest(duration=duration), self.assertRaises(CensorshipAlignmentRequired):
                self.plan("shit", self.report("shit", duration=duration))

    def test_padding_and_fade_do_not_change_next_spoken_word(self):
        report = self.report("shit now")
        report["words"][1]["start"] = report["words"][0]["end"]
        audit = self.plan("shit now", report)
        mask = audit["masks"][0]
        self.assertEqual(mask["fade_in_frames"], 0)
        self.assertEqual(mask["mute_end_frame"], mask["word_end_frame"])
        self.validate(audit, "shit now")

    def test_adjacent_targets_share_a_frame_boundary_without_overlap_or_gap(self):
        report = self.report("shit fuck", start=0.10025, step=0.20025, duration=0.20025)
        audit = self.plan("shit fuck", report)
        first, second = audit["masks"]
        self.assertEqual(first["mute_end_frame"], second["word_start_frame"])
        self.validate(audit, "shit fuck")

    def test_whole_word_mode_and_no_fades_are_supported(self):
        policy = dict(self.policy, preserve_ms=0, fade_ms=0, padding_ms=0)
        audit = plan_masks("shit", self.report("shit"), policy, sample_rate=1000, frames=10000, audio_sha256=self.audio)
        mask = audit["masks"][0]
        self.assertEqual(mask["word_start_frame"], mask["mute_start_frame"])
        self.assertEqual(mask["word_end_frame"], mask["mute_end_frame"])
        validate_mask_plan(audit, "shit", policy, sample_rate=1000, frames=10000, audio_sha256=self.audio)

    def test_persisted_plan_rejects_changed_audio_text_policy_targets_or_intervals(self):
        text, audit = "Oh shit no", self.plan("Oh shit no")
        for field, value in (("audio_sha256", "0" * 64), ("text_sha256", "0" * 64), ("status", "pending"), ("sample_rate", 24000), ("targets", [])):
            changed = copy.deepcopy(audit)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(changed, text)
        changed = copy.deepcopy(audit)
        changed["masks"][0]["mute_end_frame"] += 1
        with self.assertRaises(ValueError):
            self.validate(changed, text)
        with self.assertRaises(ValueError):
            validate_mask_plan(audit, text, dict(self.policy, preserve_ms=40), sample_rate=1000, frames=10000, audio_sha256=self.audio)

    def test_invalid_pcm_arguments_are_rejected(self):
        for pcm, offset, channels in ((b"x", 0, 1), (b"xx", -1, 1), (b"xx", 0, 0), (b"xx", 0, 2), (bytearray(b"xx"), 0, 1)):
            with self.subTest(pcm=pcm, offset=offset, channels=channels), self.assertRaises(ValueError):
                apply_masks(pcm, offset, [], channels=channels)


if __name__ == "__main__":
    unittest.main()
