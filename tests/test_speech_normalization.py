"""Analysis normalizes written performance without rewriting source content."""

from copy import deepcopy
from threading import Lock
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from epub_tts_narrator.speech_normalization import (
    REVIEW, SCAN, _validate_scan, normalization_pending, normalize_document_speech,
)
from epub_tts_narrator.spoken_text import build_spoken, spoken_source_ranges


def document(*texts):
    blocks = [{"id": f"b{number}", "chapter_id": "ch1", "text": text}
              for number, text in enumerate(texts, 1)]
    return {
        "source": {"book": {"id": "book", "title": "Invented examples"}, "blocks": blocks},
        "cast": {"roles": [{"id": "speaker", "kind": "dialogue", "description": "A character speaks.",
                             "voice_id": "voice", "character_id": "c1"}],
                 "characters": [{"name": "Vern", "aliases": []}]},
        "segments": [{"id": block["id"] + ":0:" + str(len(block["text"])), "block_id": block["id"],
                      "start": 0, "end": len(block["text"]), "text": block["text"], "kind": "dialogue",
                      "role_id": "speaker", "voice_id": "voice", "delivery": {"direction": "", "evidence": []}}
                     for block in blocks], "provenance": {},
    }


def edit(text, original, replacement, direction="Sustain the vowels and sibilants naturally."):
    start = text.index(original)
    return {"start": start, "end": start + len(original), "original": original,
            "replacement": replacement, "direction": direction,
            "reason": "Written pronunciation conveys the same spoken word."}


class Requests:
    def __init__(self, changes=None, verdicts=None):
        self.changes = changes or {}
        self.verdicts = verdicts or {}
        self.calls, self.active, self.maximum = [], 0, 0
        self.lock = Lock()

    def run(self, system, payload, validate, *, cancelled=None):
        with self.lock:
            self.calls.append((system, deepcopy(payload)))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            time.sleep(.002)
            if payload["mode"] == "speech_spelling_scan":
                response = {"segments": []}
                for row in payload["target_segments"]:
                    edits = [edit(row["text"], old, new) for old, new in self.changes.items() if old in row["text"]]
                    edits.sort(key=lambda item: item["start"])
                    response["segments"].append({"segment_id": row["segment_id"], "edits": edits})
            else:
                response = {"votes": [{"edit_id": row["edit_id"],
                                       "verdict": self.verdicts.get((payload["reviewer_id"], row["original"]), "approve"),
                                       "reason": "The spelling represents the same word in context."}
                                      for row in payload["proposals"]]}
            return validate(response)
        finally:
            with self.lock:
                self.active -= 1


class SpeechNormalizationTests(unittest.TestCase):
    def test_real_hissing_failure_becomes_three_correct_words_and_retains_source(self):
        source = document("wheerrre issss sheeee")
        before = deepcopy(source)
        requests = Requests({"wheerrre": "where", "issss": "is", "sheeee": "she"})
        result = normalize_document_speech(source, requests)
        row = result["segments"][0]
        self.assertEqual(row["spoken_text"], "where is she")
        self.assertEqual(source, before)
        self.assertEqual(result["source"], before["source"])
        self.assertEqual(result["cast"], before["cast"])
        self.assertEqual(row["text"], before["segments"][0]["text"])
        self.assertEqual(row["delivery"], before["segments"][0]["delivery"])
        self.assertEqual(spoken_source_ranges(row, 0, 5), [{"start": 0, "end": 8}])
        self.assertEqual(len(requests.calls), 4)
        self.assertTrue(all("votes" not in payload for _, payload in requests.calls))
        self.assertEqual(len({payload["reviewer_id"] for _, payload in requests.calls if "reviewer_id" in payload}), 3)
        self.assertEqual(result["provenance"]["speech_normalization"]["applied_edits"], 3)
        self.assertFalse(normalization_pending(result))

    def test_arbitrary_dialect_and_drawn_out_cry_preserve_grammar(self):
        source = document("wull, i denno, Vern... Ahh reckon ahh ain't got no ideas.", "Nooooooo!!!!!")
        changes = {"wull": "Well", "denno": "don't know", "Ahh": "I", "ahh": "I", "Nooooooo!!!!!": "No!"}
        result = normalize_document_speech(source, Requests(changes))
        self.assertEqual(result["segments"][0]["spoken_text"], "Well, i don't know, Vern... I reckon I ain't got no ideas.")
        self.assertEqual(result["segments"][1]["spoken_text"], "No!")
        self.assertIn("ain’t got no", SCAN)
        self.assertIn("same", REVIEW.lower())

    def test_hissing_prefixes_and_brackets_compose(self):
        source = document("[sssso sssssaid]")
        source["segments"][0]["performance"] = {"kind": "bracketed"}
        source["segments"][0]["spoken_text"] = "sssso sssssaid"
        result = normalize_document_speech(source, Requests({"sssso": "so", "sssssaid": "said"}))
        self.assertEqual(result["segments"][0]["spoken_text"], "so said")
        self.assertEqual(build_spoken(result["segments"][0])[0], "so said")

    def test_every_spoken_segment_scanned_including_unusual_spelling_without_repeats(self):
        source = document("Ordinary speech.", "wull", "!!!", "Another ordinary line.")
        requests = Requests({"wull": "well"})
        normalize_document_speech(source, requests)
        rows = [row for _, payload in requests.calls if payload["mode"] == "speech_spelling_scan"
                for row in payload["target_segments"]]
        self.assertEqual([row["text"] for row in rows], ["Ordinary speech.", "wull", "Another ordinary line."])

    def test_each_edit_requires_two_independent_approvals(self):
        source = document("wull Ahh")
        requests = Requests({"wull": "well", "Ahh": "I"}, {
            ("speech_spelling_reviewer_1", "Ahh"): "reject",
            ("speech_spelling_reviewer_2", "Ahh"): "uncertain",
            ("speech_spelling_reviewer_1", "wull"): "reject",
        })
        result = normalize_document_speech(source, requests)
        self.assertEqual(result["segments"][0]["spoken_text"], "well Ahh")
        marker = result["provenance"]["speech_normalization"]
        self.assertEqual((marker["applied_edits"], marker["rejected_edits"]), (1, 1))
        self.assertFalse(marker["decisions"][1]["applied"])
        self.assertEqual(len(marker["decisions"][1]["votes"]), 3)

    def test_completed_stage_noop_and_changed_context_role_or_delivery_rescans(self):
        result = normalize_document_speech(document("wull"), Requests({"wull": "well"}))
        requests = Requests()
        self.assertEqual(normalize_document_speech(result, requests), result)
        self.assertEqual(requests.calls, [])
        for mutate in (lambda d: d["cast"]["roles"][0].update(description="Another role."),
                       lambda d: d["segments"][0]["delivery"].update(direction="whisper"),
                       lambda d: d["source"]["book"].update(title="Different context")):
            changed = deepcopy(result)
            mutate(changed)
            self.assertTrue(normalization_pending(changed))
        incomplete = deepcopy(result)
        incomplete["segments"][0].pop("speech_normalization")
        self.assertTrue(normalization_pending(incomplete))
        linked = deepcopy(result)
        linked["cast"]["roles"][0].update(id="series_role", voice_id="series_voice", character_id="series_char")
        linked["segments"][0].update(role_id="series_role", voice_id="series_voice")
        self.assertFalse(normalization_pending(linked))

    def test_no_changes_still_marks_complete_without_jury_calls(self):
        source = document("Perfectly ordinary words.")
        requests = Requests()
        result = normalize_document_speech(source, requests)
        self.assertEqual(len(requests.calls), 1)
        self.assertNotIn("spoken_text", result["segments"][0])
        self.assertFalse(normalization_pending(result))

    def test_runtime_guards_reject_omissions_names_numbers_negation_and_bad_offsets(self):
        examples = [
            ("wheerrre issss sheeee", "where"), ("never ever", "ever"), ("can't", "can"),
            ("noooo", "yes"), ("red", "blue"), ("not", "now"), ("where is she", "where she"), ("Vern", "John"), ("123", "124"), ("three", "four"),
            ("wull", ""), ("wull", "<whisper>well</whisper>"),
        ]
        for before, after in examples:
            with self.subTest(before=before, after=after):
                source = document(before)
                with self.assertRaises(ValueError):
                    normalize_document_speech(source, Requests({before: after}))
        row = document("wull")["segments"][0]
        change = edit(row["text"], "wull", "well")
        change["start"] = 1
        with self.assertRaises(ValueError):
            _validate_scan({"segments": [{"segment_id": row["id"], "edits": [change]}]}, [row], [])

    def test_contextual_spelling_forms_work_without_a_word_lookup_table(self):
        from epub_tts_narrator.speech_normalization import _word_forms_accounted_for
        for before, after in (("wull", "well"), ("denno", "don't know"), ("Ahh", "I"),
                              ("ssssomething", "something"), ("sez", "says"), ("feller", "fellar")):
            with self.subTest(before=before, after=after):
                self.assertTrue(_word_forms_accounted_for(before, after))
        for before, after in (("red", "blue"), ("where is she", "where she"),
                              ("were", "are"), ("a", "I"), ("go", "stop")):
            with self.subTest(before=before, after=after):
                self.assertFalse(_word_forms_accounted_for(before, after))

    def test_subword_edits_cannot_erase_names_or_negations(self):
        row = document("Vern cannot go.")["segments"][0]
        for original, replacement in (("e", "u"), ("not", "now")):
            change = edit(row["text"], original, replacement)
            with self.assertRaises(ValueError):
                _validate_scan({"segments": [{"segment_id": row["id"], "edits": [change]}]}, [row], ["Vern"])

    def test_case_only_edits_with_empty_style_are_ignored(self):
        row = document("well")["segments"][0]
        change = edit(row["text"], "well", "Well", direction="")
        result = _validate_scan({"segments": [{"segment_id": row["id"], "edits": [change]}]}, [row], [])
        self.assertEqual(result[0]["edits"], [])

    def test_composed_directions_are_validated_before_scanner_response_is_cached(self):
        row = document("wull Ahh sssstupid")["segments"][0]
        row["delivery"]["direction"] = "Speak clearly. " * 50
        edits = [edit(row["text"], old, new, direction=(f"Lengthen '{new}'. " + "Maintain clear articulation. " * 7).strip())
                 for old, new in (("wull", "well"), ("Ahh", "I"), ("sssstupid", "stupid"))]
        with self.assertRaisesRegex(ValueError, "Combined speech delivery exceeds"):
            _validate_scan({"segments": [{"segment_id": row["id"], "edits": edits}]}, [row], [])
        self.assertIn("applied to the WHOLE passage", SCAN)
        self.assertIn("affected normalized word(s)", SCAN)
        self.assertIn("under 1200", SCAN)

    def test_scan_cannot_drop_rows_or_overlap_edits(self):
        source = document("wull wull")
        row = source["segments"][0]
        change = edit(row["text"], "wull", "well")
        for response in ({"segments": []}, {"segments": [{"segment_id": row["id"], "edits": [change, change]}]}):
            with self.assertRaises(ValueError):
                _validate_scan(response, [row], [])

    def test_one_worker_pool_limits_all_calls(self):
        source = document(*["wull" for _ in range(12)])
        requests = Requests({"wull": "well"})
        with patch("epub_tts_narrator.speech_normalization.BATCH_SEGMENTS", 1):
            result = normalize_document_speech(source, requests, workers=3)
        self.assertEqual(result["provenance"]["speech_normalization"]["applied_edits"], 12)
        self.assertLessEqual(requests.maximum, 3)
        self.assertEqual(len(requests.calls), 48)

    def test_series_link_and_publication_preserve_completed_normalization(self):
        from test_contracts import valid_document
        from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory, _completed_directory
        from epub_tts_narrator.cli import export_bundle
        from epub_tts_narrator.series import link_analysis
        from epub_tts_narrator.contracts import validate_document
        source = valid_document()
        source["source"]["blocks"][0]["text"] = source["source"]["blocks"][0]["text"].replace("Hello", "Hellooo")
        source["segments"][1]["text"] = source["segments"][1]["text"].replace("Hello", "Hellooo")
        source["segments"][1]["end"] += 2
        result = normalize_document_speech(source, Requests({"Hellooo": "Hello"}))
        self.assertEqual(validate_document(result), [])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "book"
            registry = fresh_registry(output)
            linked = link_analysis(result, registry)
            self.assertFalse(normalization_pending(linked))
            publish_directory(linked, registry, output, export_bundle)
            saved = _completed_directory(output)[0]
            self.assertFalse(normalization_pending(saved))
            self.assertEqual(saved["segments"][1]["spoken_text"], "“Hello.”")

    def test_interruption_reuses_validated_scan_and_votes_from_request_cache(self):
        from epub_tts_narrator.pipeline import CachedRequests
        from epub_tts_narrator.provider import ProviderError
        class Client:
            identity = {"model": "offline-test", "base_url": "https://example.invalid"}
            def __init__(self):
                self.calls, self.fail = [], True
                self.behavior = Requests({"wull": "well"})
            def complete(self, system, payload):
                self.calls.append(payload.get("reviewer_id", "scan"))
                if payload.get("reviewer_id") == "speech_spelling_reviewer_2" and self.fail:
                    self.fail = False
                    raise ProviderError("Timeout")
                return self.behavior.run(system, payload, lambda response: response)
        source, client = document("wull"), Client()
        with tempfile.TemporaryDirectory() as temporary:
            requests = CachedRequests(client, temporary)
            with self.assertRaises(ProviderError):
                normalize_document_speech(source, requests, workers=1)
            result = normalize_document_speech(source, requests, workers=1)
            self.assertEqual(result["segments"][0]["spoken_text"], "well")
            self.assertEqual(client.calls, ["scan", "speech_spelling_reviewer_1", "speech_spelling_reviewer_2",
                                            "speech_spelling_reviewer_2", "speech_spelling_reviewer_3"])
            self.assertEqual(requests.cache_hits, 2)

    def test_failed_stage_does_not_mutate_document_or_mark_it_complete(self):
        source = document("wull")
        before = deepcopy(source)
        class Failed(Requests):
            def run(self, system, payload, validate, **kwargs):
                if payload["mode"] == "speech_spelling_review":
                    raise RuntimeError("interrupted")
                return super().run(system, payload, validate, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            normalize_document_speech(source, Failed({"wull": "well"}))
        self.assertEqual(source, before)
        self.assertTrue(normalization_pending(source))


if __name__ == "__main__":
    unittest.main()
