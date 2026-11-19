"""``analyze BOOK --instructions``: each section's narration is read by the character whose point of view it follows."""

from copy import deepcopy
import contextlib
import io
import json
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator import narrator_review
from epub_tts_narrator.cli import main
from epub_tts_narrator.config import DEFAULTS
from shared_fixtures import skip_rerun_reviews
from test_attribution_verification_cli import AttributionCLIHarness

NOTES = "The point of view changes between scenes; each scene's narration follows one character."


class PointOfViewClient:
    """Names Mara for a section that mentions her and Tom for one that mentions him."""

    def __init__(self):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.calls, self.lock = [], Lock()

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        assert payload["mode"] == "narrator_review", payload["mode"]
        ids = {line.split("|")[1]: line.split("|")[0] for line in payload["cast"]}
        answers = []
        for item in payload["items"]:
            for line in item["text"].splitlines():
                number, text = line.split("|", 1)
                name = next((name for name in ("Tom", "Mara") if name in text), None)
                if name:
                    answers.append([item["id"], ids[name], int(number), text[:10]])
                    break
            else:
                answers.append([item["id"], "-"])
        return {"r": answers}


def two_scenes(document):
    """Mara's scene (b1, b2), a scene break (b3) and Tom's scene (b4), all narration outside Mara's line."""
    blocks = document["source"]["blocks"]
    for block_id, text in (("b3", "#"), ("b4", "Tom watched the door and wondered.")):
        blocks.append({**deepcopy(blocks[1]), "id": block_id, "text": text})
        document["source"]["chapters"][0]["block_ids"].append(block_id)
    segments = document["segments"]
    segments[2].update(kind="narration", role_id="narrator_main", voice_id="vn")
    for block_id, text in (("b3", "#"), ("b4", "Tom watched the door and wondered.")):
        segments.append({**deepcopy(segments[2]), "id": f"{block_id}:0:{len(text)}", "block_id": block_id,
                         "start": 0, "end": len(text), "text": text})


class NarratorReviewTests(AttributionCLIHarness):
    def review(self, client, *, instructions=NOTES, interactive=True, answer="a\n"):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        output = io.StringIO()
        with (patch("sys.stdin", io.StringIO(answer)),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO())):
            changed = narrator_review.review_narrators(self.book, config, lambda settings: client,
                                                       lambda message: None, instructions=instructions,
                                                       interactive=interactive)
        return changed, output.getvalue()

    def analysis(self):
        return json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))

    def readers(self):
        return {row["id"]: row["role_id"] for row in self.analysis()["segments"]}

    def test_each_scene_is_narrated_in_its_point_of_view_characters_voice(self):
        self.make_book(risk=False, prepare=two_scenes)
        changed, output = self.review(PointOfViewClient())
        self.assertTrue(changed, output)
        analysis = self.analysis()
        cast = {row["name"]: row for row in analysis["cast"]["characters"]}
        mara, tom = (f"narrator_{cast[name]['id']}" for name in ("Mara", "Tom"))
        roles = {role["id"]: role for role in analysis["cast"]["roles"]}
        self.assertEqual(roles[mara], {
            "id": mara, "kind": "narration", "character_id": cast["Mara"]["id"], "voice_id": cast["Mara"]["voice_id"],
            "description": "Narration of the sections told from Mara's point of view."})
        self.assertEqual(roles[tom]["voice_id"], cast["Tom"]["voice_id"])
        readers = self.readers()
        self.assertEqual(readers, {**readers, "b1:0:18": mara, "b2:0:10": mara, "b3:0:1": "narrator_main",
                                   "b4:0:34": tom})
        self.assertNotIn(readers["b1:18:26"], {mara, tom})
        voices = {row["id"]: row["voice_id"] for row in analysis["segments"]}
        self.assertEqual((voices["b2:0:10"], voices["b4:0:34"]), (cast["Mara"]["voice_id"], cast["Tom"]["voice_id"]))
        self.assertEqual(analysis["provenance"]["narrator_review"]["notes"], NOTES)
        self.assertIn("Tom", output)

    def test_a_later_rerun_uses_the_saved_notes_and_asks_nothing(self):
        self.make_book(risk=False, prepare=two_scenes)
        self.review(PointOfViewClient())
        manifest = (self.book / "manifest.json").read_bytes()
        client = PointOfViewClient()
        changed, output = self.review(client, instructions=None)
        self.assertFalse(changed, output)
        self.assertEqual(client.calls, [])
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_without_notes_nothing_is_checked(self):
        self.make_book(risk=False, prepare=two_scenes)
        client = PointOfViewClient()
        self.assertFalse(self.review(client, instructions=None)[0])
        self.assertEqual(client.calls, [])

    def test_without_a_terminal_the_narrators_are_only_listed(self):
        self.make_book(risk=False, prepare=two_scenes)
        manifest = (self.book / "manifest.json").read_bytes()
        changed, output = self.review(PointOfViewClient(), interactive=False)
        self.assertFalse(changed)
        self.assertIn("nothing was saved", output)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_leaving_the_narration_saves_nothing(self):
        self.make_book(risk=False, prepare=two_scenes)
        manifest = (self.book / "manifest.json").read_bytes()
        changed, _ = self.review(PointOfViewClient(), answer="l\n")
        self.assertFalse(changed)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_the_narrators_scope_refuses_anything_but_moving_narration(self):
        self.make_book(risk=False, prepare=two_scenes)
        document = self.analysis()
        mara = next(row for row in document["cast"]["characters"] if row["name"] == "Mara")
        tom = next(row for row in document["cast"]["characters"] if row["name"] == "Tom")
        chosen = {section["key"]: mara["id"] for section in narrator_review.sections(document)}
        revised = narrator_review.apply_narrators(document, chosen)
        narrator_review.validate_narrator_revision(document, revised)
        dialogue = deepcopy(revised)
        tom_role = next(role for role in document["cast"]["roles"] if role["character_id"] == tom["id"])
        dialogue["segments"][1].update(role_id=tom_role["id"], voice_id=tom["voice_id"])
        with self.assertRaisesRegex(ValueError, "only move narration"):
            narrator_review.validate_narrator_revision(document, dialogue)
        stranger = deepcopy(revised)
        stranger["cast"]["roles"][-1]["voice_id"] = tom["voice_id"]
        with self.assertRaisesRegex(ValueError, "existing character's voice"):
            narrator_review.validate_narrator_revision(document, stranger)

    def test_analyze_book_with_instructions_runs_the_check(self):
        self.make_book(risk=False, prepare=two_scenes)
        skip_rerun_reviews(self)
        config = {**deepcopy(DEFAULTS), "analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {**deepcopy(DEFAULTS["provider"]), "model": "test-model",
                               "base_url": "https://example.invalid"}}
        client = PointOfViewClient()
        output, errors = io.StringIO(), io.StringIO()
        with (patch("epub_tts_narrator.cli.load_config", return_value=config),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client),
              patch("sys.stdin", io.StringIO()),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["analyze", str(self.book), "--instructions", NOTES])
        self.assertIn(code, (0, 2), errors.getvalue() + output.getvalue())
        self.assertEqual([call["notes"] for call in client.calls], [NOTES])
        self.assertIn("Narrator check: whose point of view", output.getvalue())


if __name__ == "__main__":
    unittest.main()
