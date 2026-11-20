"""``analyze BOOK --instructions``: each section's narration is read by the character whose point of view it follows."""

from copy import deepcopy
import contextlib
import hashlib
import io
import json
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator import narrator_review
from epub_tts_narrator.cli import main
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.narrator_read import _narrating_description, build_cast
from epub_tts_narrator.renderer.speech_traits import accent_cues
from epub_tts_narrator.renderer.voice_bindings import linked_voices
from epub_tts_narrator.analysis_revision import revise_directories
from epub_tts_narrator.pipeline import fingerprint, write_json
from epub_tts_narrator.tui_support import screen_host
from shared_fixtures import ScreenHost, skip_rerun_reviews
from test_attribution_verification_cli import AttributionCLIHarness

NOTES = "The point of view changes between scenes; each scene's narration follows one character."


class PointOfViewClient:
    """Names Mara for a section that mentions her and Tom for one that mentions him."""

    def __init__(self, asks="y"):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.calls, self.lock, self.asks = [], Lock(), asks

    def sections_checked(self):
        return [call for call in self.calls if call["mode"] == "narrator_review"]

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        if payload["mode"] == "narrator_notes":
            return {"r": self.asks}
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
        self.assertEqual({key: roles[mara][key] for key in ("kind", "character_id", "description")}, {
            "kind": "narration", "character_id": cast["Mara"]["id"],
            "description": "Narration of the sections told from Mara's point of view."})
        # The narration has a voice of its own, linked to the character's recording and accent.
        links = linked_voices(analysis["cast"])
        self.assertEqual((links[roles[mara]["voice_id"]], links[roles[tom]["voice_id"]]),
                         (cast["Mara"]["voice_id"], cast["Tom"]["voice_id"]))
        self.assertNotIn(roles[mara]["voice_id"], {cast["Mara"]["voice_id"], cast["Tom"]["voice_id"]})
        bundle = {"cast": analysis["cast"], "source": analysis["source"], "directory": str(self.book)}
        with (patch("epub_tts_narrator.renderer.speech_traits.accent_overrides", return_value={}),
              patch("epub_tts_narrator.renderer.speech_traits.speech_trait_overrides", return_value={}),
              patch("epub_tts_narrator.renderer.accent_assignment.assigned_accents",
                    return_value={cast["Mara"]["voice_id"]: "Scottish"})):
            cues = accent_cues(bundle, {})
        self.assertEqual(cues[roles[mara]["voice_id"]], "Scottish")
        readers = self.readers()
        self.assertEqual(readers, {**readers, "b1:0:18": mara, "b2:0:10": mara, "b3:0:1": "narrator_main",
                                   "b4:0:34": tom})
        self.assertNotIn(readers["b1:18:26"], {mara, tom})
        voices = {row["id"]: row["voice_id"] for row in analysis["segments"]}
        self.assertEqual((voices["b2:0:10"], voices["b4:0:34"]), (roles[mara]["voice_id"], roles[tom]["voice_id"]))
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

    def test_notes_about_the_cast_leave_the_books_narrator(self):
        # Close third person whose point of view moves (Eddings) keeps its one storyteller.
        self.make_book(risk=False, prepare=two_scenes)
        manifest = (self.book / "manifest.json").read_bytes()
        client = PointOfViewClient(asks="n")
        changed, output = self.review(client, instructions="Velvet and Liselle are the same person.")
        self.assertFalse(changed, output)
        self.assertEqual(client.sections_checked(), [])
        self.assertIn("stays with the book's narrator", output)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_unclear_notes_ask_the_person_first(self):
        self.make_book(risk=False, prepare=two_scenes)
        manifest = (self.book / "manifest.json").read_bytes()
        for interactive, answer in ((True, "k\n"), (False, "")):
            client = PointOfViewClient(asks="?")
            changed, output = self.review(client, interactive=interactive, answer=answer)
            self.assertFalse(changed, output)
            self.assertEqual(client.sections_checked(), [])
            self.assertEqual("[P] read each section's narration" in output, interactive)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)
        client = PointOfViewClient(asks="?")
        changed, output = self.review(client, answer="p\na\n")
        self.assertTrue(changed, output)
        self.assertIn("[P] read each section's narration", output)

    def test_later_notes_about_something_else_keep_the_saved_narrators(self):
        # ECHO: its saved point-of-view notes stay in force when a later rerun's notes are about the cast.
        self.make_book(risk=False, prepare=two_scenes)
        self.review(PointOfViewClient())
        readers, manifest = self.readers(), (self.book / "manifest.json").read_bytes()
        client = PointOfViewClient(asks="n")
        changed, output = self.review(client, instructions="Tom and Mara are different people.")
        self.assertFalse(changed, output)
        self.assertEqual(client.sections_checked(), [])
        self.assertIn("stays with its saved narrators", output)
        self.assertEqual((self.readers(), (self.book / "manifest.json").read_bytes()), (readers, manifest))
        self.assertEqual(self.analysis()["provenance"]["narrator_review"]["notes"], NOTES)

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

    def test_in_the_run_shell_the_sections_and_readers_have_their_own_screen(self):
        self.make_book(risk=False, prepare=two_scenes)
        manifest = (self.book / "manifest.json").read_bytes()
        host = ScreenHost("l")
        with screen_host(host):
            changed, output = self.review(PointOfViewClient(), answer="")
        self.assertFalse(changed)
        self.assertNotIn("[A] apply these narrators", output)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)
        [screen] = host.screens
        self.assertEqual(type(screen).__name__, "NarratorScreen")
        view = screen.view
        self.assertEqual(view["notes"], NOTES)
        self.assertEqual([(row["reader"], row["pov"]) for row in view["sections"]], [("Mara", True), ("Tom", True)])
        self.assertEqual(view["moved"], sum(row["changed"] for row in view["sections"]))
        self.assertGreater(view["moved"], 0)
        self.assertTrue(all(row["quote"] for row in view["sections"]))
        with screen_host(ScreenHost("a")):
            changed, output = self.review(PointOfViewClient(), answer="")
        self.assertTrue(changed, output)

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
        with self.assertRaisesRegex(ValueError, "new voice linked to an existing character"):
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
        self.assertEqual([(call["mode"], call["notes"]) for call in client.calls],
                         [("narrator_notes", NOTES), ("narrator_review", NOTES)])
        self.assertIn("Narrator check: whose point of view", output.getvalue())

    def test_a_first_person_narrator_has_its_own_voice_linked_to_the_character(self):
        row = {"id": "7", "name": "Wren", "aliases": [], "note": "The narrator.", "voice": [], "series": None}
        ledger = {"characters": [row], "narration": {"mode": "first", "narrator": "7"}}
        cast = build_cast({"blocks": []}, ledger, {"7"})
        narrator = next(role for role in cast["roles"] if role["id"] == "narrator_main")
        self.assertEqual((narrator["character_id"], narrator["voice_id"]), ("char_7", "voice_narrator_main"))
        self.assertEqual(linked_voices(cast), {"voice_narrator_main": "voice_7"})
        next(voice for voice in cast["voices"] if voice["id"] == "voice_7")["description"] = "Female, dry and low."
        described = _narrating_description(cast, "voice_narrator_main", "voice_7")
        self.assertEqual(next(voice for voice in described["voices"] if voice["id"] == "voice_narrator_main")
                         ["description"], "Female, dry and low.")

    def test_auditions_follow_the_new_cast(self):
        self.make_book(risk=False, prepare=two_scenes)
        document = self.analysis()
        write_json(self.book / "audition-scripts.json",
                   {"version": 1, "source_fingerprint": fingerprint(document["source"]),
                    "cast_fingerprint": fingerprint(document["cast"]), "scripts": {}})
        self.assertTrue(self.review(PointOfViewClient())[0])
        saved = json.loads((self.book / "audition-scripts.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["cast_fingerprint"], fingerprint(self.analysis()["cast"]))

    def test_a_narrator_on_the_characters_own_voice_moves_to_a_linked_voice(self):
        def old_shape(document):
            # What the first version saved: a narration role on the character's own voice.
            two_scenes(document)
            document["cast"]["roles"].append({"id": "pov_mara", "kind": "narration", "character_id": "c1",
                                              "voice_id": "v1", "description": "Mara's point of view."})
            document["segments"][2].update(role_id="pov_mara", voice_id="v1")
        self.make_book(risk=False, prepare=old_shape)
        document = self.analysis()
        mara = next(row for row in document["cast"]["characters"] if row["name"] == "Mara")
        self.assertEqual(narrator_review.upgradable_roles(document), ["pov_mara"])
        revised = narrator_review.apply_narrators(document, {})
        self.assertEqual(narrator_review.validate_narrator_revision(document, revised),
                         [narrator_review.narrator_voice_id(document, "pov_mara")])
        role = next(item for item in revised["cast"]["roles"] if item["id"] == "pov_mara")
        self.assertEqual(linked_voices(revised["cast"])[role["voice_id"]], mara["voice_id"])
        self.assertEqual(next(row for row in revised["segments"] if row["id"] == "b2:0:10")["voice_id"],
                         role["voice_id"])
        hashes = {str(self.book): hashlib.sha256((self.book / "manifest.json").read_bytes()).hexdigest()}
        revise_directories({self.book: revised}, directories=[self.book], expected_manifest_hashes=hashes,
                           scope="narrators", progress=lambda _: None)
        self.assertEqual(narrator_review.upgradable_roles(self.analysis()), [])

if __name__ == "__main__":
    unittest.main()
