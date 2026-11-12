"""The book-wide accent assignment groups the cast by origin and feeds every voice's accent cue."""

from copy import deepcopy
from pathlib import Path
import io
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer import accent_assignment
from epub_tts_narrator.renderer.accent_assignment import (
    ASSIGNMENTS_FILENAME, GUIDANCE_FILENAME, accent_cue, assigned_accents, guidance, prepare_accents, read_groups,
    request_assignment, run_accents_command, series_origins, voice_entries,
)
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from epub_tts_narrator.renderer.speech_traits import accent_cues, save_accent


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


class FakeRequests:
    """Answers each request with ``reply(payload)``; records what was asked."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def run(self, system, payload, validate):
        self.calls.append((system, deepcopy(payload)))
        return validate(self.reply(payload))


def everyone(accent, origin="the town"):
    return lambda payload: {"g": [[origin, accent, [row[0] for row in payload["voices"]]]]}


class BookTestCase(unittest.TestCase):
    """A copy of the example book, its config and voices; no tests of its own."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "book"
        shutil.copytree(EXAMPLE, self.directory)
        self.bundle = prepare_bundle(load_bundle(self.directory))
        self.config = resolve_book_config(self.bundle, deepcopy(DEFAULTS))
        self.entries = voice_entries(self.bundle)
        self.voices = sorted(self.entries)

    def prepare(self, requests, **kwargs):
        with patch.object(accent_assignment, "_requests", return_value=requests), \
                patch.object(accent_assignment, "_book_cache", return_value=self.directory / ".cache"):
            return prepare_accents(self.bundle, self.config, None, notify=lambda _: None, **kwargs)

    def run_command(self, requests, *, instructions=None, apply=False):
        args = SimpleNamespace(instructions=instructions, apply=apply, resume=None, keep=False)
        output = io.StringIO()
        with patch.object(accent_assignment, "_requests", return_value=requests), \
                patch.object(accent_assignment, "_book_cache", return_value=self.directory / ".cache"), \
                patch("sys.stdout", output), patch("sys.stderr", io.StringIO()):
            code = run_accents_command(self.bundle, self.config, args, None)
        return code, output.getvalue()


class AssignmentTests(BookTestCase):
    def test_each_speaking_voice_is_described_by_its_names_and_where_the_story_introduces_it(self):
        self.assertTrue(self.voices)
        for voice_id, entry in self.entries.items():
            character = next(c for c in self.bundle["cast"]["characters"] if c["voice_id"] == voice_id)
            self.assertIn(character["name"], entry["names"])
            self.assertTrue(entry["intro"])
            self.assertTrue(any(character["name"] in text for text in entry["intro"]))

    def test_a_voice_reading_the_authors_preface_is_named_as_the_author_not_the_narrator(self):
        bundle = deepcopy(self.bundle)
        bundle["source"]["book"]["authors"] = ["Ada Quill"]
        voice_id = self.voices[0]
        bundle["cast"]["characters"] = [c for c in bundle["cast"]["characters"] if c["voice_id"] != voice_id]
        bundle["cast"]["roles"] = [role for role in bundle["cast"]["roles"] if role["voice_id"] != voice_id] + [
            {"id": "role_author", "kind": "narration", "character_id": None, "voice_id": voice_id,
             "description": "Author narration for the preface, read as the book's author."}]
        self.assertEqual(voice_entries(bundle)[voice_id]["label"], "Ada Quill (author)")

    def test_a_returning_character_is_introduced_from_the_earliest_book_that_names_them(self):
        voice_id = self.voices[0]
        character = next(c for c in self.bundle["cast"]["characters"] if c["voice_id"] == voice_id)
        earlier = {"cast": {"characters": [{"name": character["name"], "aliases": ["the Harbormaster"],
                                            "voice_id": voice_id, "description": "A fisher from the northern isles"}]},
                   "source": {"blocks": [{"text": "Nothing here."},
                                         {"text": f"The Harbormaster, {character['name']}, came from the northern isles."}]}}
        entry = voice_entries(self.bundle, [earlier])[voice_id]
        self.assertIn("came from the northern isles", entry["intro"][0])
        self.assertTrue(entry["about"].startswith("A fisher from the northern isles (when first met)"))

    def test_each_voice_goes_with_its_designed_description_and_traits(self):
        voice_id = self.voices[0]
        profile = next(v for v in self.bundle["cast"]["voices"] if v["id"] == voice_id)
        profile.update(description="Deep voice, sturdy Yorkshire baritone", speechTraits="Sturdy Yorkshire; unhurried")
        requests = FakeRequests(everyone("Yorkshire"))
        self.prepare(requests)
        row = next(row for row in requests.calls[0][1]["voices"] if row[1] == self.entries[voice_id]["names"])
        self.assertEqual(row[-2:], ["Deep voice, sturdy Yorkshire baritone", "Sturdy Yorkshire; unhurried"])

    def test_an_accent_in_traits_a_person_set_is_their_choice(self):
        from epub_tts_narrator.renderer.speech_traits import save_speech_traits
        voice_id = self.voices[0]
        save_speech_traits(self.bundle, self.config, voice_id, "General American; cool measured pace")
        requests = FakeRequests(everyone("Welsh"))
        self.prepare(requests)
        payload = requests.calls[0][1]
        self.assertEqual(payload["fixed"], [["f1", self.entries[voice_id]["names"], "General American accent"]])
        self.assertNotIn(self.entries[voice_id]["names"], [row[1] for row in payload["voices"]])
        cues = accent_cues(self.bundle, self.config)
        self.assertEqual(cues[voice_id], "General American accent")
        self.assertEqual(cues[self.voices[1]], "Welsh accent")
        # Traits that name no accent are only a clue: the next assignment places that voice too.
        save_speech_traits(self.bundle, self.config, voice_id, "cool measured pace, clipped consonants")
        self.assertNotIn(voice_id, accent_cues(self.bundle, self.config))
        again = FakeRequests(everyone("Irish", "the coast"))
        self.prepare(again)
        row = next(row for row in again.calls[0][1]["voices"] if row[1] == self.entries[voice_id]["names"])
        self.assertEqual(row[-1], "cool measured pace, clipped consonants")
        self.assertEqual(accent_cues(self.bundle, self.config)[voice_id], "Irish accent")

    def test_groups_become_each_voices_cue_and_a_voice_left_out_gets_none(self):
        first, *rest = self.voices
        requests = FakeRequests(lambda payload: {"g": [["the harbor", "broad Scottish", ["v1"]],
                                                        ["inland", "Welsh accent", ["v2"]]]})
        result = request_assignment(self.bundle, requests, self.voices, {})
        self.assertEqual(result["accents"][first], "broad Scottish accent")
        if len(rest) > 1:
            self.assertNotIn(rest[-1], result["accents"])
        self.assertEqual(result["groups"][0]["origin"], "the harbor")
        self.assertEqual([row[1] for row in requests.calls[0][1]["voices"]],
                         [self.entries[voice_id]["names"] for voice_id in self.voices])

    def test_an_accent_named_after_the_book_or_without_a_place_is_refused_so_the_model_repairs_it(self):
        for accent in ("Mara", "heavy", "", "no"):
            with self.subTest(accent=accent), self.assertRaises(ValueError):
                request_assignment(self.bundle, FakeRequests(everyone(accent)), self.voices, {},
                                   terms=frozenset({"mara"}))
        for accent, cue in (("Received Pronunciation", "Received Pronunciation accent"),
                            ("Hungarian", "Hungarian accent"), ("thick Geordie accent", "thick Geordie accent")):
            self.assertEqual(accent_cue(accent), cue)

    def test_a_render_assigns_only_voices_without_an_accent_around_the_decided_ones(self):
        chosen, other = self.voices[0], self.voices[1]
        save_accent(self.bundle, self.config, chosen, "heavy Brooklyn accent")
        requests = FakeRequests(everyone("Welsh"))
        self.prepare(requests)
        payload = requests.calls[0][1]
        self.assertEqual([row[1] for row in payload["voices"]],
                         [self.entries[voice_id]["names"] for voice_id in self.voices[1:]])
        self.assertEqual(payload["fixed"], [["f1", self.entries[chosen]["names"], "heavy Brooklyn accent"]])
        self.assertEqual(assigned_accents(self.bundle, self.config), {v: "Welsh accent" for v in self.voices[1:]})
        cues = accent_cues(self.bundle, self.config)
        self.assertEqual((cues[chosen], cues[other]), ("heavy Brooklyn accent", "Welsh accent"))
        # Nothing is pending afterwards, so the next render asks nothing.
        again = FakeRequests(everyone("Irish"))
        self.assertIsNone(self.prepare(again))
        self.assertEqual(again.calls, [])

    def test_casting_shows_an_assigned_cue_and_where_it_came_from(self):
        from epub_tts_narrator.renderer.casting import CastingSession
        self.prepare(FakeRequests(everyone("Welsh")))
        session = CastingSession(self.bundle, self.config, notify=lambda _: None)
        self.addCleanup(session.close)
        row = next(row for row in session.rows() if row["voice_id"] == self.voices[0])
        self.assertEqual((row["accent"], row["accent_source"], row["accent_automatic"]),
                         ("Welsh accent", "assigned", True))

    def test_an_assignment_replaces_a_descriptions_default_accent(self):
        voice_id = self.voices[0]
        next(v for v in self.bundle["cast"]["voices"] if v["id"] == voice_id)["description"] = \
            "Warm voice with a neutral General American accent"
        self.assertEqual(accent_cues(self.bundle, self.config)[voice_id], "neutral General American accent")
        self.prepare(FakeRequests(everyone("soft Irish")))
        self.assertEqual(accent_cues(self.bundle, self.config)[voice_id], "soft Irish accent")

    def test_assignment_can_be_turned_off_and_needs_a_model(self):
        requests = FakeRequests(everyone("Welsh"))
        self.config["casting"]["assign_accents"] = False
        self.assertIsNone(self.prepare(requests))
        self.config["casting"]["assign_accents"] = True
        self.assertIsNone(self.prepare(None))
        self.assertEqual(requests.calls, [])
        self.assertFalse((self.directory / ASSIGNMENTS_FILENAME).exists())

    def test_the_command_reports_without_saving_unless_applied_and_keeps_guidance_for_later(self):
        self.prepare(FakeRequests(everyone("Welsh")))
        code, output = self.run_command(FakeRequests(everyone("Yorkshire")), instructions="They are all from Leeds")
        self.assertEqual(code, 0)
        self.assertIn("Report only", output)
        self.assertEqual(set(assigned_accents(self.bundle, self.config).values()), {"Welsh accent"})
        requests = FakeRequests(everyone("Yorkshire"))
        self.run_command(requests, instructions="They are all from Leeds", apply=True)
        # The book already has a plan, so the instructions revise it.
        self.assertEqual(requests.calls[0][1]["revise"], "They are all from Leeds")
        self.assertEqual(requests.calls[0][1]["plan"][0][1], "Welsh accent")
        self.assertNotIn("fixed", requests.calls[0][1])  # this book's own assignments are what is revised
        self.assertEqual(set(assigned_accents(self.bundle, self.config).values()), {"Yorkshire accent"})
        self.assertEqual(guidance(self.bundle, self.config), "They are all from Leeds")
        self.assertTrue((self.directory / GUIDANCE_FILENAME).is_file())

    def test_requests_without_guidance_carry_no_notes(self):
        requests = FakeRequests(everyone("Welsh"))
        self.prepare(requests)
        system, payload = requests.calls[0]
        self.assertNotIn("notes", payload)
        self.assertNotIn('"notes"', system)


class SeriesOriginTests(BookTestCase):
    """Groups are saved with their reasons; later books keep the series' origins."""

    def reasoned(self, origin, accent, why="They are islanders"):
        return lambda payload: {"g": [[origin, accent, why, [row[0] for row in payload["voices"]]]]}

    def test_groups_are_saved_with_their_origin_and_reason(self):
        self.prepare(FakeRequests(self.reasoned("the harbor town", "broad Scottish")))
        [group] = read_groups(self.directory, self.bundle["source"]["book"]["id"])
        self.assertEqual((group["origin"], group["accent"], group["why"]),
                         ("the harbor town", "broad Scottish accent", "They are islanders"))
        self.assertEqual(sorted(group["voice_ids"]), self.voices)
        self.assertEqual(series_origins(self.bundle, self.config)[0]["accent"], "broad Scottish accent")

    def test_a_series_origin_keeps_its_accent_and_a_mismatch_is_asked_again(self):
        origins = [{"origin": "the harbor town", "accent": "broad Scottish accent", "why": "", "names": ["Old Tam"]}]
        requests = FakeRequests(self.reasoned("The Harbor  Town", "Irish"))
        with self.assertRaisesRegex(ValueError, "already has the accent"):
            request_assignment(self.bundle, requests, self.voices, {}, origins=origins)
        self.assertEqual(requests.calls[0][1]["origins"], [["the harbor town", "broad Scottish accent", "", ["Old Tam"]]])
        result = request_assignment(self.bundle, FakeRequests(self.reasoned("the harbor town", "broad Scottish")),
                                    self.voices, {}, origins=origins)
        self.assertEqual(set(result["accents"].values()), {"broad Scottish accent"})

    def test_keep_records_reasons_without_changing_an_accent(self):
        self.prepare(FakeRequests(everyone("Welsh")))
        own = assigned_accents(self.bundle, self.config, own=True)
        changed = FakeRequests(self.reasoned("the valley", "Irish"))
        with self.assertRaisesRegex(ValueError, "must keep the accent"):
            request_assignment(self.bundle, changed, self.voices, {}, keep=own)
        args = SimpleNamespace(instructions=None, apply=True, resume=None, keep=True)
        requests = FakeRequests(self.reasoned("the valley", "Welsh", "Valley miners"))
        with patch.object(accent_assignment, "_requests", return_value=requests), \
                patch.object(accent_assignment, "_book_cache", return_value=self.directory / ".cache"), \
                patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            run_accents_command(self.bundle, self.config, args, None)
        self.assertEqual(sorted(cue for _, cue in requests.calls[0][1]["keep"]), sorted(own.values()))
        self.assertEqual(assigned_accents(self.bundle, self.config, own=True), own)
        [group] = read_groups(self.directory, self.bundle["source"]["book"]["id"])
        self.assertEqual((group["origin"], group["why"]), ("the valley", "Valley miners"))


class ConfigTests(unittest.TestCase):
    def test_assign_accents_must_be_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audiobook-render.toml"
            path.write_text('[casting]\nassign_accents = "yes"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assign_accents"):
                load_config(path, require_verifier=False, require_engine=False)


if __name__ == "__main__":
    unittest.main()
