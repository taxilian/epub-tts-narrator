"""A person says two cast entries are one character: the merge is published across the series."""

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from epub_tts_narrator.analysis_directory import (_completed_directory, continuation_directories,
                                                  continuation_registry, fresh_registry, publish_directory)
from epub_tts_narrator.analysis_progress import analysis_cache_directory
from epub_tts_narrator.analysis_revision import revise_directories
from epub_tts_narrator.audition_scripts import FILENAME as AUDITIONS
from epub_tts_narrator.character_merge import (merge_document, receipt, remap_decisions, resolve_character,
                                               revoiced_segment_ids)
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.character_merge_command import apply_merge, plan_merge, run_merge_command
from epub_tts_narrator.series import link_analysis, validate_registry
from epub_tts_narrator.speaker_decisions import (_load_decision_checkpoint, _save_decision_checkpoint,
                                                 decision_checkpoint_key, prompt_speaker_decisions)
from shared_fixtures import Terminal, renderer_config


AUDITION = {"text": ("The tide is turning already, and we should reach the far light before it fails. "
                     "Keep the lantern low and tell me what you see along the water."),
            "basis": "casting", "reason": "A source-independent character audition."}


def build(book_id, title, characters, paragraphs):
    """A valid local analysis: ``characters`` are (local id, name, aliases); ``paragraphs`` are lists of
    (text, speaker) pieces where speaker is None (narration), a character id, a tuple (said together),
    or ``"?"`` (unresolved, with every character as a candidate)."""
    blocks, segments = [], []
    for number, pieces in enumerate(paragraphs, 1):
        block_id = f"b{number}"
        text = "".join(piece for piece, _ in pieces)
        blocks.append({"id": block_id, "chapter_id": "ch1", "text": text,
                       "source": {"href": "text/ch1.xhtml", "element_id": None, "tag": "p"}})
        start = 0
        for index, (piece, speaker) in enumerate(pieces):
            end = start + len(piece)
            segment = {"id": f"{block_id}:{index}", "block_id": block_id, "start": start, "end": end,
                       "text": piece, "kind": "narration" if speaker is None else "dialogue",
                       "delivery": {"direction": "", "evidence": []},
                       "attribution": {"basis": "explicit", "reason": "Named in the text.", "evidence": [],
                                       "candidates": [], "review_required": False}}
            if speaker is None:
                segment.update(role_id="narrator_main", voice_id="vn")
            elif speaker == "?":
                segment.update(role_id=None, voice_id=None)
                segment["attribution"].update(basis="unresolved", review_required=True, reason="Unclear speaker.",
                                              candidates=[f"{local}_talk" for local, _, _ in characters])
            elif isinstance(speaker, tuple):
                segment.update(role_id=None, voice_id=None, simultaneous_speakers=[
                    {"role_id": f"{local}_talk", "voice_id": f"v_{local}"} for local in speaker])
                segment["attribution"]["evidence"] = [{"block_id": block_id, "quote": piece}]
            else:
                segment.update(role_id=f"{speaker}_talk", voice_id=f"v_{speaker}")
            segments.append(segment)
            start = end
    return {
        "format": "epub-audiobook-analysis", "version": 1,
        "source": {"book": {"id": book_id, "title": title, "authors": ["A. Writer"], "language": "en"},
                   "chapters": [{"id": "ch1", "title": "Chapter One", "source_href": "text/ch1.xhtml",
                                 "block_ids": [block["id"] for block in blocks]}],
                   "blocks": blocks, "exclusions": [], "warnings": [],
                   "extraction_policy": "Read spine text; omit navigation."},
        "cast": {
            "characters": [{"id": local, "name": name, "aliases": list(aliases), "description": f"{name} speaking.",
                            "evidence": [{"block_id": "b1", "quote": blocks[0]["text"][:4]}],
                            "voice_id": f"v_{local}"} for local, name, aliases in characters],
            "roles": [{"id": "narrator_main", "kind": "narration", "character_id": None, "voice_id": "vn",
                       "description": "The book's narrator."}] + [
                      {"id": f"{local}_talk", "kind": "dialogue", "character_id": local, "voice_id": f"v_{local}",
                       "description": f"{name} speaking."} for local, name, _ in characters],
            "voices": [{"id": "vn", "description": "A measured reading voice.", "traits": []}] + [
                       {"id": f"v_{local}", "description": f"{name}'s voice.", "traits": []}
                       for local, name, _ in characters],
        },
        "segments": segments, "issues": [], "provenance": {"tool_version": "test"},
    }


FIRST = [
    [("Mara waved. ", None), ("“Hello,” ", "mara"), ("she said.", None)],
    [("The keeper called, ", None), ("“Come in.”", "keeper")],
    [("“Together!” ", ("mara", "keeper")), ("they cried.", None)],
    [("Someone asked, ", None), ("“Who is there?”", "?")],
    [("Mara laughed. ", None), ("“Only me.”", "mara")],
]
SECOND = [
    [("The keeper returned. ", None), ("“Back again,”", "keeper")],
    [("Rain fell.", None)],
]


class CharacterMergeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def publish(self, later_characters=(("keeper", "The keeper", ()),), later=SECOND):
        first = build("merge-book-1", "The Lantern", [("mara", "Mara", ()), ("keeper", "The keeper", ("Keeper",))],
                      FIRST)
        self.paths = [self.root / "book-1", self.root / "book-2"]
        one = link_analysis(first, fresh_registry(self.paths[0]))
        one["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(one["source"]),
            "cast_fingerprint": fingerprint(one["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in one["cast"]["voices"]}, "samples": {}}
        publish_directory(one, fresh_registry(self.paths[0]), self.paths[0], export_bundle)
        self.ids = {row["name"]: row for row in one["cast"]["characters"]}
        second = build("merge-book-2", "The Return", list(later_characters), later)
        registry = continuation_registry(self.paths[0], second["source"])
        for character in second["cast"]["characters"]:
            prior = self.ids.get(character["name"])
            if prior is None:
                continue
            character["series_identity"] = {
                "character_id": prior["id"], "basis": "context", "review_required": False,
                "reason": "The same lighthouse character returns.",
                "evidence": [{"block_id": "b1", "quote": second["source"]["blocks"][0]["text"]}]}
        two = link_analysis(second, registry)
        for character in two["cast"]["characters"]:
            if character["name"] in self.ids:
                self.assertEqual(character["id"], self.ids[character["name"]]["id"], "fixture identity must carry over")
        publish_directory(two, registry, self.paths[1], export_bundle, continue_from=self.paths[0])
        self.mara, self.keeper = self.ids["Mara"], self.ids["The keeper"]
        self.narrator_voice = next(role["voice_id"] for role in one["cast"]["roles"] if role["id"] == "narrator_main")
        return one, two

    def write(self, path, name, record):
        (path / name).write_text(json.dumps(record), encoding="utf-8")

    def read(self, path, name):
        return json.loads((path / name).read_text(encoding="utf-8"))

    def setting(self, path, name, field, values, series_id):
        self.write(path, name, {"format": {"voice-accents.json": "audiobook-book-voice-accents",
                                           "voice-traits.json": "audiobook-book-voice-traits"}[name],
                                "version": 1, "series_id": series_id,
                                "book_id": self.read(path, "manifest.json")["book"]["id"], field: values})

    def test_merge_moves_every_line_drops_the_entry_and_carries_it_through_a_later_book(self):
        one, two = self.publish()
        series_id = one["provenance"]["series"]["series_id"]
        mara_voice, keeper_voice = self.mara["voice_id"], self.keeper["voice_id"]
        self.setting(self.paths[0], "voice-accents.json", "accents",
                     {mara_voice: "light Welsh accent", keeper_voice: "broad Scots accent"}, series_id)
        self.setting(self.paths[1], "voice-traits.json", "speech_traits", {keeper_voice: "Slow and gravelly."},
                     series_id)
        self.write(self.paths[0], "voice-aliases.json", {
            "format": "audiobook-book-voice-aliases", "version": 1, "series_id": series_id,
            "book_id": "merge-book-1", "aliases": {self.narrator_voice: keeper_voice}})
        self.write(self.paths[0], AUDITIONS, {
            "version": 1, "source_fingerprint": fingerprint(one["source"]),
            "cast_fingerprint": fingerprint(one["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in one["cast"]["voices"]}})
        before_plan = make_plan(load_bundle(self.paths[0]), renderer_config())
        before_chunks = {chunk["segment"]["id"]: chunk["chunk_id"]
                         for chapter in before_plan["chapters"] for chunk in chapter["chunks"]}

        plan = plan_merge(load_bundle(self.paths[0]), None, "Keeper", "Mara")
        self.assertEqual(plan.kept["id"], self.mara["id"], "Mara has more lines and survives by default")
        self.assertIn("it has more lines", plan.text)
        self.assertIn("Later book The Return (gains Mara in place of The keeper)", plan.text)
        self.assertIn("accent cue", plan.text)
        result = apply_merge(plan, progress=lambda _: None)
        self.assertTrue(Path(result["backup_directory"]).is_dir())

        merged, registry, _ = _completed_directory(self.paths[0])
        self.assertEqual([row["name"] for row in merged["cast"]["characters"]], ["Mara"])
        mara = merged["cast"]["characters"][0]
        self.assertEqual(mara["aliases"], ["The keeper", "Keeper"])
        self.assertNotIn(keeper_voice, {voice["id"] for voice in merged["cast"]["voices"]})
        self.assertEqual({role["id"] for role in merged["cast"]["roles"]}, {"narrator_main", "mara_talk"})
        by_id = {segment["id"]: segment for segment in merged["segments"]}
        self.assertEqual((by_id["b2:1"]["role_id"], by_id["b2:1"]["voice_id"]), ("mara_talk", mara_voice))
        self.assertNotIn("simultaneous_speakers", by_id["b3:0"])
        self.assertEqual(by_id["b3:0"]["voice_id"], mara_voice)
        self.assertEqual(by_id["b4:1"]["attribution"]["candidates"], ["mara_talk"])
        record = merged["provenance"]["character_merges"][-1]
        self.assertEqual(record["role_map"], {"keeper_talk": "mara_talk"})
        self.assertEqual(sorted(record["revoiced_segments"]), ["b2:1", "b3:0"])
        narrator_voice = self.narrator_voice
        self.assertEqual(set(merged["provenance"]["voice_samples"]["scripts"]), {narrator_voice, mara_voice})
        self.assertEqual(merged["provenance"]["voice_samples"]["cast_fingerprint"], fingerprint(merged["cast"]))
        self.assertEqual(registry["books"][-1]["cast"], merged["cast"])

        later, later_registry, _ = _completed_directory(self.paths[1])
        validate_registry(later_registry)
        continuation_directories(self.paths[1])
        self.assertEqual(later_registry["books"][0]["cast"], merged["cast"])
        self.assertEqual([(row["id"], row["name"], row["voice_id"]) for row in later["cast"]["characters"]],
                         [(self.mara["id"], "Mara", mara_voice)])
        self.assertEqual(later["cast"]["characters"][0]["evidence"],
                         two["cast"]["characters"][0]["evidence"])
        carried = next(voice for voice in later["cast"]["voices"] if voice["id"] == mara_voice)
        self.assertEqual(carried["series_origin"]["kind"], "series_casting_reuse")
        self.assertEqual(next(row for row in later["segments"] if row["id"] == "b1:1")["voice_id"], mara_voice)

        self.assertEqual(self.read(self.paths[0], "voice-accents.json")["accents"],
                         {mara_voice: "light Welsh accent"})
        self.assertEqual(self.read(self.paths[1], "voice-traits.json")["speech_traits"], {})
        self.assertEqual(self.read(self.paths[0], "voice-aliases.json")["aliases"], {self.narrator_voice: mara_voice})
        auditions = self.read(self.paths[0], AUDITIONS)
        self.assertEqual(auditions["cast_fingerprint"], fingerprint(merged["cast"]))
        self.assertEqual(set(auditions["scripts"]), {narrator_voice, mara_voice})

        after_plan = make_plan(load_bundle(self.paths[0]), renderer_config())
        after_chunks = {chunk["segment"]["id"]: chunk["chunk_id"]
                        for chapter in after_plan["chapters"] for chunk in chapter["chunks"]}
        changed = {segment for segment in before_chunks if before_chunks[segment] != after_chunks[segment]}
        self.assertEqual(changed, {"b2:1", "b3:0"}, "only moved lines get new chunk identities")
        self.assertEqual(revoiced_segment_ids(merged["provenance"]), {"b2:1", "b3:0"})

    def test_a_later_book_with_both_entries_is_merged_too_and_earlier_books_are_left_apart(self):
        self.publish(later_characters=(("mara", "Mara", ()), ("keeper", "The keeper", ())),
                     later=[[("Mara and the keeper returned. ", None), ("“Back,”", "keeper"), (" ", None),
                             ("“Again,”", "mara")], [("Rain fell.", None)]])
        plan = plan_merge(load_bundle(self.paths[1]), None, self.mara["id"], self.keeper["voice_id"],
                          keep="The keeper", name="Mara")
        self.assertEqual([book.path for book in plan.books], [self.paths[1]])
        self.assertEqual(plan.earlier, ["The Lantern"])
        self.assertIn("Earlier book(s) keep Mara as a separate entry: The Lantern", plan.text)
        apply_merge(plan, progress=lambda _: None)
        later, registry, _ = _completed_directory(self.paths[1])
        self.assertEqual([(row["id"], row["name"], row["aliases"]) for row in later["cast"]["characters"]],
                         [(self.keeper["id"], "Mara", ["The keeper"])])
        self.assertEqual({row["voice_id"] for row in later["segments"] if row["kind"] == "dialogue"},
                         {self.keeper["voice_id"]})
        earlier, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(len(earlier["cast"]["characters"]), 2)
        self.assertEqual(registry["books"][0]["cast"], earlier["cast"])

    def test_a_character_from_an_earlier_book_is_always_kept_and_the_new_entry_merges_into_it(self):
        _, two = self.publish(
            later_characters=(("keeper", "The keeper", ()), ("stranger", "The stranger", ())),
            later=[[("The keeper returned. ", None), ("“Back,”", "keeper")],
                   [("A stranger spoke. ", None), ("“Hello,”", "stranger"), (" ", None), ("“Again,”", "stranger")],
                   [("“Still here,”", "stranger")]])
        stranger = next(row for row in two["cast"]["characters"] if row["name"] == "The stranger")
        bundle = load_bundle(self.paths[1])
        owned = {stranger["voice_id"]}
        store = SimpleNamespace(canonical_id=lambda voice: voice,
                                load=lambda voice: {"inherited": False} if voice in owned else None)
        plan = plan_merge(bundle, None, "The stranger", "The keeper", store=store)
        self.assertEqual((plan.kept["id"], plan.fixed), (self.keeper["id"], "The Lantern"),
                         "inherited beats more lines and an accepted voice")
        self.assertIn("it comes from an earlier book (The Lantern), so it must be kept", plan.text)
        with self.assertRaisesRegex(ValueError, "comes from an earlier book .* so it must be kept"):
            plan_merge(bundle, None, "The stranger", "The keeper", keep="The stranger", store=store)
        candidate = merge_document(two, receipt(stranger["id"], self.keeper["id"], "The stranger"))
        with self.assertRaisesRegex(ValueError, "a character from an earlier book must be kept"):
            revise_directories({self.paths[1]: candidate}, directories=[self.paths[1]], scope="character-merge",
                               progress=lambda _: None)
        output = StringIO()
        args = SimpleNamespace(first="The stranger", second="The keeper", keep=None, name="The stranger", yes=False,
                               dry_run=False)
        self.assertEqual(run_merge_command(bundle, None, args, input_stream=Terminal("s\ny\n"),
                                           output_stream=output), 0)
        self.assertIn("so it is the one kept", output.getvalue())
        self.assertNotIn("[s]wap", output.getvalue())
        merged, registry, _ = _completed_directory(self.paths[1])
        self.assertEqual([(row["id"], row["name"], row["aliases"]) for row in merged["cast"]["characters"]],
                         [(self.keeper["id"], "The stranger", ["The keeper"])])
        validate_registry(registry)

    def test_the_entry_with_an_accepted_voice_survives_by_default_and_the_person_may_choose(self):
        self.publish()
        bundle = load_bundle(self.paths[0])
        owned = {self.keeper["voice_id"]}
        store = SimpleNamespace(canonical_id=lambda voice: voice,
                                load=lambda voice: {"inherited": False} if voice in owned else None)
        plan = plan_merge(bundle, None, "Mara", "Keeper", store=store)
        self.assertEqual(plan.kept["id"], self.keeper["id"])
        self.assertIn("it already has an accepted voice", plan.text)
        chosen = plan_merge(bundle, None, "Mara", "Keeper", keep="Mara", store=store)
        self.assertEqual((chosen.kept["id"], chosen.reason), (self.mara["id"], "you chose it"))
        self.assertIn("The keeper's accepted recording will no longer be used.", chosen.text)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            plan_merge(bundle, None, "Mara", "Keeper", name="Somebody Else")
        with self.assertRaisesRegex(ValueError, "same cast entry"):
            plan_merge(bundle, None, "Keeper", "The keeper")

    def test_command_shows_the_change_and_saves_only_after_confirmation(self):
        self.publish()
        bundle = load_bundle(self.paths[0])
        original = (self.paths[0] / "analysis.json").read_bytes()
        args = SimpleNamespace(first="Mara", second="Keeper", keep=None, name=None, yes=False, dry_run=False)
        output = StringIO()
        self.assertEqual(run_merge_command(bundle, None, args, input_stream=StringIO("y\n"), output_stream=output), 0)
        self.assertIn("Nothing was changed", output.getvalue())
        self.assertEqual(run_merge_command(bundle, None, args, input_stream=Terminal("n\n"),
                                           output_stream=StringIO()), 0)
        self.assertEqual((self.paths[0] / "analysis.json").read_bytes(), original)
        output = StringIO()
        self.assertEqual(run_merge_command(bundle, None, args, input_stream=Terminal("s\nc\n2\ny\n"),
                                           output_stream=output), 0)
        self.assertIn("Merged Mara into Keeper", output.getvalue())
        merged, _, _ = _completed_directory(self.paths[0])
        self.assertEqual([(row["id"], row["name"]) for row in merged["cast"]["characters"]],
                         [(self.keeper["id"], "Keeper")])
        self.assertEqual(merged["cast"]["characters"][0]["aliases"], ["The keeper", "Mara"])

    def test_two_entries_with_the_same_name_are_told_apart_and_merged_by_id(self):
        document = build("same-name", "Twins", [("mara", "Mara", ()), ("mara2", "Mara", ())],
                         [[("Mara waved. ", None), ("“Hello there,”", "mara")],
                          [("Mara frowned. ", None), ("“Not you again,”", "mara2")],
                          [("“Again,”", "mara2")]])
        document["cast"]["characters"][1]["description"] = "The lighthouse keeper's sister."
        path = self.root / "twins"
        linked = link_analysis(document, fresh_registry(path))
        publish_directory(linked, fresh_registry(path), path, export_bundle)
        first, second = linked["cast"]["characters"]
        bundle = load_bundle(path)
        with self.assertRaisesRegex(ValueError, "names several characters"):
            plan_merge(bundle, None, "Mara", first["id"])
        plan = plan_merge(bundle, None, first["voice_id"], second["id"])
        self.assertEqual(plan.kept["id"], second["id"], "the entry with more lines is kept")
        self.assertIn("The lighthouse keeper's sister.", plan.text)
        self.assertIn("first line: “Not you again,”", plan.text)
        self.assertIn("first line: “Hello there,”", plan.text)
        apply_merge(plan, progress=lambda _: None)
        merged, _, _ = _completed_directory(path)
        self.assertEqual([(row["id"], row["name"], row["aliases"]) for row in merged["cast"]["characters"]],
                         [(second["id"], "Mara", [])])
        self.assertEqual({row["voice_id"] for row in merged["segments"] if row["kind"] == "dialogue"},
                         {second["voice_id"]})

    def test_renderer_cli_dry_run_shows_the_merge_and_changes_nothing(self):
        from contextlib import redirect_stdout
        from epub_tts_narrator.renderer.cli import main
        self.publish()
        config = self.root / "settings.toml"
        config.write_text("", encoding="utf-8")
        before = (self.paths[0] / "analysis.json").read_bytes()
        output = StringIO()
        with redirect_stdout(output):
            status = main(["merge-characters", str(self.paths[0]), "Mara", "Keeper", "--dry-run",
                           "--config", str(config)])
        self.assertEqual(status, 0)
        self.assertIn("Absorb: The keeper", output.getvalue())
        self.assertIn("Dry run: nothing was changed.", output.getvalue())
        self.assertEqual((self.paths[0] / "analysis.json").read_bytes(), before)

    def test_publication_refuses_a_candidate_that_differs_from_its_recorded_merge(self):
        one, _ = self.publish()
        candidate = merge_document(one, receipt(self.mara["id"], self.keeper["id"], "Mara"))
        candidate["segments"][0]["delivery"]["direction"] = "Sneaked in."
        before = (self.paths[0] / "analysis.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "differs from its recorded merge: segments"):
            revise_directories({self.paths[0]: candidate}, directories=self.paths, scope="character-merge",
                               progress=lambda _: None)
        with self.assertRaisesRegex(ValueError, "exactly one merge record"):
            revise_directories({self.paths[0]: deepcopy(one)}, directories=self.paths, scope="character-merge",
                               progress=lambda _: None)
        self.assertEqual((self.paths[0] / "analysis.json").read_bytes(), before)

    def test_unfinished_speaker_decisions_are_restated_for_the_merged_cast(self):
        one, _ = self.publish()
        cache = analysis_cache_directory(self.paths[0])
        segment = next(row for row in one["segments"] if row["id"] == "b4:1")
        decision = {"segment_id": "b4:1", "block_id": "b4", "start": segment["start"], "end": segment["end"],
                    "text_fingerprint": fingerprint(segment["text"]), "selected_roles": ["keeper_talk"],
                    "selected_voices": [self.keeper["voice_id"]],
                    "previous_attribution": deepcopy(segment["attribution"]), "previous_kind": "dialogue",
                    "kind": "dialogue", "simultaneous": False}
        _save_decision_checkpoint(cache, one, [decision], "interrupted")
        apply_merge(plan_merge(load_bundle(self.paths[0]), None, "Mara", "Keeper"), progress=lambda _: None)
        merged, _, _ = _completed_directory(self.paths[0])
        journal = _load_decision_checkpoint(cache, decision_checkpoint_key(merged))
        self.assertEqual(journal["decisions"][0]["selected_roles"], ["mara_talk"])
        self.assertEqual(journal["decisions"][0]["previous_attribution"]["candidates"], ["mara_talk"])
        resumed = prompt_speaker_decisions(merged, input_stream=Terminal(""), output_stream=StringIO(),
                                           decisions_cache=cache)
        replayed = next(row for row in resumed["segments"] if row["id"] == "b4:1")
        self.assertEqual((replayed["role_id"], replayed["attribution"]["basis"]), ("mara_talk", "user"))

    def test_merge_maps_simultaneous_groups_changed_voices_and_issue_targets(self):
        document = build("solo", "Solo", [("a", "Ann", ()), ("b", "Bea", ()), ("c", "Cy", ())],
                         [[("“All three!”", ("a", "b", "c"))], [("“Both!”", ("a", "b"))], [("Calm.", None)]])
        document["cast"]["voices"].append({"id": "v_b_young", "description": "Bea, young.", "traits": [],
                                           "derived_from": {"voice_id": "v_b", "kind": "age",
                                                            "instruction": "Younger and brighter.",
                                                            "evidence": [{"block_id": "b3", "quote": "Calm"}]}})
        document["cast"]["roles"].append({"id": "b_young", "kind": "dialogue", "character_id": "b",
                                          "voice_id": "v_b_young", "description": "Bea as a child."})
        document["issues"].append({"id": "i1", "kind": "casting", "target": "v_b", "reason": "Check.",
                                   "resolved": False})
        self.assertEqual(validate_document(document), [])
        merged = merge_document(document, receipt("a", "b", "Bea"))
        self.assertEqual(merged["segments"][0]["simultaneous_speakers"],
                         [{"role_id": "a_talk", "voice_id": "v_a"}, {"role_id": "c_talk", "voice_id": "v_c"}])
        self.assertEqual((merged["segments"][1]["role_id"], merged["segments"][1]["voice_id"]), ("a_talk", "v_a"))
        young = next(voice for voice in merged["cast"]["voices"] if voice["id"] == "v_b_young")
        self.assertEqual(young["derived_from"]["voice_id"], "v_a")
        self.assertEqual(next(role for role in merged["cast"]["roles"] if role["id"] == "b_young")["character_id"], "a")
        self.assertEqual(merged["issues"][0]["target"], "v_a")
        self.assertEqual((merged["cast"]["characters"][0]["name"], merged["cast"]["characters"][0]["aliases"]),
                         ("Bea", ["Ann"]))
        self.assertEqual(resolve_character(merged, "ann")["id"], "a")
        decisions = remap_decisions([{"selected_roles": ["b_talk", "a_talk"], "selected_voices": ["v_b", "v_a"],
                                      "simultaneous": True, "previous_attribution": {"candidates": ["b_talk"]}}],
                                    merged)
        self.assertEqual((decisions[0]["selected_roles"], decisions[0]["simultaneous"]), (["a_talk"], False))
        with self.assertRaisesRegex(ValueError, "names several characters"):
            resolve_character(build("x", "X", [("a", "Sam", ()), ("b", "Sam", ())], [[("Hi.", None)]]), "Sam")


class CastMergeActionTests(unittest.TestCase):
    def controller(self, *, active=False):
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        bundle = {"cast": {"characters": [{"id": "c1", "name": "Ann", "voice_id": "v1", "description": "A sailor."},
                                          {"id": "c2", "name": "Ann", "voice_id": "v2",
                                           "description": "The innkeeper's daughter."}]},
                  "chapters": [{"rows": [{"voice_id": "v2"}, {"voice_id": "vn"},
                                         {"voice_id": None, "simultaneous_speakers": [{"voice_id": "v1"},
                                                                                      {"voice_id": "v2"}]}]}]}
        session = SimpleNamespace(bundle=bundle, config={}, store=None, rows=lambda: [
            {"voice_id": "vn", "label": "Narrator", "characters": [], "candidates": [], "selected": None},
            {"voice_id": "v1", "label": "Ann", "characters": ["Ann"], "candidates": [], "selected": None},
            {"voice_id": "v2", "label": "Ann", "characters": ["Ann"], "candidates": [], "selected": None}])
        controller = CastController(session, SimpleNamespace(stop=lambda: None),
                                    describe_error=lambda error: str(error))
        self.addCleanup(controller.close)
        controller.refresh()
        return controller

    def test_only_characters_own_voices_are_offered_and_a_merge_waits_for_jobs(self):
        controller = self.controller()
        self.assertEqual(controller.merge_options(controller.row("v1")),
                         [("v2", "Ann — missing, 2 line(s) · The innkeeper's daughter.  (v2)")])
        with self.assertRaisesRegex(ValueError, "not a character's own voice"):
            controller.merge_options(controller.row("vn"))
        import threading
        release = threading.Event()
        controller.jobs.submit("Busy", release.wait)
        self.addCleanup(release.set)
        with self.assertRaisesRegex(ValueError, "Wait for the running and queued jobs"):
            controller.merge(SimpleNamespace(absorbed={"name": "Bea", "voice_id": "v2"}, name="Ann",
                                             kept={"voice_id": "v1"}))


def highlighted(table):
    from epub_tts_narrator.tui_widgets import highlighted_key
    return highlighted_key(table)


class CastColumnFitTests(unittest.TestCase):
    def test_columns_take_what_they_need_when_it_fits_and_share_the_room_when_it_does_not(self):
        from epub_tts_narrator.renderer.cast_tui import MIN_COLUMN, fit_columns
        self.assertEqual(fit_columns(120, ["Ann", "Belgarath"], ["accepted"], ["", "uses v-ann"]), (9, 10))
        voice, note = fit_columns(60, ["x" * 80], ["accepted"], ["y" * 50])
        self.assertEqual(voice + note, 60 - 8 - len("accepted"))
        self.assertGreater(voice, note, "the name keeps most of the room")
        self.assertEqual(fit_columns(60, ["x" * 80], ["accepted"], [""]), (40, 4),
                         "an empty note column leaves the name the rest")
        self.assertEqual(fit_columns(10, ["x" * 80], ["needs regeneration"], ["y" * 50]), (MIN_COLUMN, MIN_COLUMN))


class CastScreenMergeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cast_table_columns_fit_the_screen_width_and_refit_when_it_changes(self):
        import asyncio
        from cast_tui_fixtures import FakePlayer, FakeSession, voice_row
        from epub_tts_narrator.renderer.cast_tui import CastApp
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        from textual.widgets import DataTable
        asyncio.get_running_loop().slow_callback_duration = 10
        long_name = "Belgarath (condition: Weakened, rushed, unfocused from exhaustion, barely awake)"
        session = FakeSession([voice_row("v-bel", "Belgarath"), voice_row("v-ann", "Ann"),
                               voice_row("v-bel-tired", long_name,
                                         derived={"voice_id": "v-bel", "kind": "condition", "instruction": "x"})])
        controller = CastController(session, FakePlayer(), describe_error=str)
        self.addCleanup(controller.close)
        app = CastApp(controller, title="Fixture Book", startup=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            table = app.query_one("#voices", DataTable)
            table.focus()
            table.move_cursor(row=2)
            await pilot.pause()
            shown = str(table.get_row("v-bel-tired")[0])
            self.assertTrue(shown.endswith("…") and long_name.startswith(shown[:-1]))
            self.assertLessEqual(sum(column.get_render_width(table) for column in table.ordered_columns),
                                 table.size.width, "the columns fit inside the table")
            self.assertIn(long_name, str(app.query_one("#detail").render()))
            await pilot.resize_terminal(300, 40)
            await pilot.pause(0.7)
            self.assertEqual(str(table.get_row("v-bel-tired")[0]), long_name, "a wide screen shows the name whole")
            self.assertEqual(highlighted(table), "v-bel-tired", "refitting keeps the highlighted row")
            await pilot.resize_terminal(100, 40)
            await pilot.pause(0.7)
            self.assertTrue(str(table.get_row("v-bel-tired")[0]).endswith("…"))
            self.assertLessEqual(sum(column.get_render_width(table) for column in table.ordered_columns),
                                 table.size.width)

    async def test_j_picks_the_same_character_and_its_survivor_confirms_and_reopens_the_cast(self):
        plans, apply = await self.merge_in_cast_screen(fixed="")
        self.assertEqual(plans, [("v-ava", "v-ben", None), ("v-ava", "v-ben", "v-ben")])
        self.assertEqual(apply.call_args.args[0].kept["name"], "Ben")

    async def test_a_character_from_an_earlier_book_is_kept_without_offering_a_choice(self):
        plans, apply = await self.merge_in_cast_screen(fixed="Book One")
        self.assertEqual(plans, [("v-ava", "v-ben", None)])
        self.assertEqual(apply.call_args.args[0].kept["name"], "Ava")

    async def merge_in_cast_screen(self, *, fixed):
        import asyncio
        from unittest.mock import patch
        from cast_tui_fixtures import FakePlayer, FakeSession
        from epub_tts_narrator.renderer.cast_tui import CastApp
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        from epub_tts_narrator.tui_widgets import ChoiceScreen, ConfirmScreen
        from textual.widgets import DataTable
        asyncio.get_running_loop().slow_callback_duration = 10
        session = FakeSession()
        session.store = None
        session.bundle = {"cast": {"characters": [{"id": "c-ava", "name": "Ava", "voice_id": "v-ava"},
                                                  {"id": "c-ben", "name": "Ben", "voice_id": "v-ben"}]}}
        controller = CastController(session, FakePlayer(), describe_error=str)
        self.addCleanup(controller.close)
        plans = []

        def merge_plan(row, other, keep=None):
            plans.append((row["voice_id"], other["voice_id"], keep and keep["voice_id"]))
            kept, absorbed = ({"voice_id": "v-ben", "name": "Ben"}, {"voice_id": "v-ava", "name": "Ava"}) \
                if keep else ({"voice_id": "v-ava", "name": "Ava"}, {"voice_id": "v-ben", "name": "Ben"})
            return SimpleNamespace(kept=kept, absorbed=absorbed, name=kept["name"], reason="it has more lines",
                                   kept_voice="accepted voice", absorbed_voice="no accepted voice yet",
                                   kept_lines=3, absorbed_lines=1, fixed=fixed,
                                   text=f"Keep {kept['name']}, absorb {absorbed['name']}.")
        controller.merge_plan = merge_plan
        app = CastApp(controller, title="Fixture Book", startup=False)
        with patch("epub_tts_narrator.renderer.character_merge_command.apply_merge") as apply:
            async with app.run_test(size=(160, 45)) as pilot:
                table = app.query_one("#voices", DataTable)
                table.focus()
                await pilot.pause()
                await pilot.press("j")
                self.assertIsInstance(app.screen, ChoiceScreen)
                await pilot.press("B", "e", "n", "enter")
                if not fixed:
                    self.assertIsInstance(app.screen, ChoiceScreen)
                    await pilot.press("down", "enter")
                self.assertIsInstance(app.screen, ConfirmScreen)
                await pilot.press("y")
                controller.jobs.wait_idle(5)
                await pilot.pause(1)
        self.assertTrue(controller.reopen)
        return plans, apply


if __name__ == "__main__":
    unittest.main()
