"""Analysis proposes groups and their line cues from speaker tags; names alone never decide."""

import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.group_cues import group_warnings
from epub_tts_narrator.group_proposals import propose_groups
from test_character_merge import build


def proposed(characters, paragraphs):
    document = build("groups", "Groups", characters, paragraphs)
    candidate, changes = propose_groups(document)
    groups = {character["id"]: character["group"] for character in candidate["cast"]["characters"]
              if "group" in character}
    lines = {row["id"]: {key: value for key, value in row["ensemble"].items() if key != "evidence"}
             for row in candidate["segments"] if "ensemble" in row}
    return candidate, groups, lines, changes


class GroupProposalTests(unittest.TestCase):
    def test_a_group_whose_tags_say_several_people_speak_is_proposed_with_the_tag_as_evidence(self):
        candidate, groups, _, _ = proposed([("farmhands", "seven masked farmhands", ())], [
            [("“We are the Gods,” ", "farmhands"), ("they said in unison.", None)],
            [("“Bless this house,” ", "farmhands"), ("the seven responded.", None)]])
        self.assertEqual(validate_document(candidate), [])
        group = groups["farmhands"]
        self.assertEqual((group["size"], group["scale"], group["proposed_voices"], group["proposed_mode"]),
                         (7, "many", 7, "unison"))
        self.assertEqual(group["basis"], "analysis")
        self.assertEqual(group["evidence"][0]["quote"], "they said in unison.")

    def test_anonymous_members_get_pool_slots_and_the_same_individual_keeps_hers(self):
        _, groups, lines, _ = proposed([("dryads", "Dryad girls", ())], [
            [("“It looks healthy,” ", "dryads"), ("one of the others observed.", None)],
            [("“Let us keep it,” ", "dryads"), ("another one said.", None)],
            [("“Kisses,” ", "dryads"), ("the Dryad offered.", None)],
            [("“Eight,” ", "dryads"), ("someone countered.", None)]])
        self.assertEqual(groups["dryads"]["proposed_mode"], "pool")
        self.assertEqual([lines[key].get("slot") for key in ("b1:0", "b2:0", "b3:0", "b4:0")], [1, 2, 2, 3])

    def test_a_crowd_and_a_part_of_a_group_are_sized_from_the_tags(self):
        _, groups, lines, _ = proposed([("legion", "Tolnedran legionnaires", ())], [
            [("“From Cthol Murgos,” ", "legion"), ("several of them answered her.", None)],
            [("“Ce-Ne-dra!” ", "legion"), ("the legions chanted together.", None)]])
        # "the legions" make it a crowd; the line "several of them" answer takes three of its voices.
        self.assertEqual((groups["legion"]["scale"], groups["legion"]["proposed_voices"]), ("crowd", 8))
        self.assertEqual(lines["b1:0"]["lanes"], 3)

    def test_one_persons_lines_and_a_named_twin_are_never_proposed(self):
        _, groups, _, _ = proposed([("farmer", "stout farmer", ()), ("beltira", "Beltira", ()),
                                    ("rabbas", "Rabbas", ())], [
            [("“Good day,” ", "farmer"), ("the farmer said.", None)],
            [("“Come in,” ", "farmer"), ("the farmer said.", None)],
            [("“The Guide,” ", "beltira"), ("the twins said in unison.", None)],
            [("“The Bear,” ", "beltira"), ("they said together.", None)],
            [("“Never,” ", "rabbas"), ("Rabbas protested.", None)]])
        self.assertEqual(groups, {})

    def test_a_one_person_name_needs_repeated_plural_tags(self):
        _, groups, _, _ = proposed([("nobles", "Arendish noble", ())], [
            [("“Who rules us?” ", "nobles"), ("the nobles asked.", None)]])
        self.assertEqual(groups, {}, "one plural tag on a one-person name stays a flag")
        _, groups, _, _ = proposed([("nobles", "Arendish noble", ())], [
            [("“Who rules us?” ", "nobles"), ("the nobles asked.", None)],
            [("“Korodullin,” ", "nobles"), ("they replied.", None)]])
        self.assertIn("nobles", groups)

    def test_a_band_named_after_its_leader_is_proposed_from_one_plural_tag(self):
        # "band", "troop", "retinue" name several people; "those behind Mara" is several of them speaking.
        _, groups, _, _ = proposed([("mara", "Mara", ()), ("band", "Mara's band", ())], [
            [("“The Hills!” ", "band"), ("those behind Mara exclaimed.", None)],
            [("“To war! To war!”", "band")],
            [("“Hail,” ", "mara"), ("Mara said.", None)]])
        self.assertEqual(set(groups), {"band"})
        self.assertEqual(groups["band"]["evidence"][0]["quote"], "those behind Mara exclaimed.")

    def test_a_person_or_an_earlier_proposal_is_never_overwritten_and_it_is_idempotent(self):
        document = build("groups", "Groups", [("chorus", "eunuch chorus", ())],
                         [[("“Welcome,” ", "chorus"), ("the chorus replied.", None)]])
        document["cast"]["characters"][0]["group"] = {"size": None, "scale": None, "members": [], "basis": "user"}
        candidate, changes = propose_groups(document)
        self.assertEqual((candidate, changes), (document, []))
        fresh = build("groups", "Groups", [("chorus", "eunuch chorus", ())],
                      [[("“Welcome,” ", "chorus"), ("the chorus replied.", None)]])
        once, _ = propose_groups(fresh)
        twice, changes = propose_groups(once)
        self.assertEqual((twice, changes), (once, []))

    def test_a_proposed_groups_plural_tags_are_no_longer_flags(self):
        candidate, _, _, _ = proposed([("chorus", "eunuch chorus", ())],
                                      [[("“Welcome,” ", "chorus"), ("the chorus replied.", None)]])
        self.assertEqual(group_warnings(candidate), [])


class SavedBookProposalTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from copy import deepcopy
        from pathlib import Path
        from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
        from epub_tts_narrator.cli import export_bundle
        from epub_tts_narrator.pipeline import fingerprint
        from epub_tts_narrator.series import link_analysis
        from test_character_merge import AUDITION
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "book"
        document = build("chorus-book", "The Chorus", [("mara", "Mara", ()), ("chorus", "eunuch chorus", ())], [
            [("Mara waved. ", None), ("“Hello,” ", "mara"), ("she said.", None)],
            [("“Welcome,” ", "chorus"), ("the chorus replied in unison.", None)]])
        linked = link_analysis(document, fresh_registry(self.path))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(self.path), self.path, export_bundle)

    def test_rerunning_analyze_proposes_the_books_groups_once(self):
        from epub_tts_narrator.analysis_directory import _completed_directory
        from epub_tts_narrator.group_proposals import review_groups
        said = []
        self.assertTrue(review_groups(self.path, said.append))
        document = _completed_directory(self.path)[0]
        chorus = next(row for row in document["cast"]["characters"] if row["name"] == "eunuch chorus")
        self.assertEqual((chorus["group"]["basis"], chorus["group"]["proposed_mode"]), ("analysis", "unison"))
        self.assertTrue(any("cast screen (C)" in line for line in said))
        self.assertFalse(review_groups(self.path, said.append), "nothing more to propose")

    def test_the_proposal_scope_accepts_nothing_but_proposals(self):
        from epub_tts_narrator.analysis_directory import _completed_directory
        from epub_tts_narrator.analysis_revision import revise_directories
        document = _completed_directory(self.path)[0]
        candidate, _ = propose_groups(document)
        candidate["cast"]["voices"][0]["description"] = "Changed."
        with self.assertRaisesRegex(ValueError, "preserve"):
            revise_directories({self.path: candidate}, scope="group-proposals", progress=lambda _: None)


if __name__ == "__main__":
    unittest.main()
