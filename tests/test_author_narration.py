"""Author casting follows exact authorship without merging story narrators."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.narration_identity import author_narrator_name
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.series import link_analysis, load_registry, save_registry, series_context
from test_series import fixture


def author_role(**changes):
    return {"id": "author_afterword", "kind": "narration", "character_id": None,
            "voice_id": "va", "description": "Author afterword and acknowledgments.", **changes}


def author_document(number=1, *, authors=None, description=None, editor=False):
    document = fixture(sequel=number > 1)
    document["source"]["book"].update(id=f"book{number}", title=f"Book {number}",
                                      authors=["Nathan Thompson"] if authors is None else authors)
    role = author_role(**({"description": description} if description is not None else {}))
    document["cast"]["roles"].append(role)
    document["cast"]["voices"].append({"id": "va", "description": "A conversational author reading voice.",
                                        "traits": [{"name": "pace", "value": "relaxed", "basis": "casting", "evidence": []}]})
    document["segments"][-1].update(role_id=role["id"], voice_id="va")
    if editor:
        document["cast"]["roles"].append(author_role(id="editorial_asides", description="Editorial margin-note commentary; author/editor asides."))
    return document


def role_named(document, role_id="author_afterword"):
    return next(role for role in document["cast"]["roles"] if role["id"] == role_id)


class AuthorNarrationIdentityTests(unittest.TestCase):
    def test_author_scopes_need_unambiguous_book_authorship(self):
        descriptions = [
            "Author afterword and acknowledgments (ch0014) and front/back matter; nonstory text distinct from the main narrative frame.",
            "Author afterword narration by Nathan Thompson: the post-story address, acknowledgments, and promotional links; a distinct narration scope from the story frame.",
            "Author afterword and acknowledgments by Nathan Thompson; a distinct narration scope separate from the story frame.",
            "Author back-matter narration: the afterword, acknowledgments, and the Advancement stages appendix, read in the author's own voice rather than the story narrator's.",
            "Author narration for the copyright page, dedication, the Advancement Stages and Substages appendix, and the afterword; nonstory text outside Jasper's narrative frame.",
            "Author afterword narration scope for the book's closing chapter; covers the author's direct address, acknowledgments, and promotional notes. Distinct from Jasper's first-person narrative frame, which remains narrator_main.",
            "The author's direct address to the reader.",
            "The afterword by the author.",
            "From the Author",
            "A note from the author.",
            "Nonstory From the Author section, separate from the story narrator.",
        ]
        for description in descriptions:
            with self.subTest(description=description):
                self.assertEqual(author_narrator_name(author_role(description=description), {"authors": ["Nathan Thompson"]}), "Nathan Thompson")
        for authors in ([], ["Nathan Thompson", "Another Author"], ["Nathan Thompson and Another Author"],
                        ["Nathan Thompson & Another Author"], ["Anonymous"], ["Unknown"], [""], [None], "Nathan Thompson"):
            with self.subTest(authors=authors):
                self.assertIsNone(author_narrator_name(author_role(), {"authors": authors}))

    def test_generic_guest_editor_character_and_main_story_roles_are_not_author_roles(self):
        descriptions = ["The book's narrator.", "Omniscient authorial narrator.", "Guest foreword narrator.",
                        "Author afterword read by a guest contributor.", "Editorial margin-note commentary; author/editor asides.",
                        "Generic narration of front matter.", "The narrator directly addresses the reader.",
                        "Author afterword by Someone Else.", "Author foreword narration by Jane Thompson.",
                        "Author afterword by Nathan Thompson Jr.", "This is not author narration.",
                        "Non-author narration of an afterword.", "From the Author, read by a guest."]
        for description in descriptions:
            with self.subTest(description=description):
                self.assertIsNone(author_narrator_name(author_role(description=description), {"authors": ["Nathan Thompson"]}))
        for changes in ({"character_id": "fictional-author"}, {"kind": "dialogue"}, {"id": "narrator_main"}):
            self.assertIsNone(author_narrator_name(author_role(**changes), {"authors": ["Nathan Thompson"]}))


class AuthorNarrationContinuityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "series.json"
        self.empty = load_registry(self.path)

    def save(self, document, registry=None):
        registry = load_registry(self.path) if registry is None else registry
        linked = link_analysis(document, registry)
        save_registry(self.path, registry, linked)
        return linked, load_registry(self.path)

    def test_same_author_reuses_voice_and_source_qualified_origin_through_third_book(self):
        first, registry = self.save(author_document())
        original = author_document(2, authors=["  NATHAN  THOMPSON "])
        second, registry = self.save(original, registry)
        third = link_analysis(author_document(3), registry)
        voice = role_named(first)["voice_id"]
        self.assertEqual(role_named(second)["voice_id"], voice)
        self.assertEqual(role_named(third)["voice_id"], voice)
        self.assertNotEqual(voice, role_named(second, "narrator_main")["voice_id"])
        self.assertNotEqual(voice, second["cast"]["characters"][0]["voice_id"])
        self.assertEqual(second["source"], original["source"])
        self.assertEqual(second["segments"][-1]["voice_id"], voice)
        origin = role_named(third)["series_origin"]
        self.assertEqual(origin["basis"], "author_narration_casting_continuity")
        self.assertEqual(origin["author_name"], "Nathan Thompson")
        self.assertEqual(origin["source_fingerprint"], fingerprint(second["source"]))
        self.assertEqual(origin["book"], second["source"]["book"])
        self.assertEqual(validate_document(third), [])
        self.assertEqual(link_analysis(third, registry), third)

    def test_different_missing_ambiguous_and_similar_author_names_do_not_reuse(self):
        first, registry = self.save(author_document())
        for authors in (["Nathan Thomson"], ["N. Thompson"], ["Another Author"], [],
                        ["Nathan Thompson", "Another Author"], ["Anonymous"]):
            with self.subTest(authors=authors):
                linked = link_analysis(author_document(2, authors=authors), registry)
                self.assertNotEqual(role_named(linked)["voice_id"], role_named(first)["voice_id"])
                self.assertNotIn("series_origin", role_named(linked))

    def test_nearest_legacy_author_voice_wins_over_earlier_unique_voices(self):
        first, registry = self.save(author_document())
        # Simulate a book produced before author continuity existed.
        second = link_analysis(author_document(2), self.empty)
        save_registry(self.path, registry, second)
        registry = load_registry(self.path)
        third = link_analysis(author_document(3), registry)
        self.assertNotEqual(role_named(first)["voice_id"], role_named(second)["voice_id"])
        self.assertEqual(role_named(third)["voice_id"], role_named(second)["voice_id"])
        context = series_context(registry, third["source"])["author_narrator_voices"]
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["voice_ids"], [role_named(second)["voice_id"]])
        self.assertEqual(context[0]["source_fingerprint"], fingerprint(second["source"]))

    def test_nearest_profile_for_same_voice_is_used_and_prior_records_are_unchanged(self):
        _, registry = self.save(author_document())
        second = link_analysis(author_document(2), registry)
        identity = role_named(second)["voice_id"]
        next(voice for voice in second["cast"]["voices"] if voice["id"] == identity)["description"] = "Updated author performance."
        save_registry(self.path, registry, second)
        registry = load_registry(self.path)
        before = deepcopy(registry)
        third = link_analysis(author_document(3), registry)
        voice = next(voice for voice in third["cast"]["voices"] if voice["id"] == identity)
        self.assertEqual(voice["description"], "Updated author performance.")
        self.assertEqual(voice["series_origin"]["source_fingerprint"], fingerprint(second["source"]))
        self.assertEqual(registry, before)

    def test_shared_author_editor_roles_keep_one_voice_and_consistent_voice_map(self):
        first, registry = self.save(author_document())
        second = link_analysis(author_document(2, editor=True), registry)
        identity = role_named(first)["voice_id"]
        self.assertEqual(role_named(second)["voice_id"], identity)
        self.assertEqual(role_named(second, "editorial_asides")["voice_id"], identity)
        self.assertEqual(second["provenance"]["series"]["voice_id_map"]["va"], identity)
        self.assertEqual(sum(voice["id"] == identity for voice in second["cast"]["voices"]), 1)
        self.assertEqual(validate_document(second), [])

    def test_ambiguous_nearest_author_cast_does_not_fall_back_to_older_voice(self):
        first, registry = self.save(author_document())
        legacy = author_document(2)
        legacy["cast"]["roles"].append(author_role(id="author_note", voice_id="vb", description="Author notes."))
        legacy["cast"]["voices"].append({"id": "vb", "description": "A different author reading.", "traits": []})
        second = link_analysis(legacy, self.empty)
        save_registry(self.path, registry, second)
        registry = load_registry(self.path)
        third = link_analysis(author_document(3), registry)
        prior = {role_named(first)["voice_id"], role_named(second)["voice_id"], role_named(second, "author_note")["voice_id"]}
        self.assertNotIn(role_named(third)["voice_id"], prior)
        self.assertEqual(len(series_context(registry, third["source"])["author_narrator_voices"][0]["voice_ids"]), 2)

    def test_author_role_does_not_override_explicit_shared_story_or_character_voice(self):
        first, registry = self.save(author_document())
        for identity in ("vn", "v1"):
            current = author_document(2)
            role_named(current)["voice_id"] = identity
            current["segments"][-1]["voice_id"] = identity
            linked = link_analysis(current, registry)
            with self.subTest(shared=identity):
                self.assertNotEqual(role_named(linked)["voice_id"], role_named(first)["voice_id"])
                self.assertNotIn("author_narration", role_named(linked).get("series_origin", {}).get("basis", ""))
                if identity == "v1":
                    self.assertEqual(role_named(linked)["voice_id"], role_named(linked, "mara")["voice_id"])


if __name__ == "__main__":
    unittest.main()
