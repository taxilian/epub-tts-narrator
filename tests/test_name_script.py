import unittest


class NameScriptGrammarTests(unittest.TestCase):
    def test_unicode_offsets_and_literal_markers_roundtrip(self):
        from epub_tts_narrator.name_script import escape_text, parse_text
        names = {"N17": {"character_id": "ally", "aliases": ["Ally"]}}
        source = "é \\ (N1:Ally) Ally greeted her ally."
        prefix, suffix = source.rsplit("Ally", 1)
        annotated = escape_text(prefix) + "(N17:Ally)" + escape_text(suffix)
        self.assertEqual(parse_text(annotated, source, names), [
            {"start": len(prefix), "end": len(prefix) + 4, "name_id": "N17"}])

    def test_invalid_or_nonalias_markers_are_rejected(self):
        from epub_tts_narrator.name_script import parse_text
        names = {"N1": {"character_id": "ally", "aliases": ["Ally"]}}
        for annotated in ["(N2:Ally)", "(N01:Ally)", "(N1:(N1:Ally))", "(N1:Ally", "(N1:friend)"]:
            with self.subTest(annotated=annotated), self.assertRaises(ValueError):
                parse_text(annotated, "Ally", names)

    def test_source_literal_cannot_be_injected_as_a_name(self):
        from epub_tts_narrator.name_script import parse_text
        with self.assertRaises(ValueError):
            parse_text("(N1:Ally)", "(N1:Ally)", {"N1": {"aliases": ["Ally"]}})
