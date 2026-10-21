"""Human decisions happen after bounded automatic work and survive retries."""

from copy import deepcopy
from contextlib import nullcontext
from io import StringIO
import os
from pathlib import Path
from unittest.mock import call, patch
import unittest

from epub_audiobook.contracts import readiness, validate_document
from epub_audiobook.pipeline import fingerprint
from epub_audiobook.speaker_completion import complete_speakers, unassigned_segments
from epub_audiobook.speaker_decisions import prompt_speaker_decisions
from test_contracts import valid_document


class Terminal(StringIO):
    def isatty(self):
        return True


def missing(indices=(1,)):
    document = valid_document()
    for index in indices:
        document["segments"][index].update(role_id=None, voice_id=None)
        document["segments"][index]["attribution"].update(basis="unresolved", review_required=True,
                                                          reason="No consensus.", candidates=["mara"])
    return document


def add_lena(document):
    document["cast"]["characters"].append({"id": "c2", "name": "Lena", "aliases": [], "description": "Another speaker.", "evidence": [], "voice_id": "v2"})
    document["cast"]["voices"].append({"id": "v2", "description": "Second voice.", "traits": []})
    document["cast"]["roles"].append({"id": "lena", "kind": "dialogue", "voice_id": "v2", "character_id": "c2", "description": "Lena speaking."})


class SpeakerDecisionTests(unittest.TestCase):
    def decide(self, document, answers):
        output = StringIO()
        result = prompt_speaker_decisions(document, book_path=Path("books/A Book"),
                                          input_stream=Terminal(answers), output_stream=output)
        self.assertEqual(validate_document(result), [])
        return result, output.getvalue()

    def test_number_assigns_existing_role_and_preserves_source_voice_delivery_and_offsets(self):
        document = missing()
        document["segments"][1]["delivery"] = {"direction": "Quietly", "evidence": [{"block_id": "b1", "quote": "softly"}]}
        before = deepcopy(document)
        result, output = self.decide(document, "1\n")
        self.assertEqual(document, before)
        self.assertEqual(result["source"], before["source"])
        self.assertEqual(result["cast"], before["cast"])
        selected = result["segments"][1]
        self.assertEqual((selected["role_id"], selected["voice_id"]), ("mara", "v1"))
        self.assertEqual(selected["attribution"]["basis"], "user")
        for key in ("text", "id", "block_id", "start", "end", "delivery"):
            self.assertEqual(selected[key], before["segments"][1][key])
        audit = result["provenance"]["speaker_decisions"]
        self.assertEqual(audit["source_fingerprint"], fingerprint(document["source"]))
        self.assertEqual(audit["decisions"][0]["selected_voices"], ["v1"])
        self.assertEqual(audit["remaining_unassigned_segments"], 0)
        self.assertIn("1. Mara — best guess (Enter)", output)
        self.assertNotIn("dialogue; mara", output)
        self.assertIn("[b1:18:26]", output)
        self.assertIn("PASSAGE TO ASSIGN — 8 characters\n> “Hello.”", output)
        self.assertTrue(result["readiness"]["ready"])

    def test_comma_numbers_explicitly_assign_simultaneous_characters(self):
        document = missing()
        add_lena(document)
        result, output = self.decide(document, "o\n2,3\n")
        selected = result["segments"][1]
        self.assertIsNone(selected["role_id"])
        self.assertIsNone(selected["voice_id"])
        self.assertEqual(selected["simultaneous_speakers"], [
            {"role_id": "mara", "voice_id": "v1"}, {"role_id": "lena", "voice_id": "v2"}])
        self.assertEqual(selected["attribution"]["basis"], "user")
        self.assertIn("SAME words together", output)
        self.assertEqual(unassigned_segments(result), [])

    def test_mixed_block_shows_assigned_narration_tail_and_exact_unassigned_range(self):
        document = missing()
        text = "As I had done all of my life, a voice reminded me as I worked."
        boundary = len("As I had done all of my life,")
        document["source"]["blocks"][0]["text"] = text
        document["source"]["blocks"][0]["inline_spans"] = [
            {"start": 0, "end": boundary, "kind": "emphasis", "origin": "style"}]
        document["cast"]["characters"][0]["evidence"] = []
        document["cast"]["voices"][1]["traits"] = []
        thought, tail = deepcopy(document["segments"][1]), deepcopy(document["segments"][0])
        thought.update(id="b1:0:29", start=0, end=boundary, text=text[:boundary], kind="thought")
        tail.update(id=f"b1:{boundary}:{len(text)}", start=boundary, end=len(text), text=text[boundary:])
        document["segments"] = [thought, tail, document["segments"][2]]
        before = deepcopy(document)
        result, output = self.decide(document, "q\n")
        self.assertIn("After (context)\n  [b1]\n  " + text[boundary:].strip(), output)
        self.assertIn(f"Already assigned: Narrator [b1:{boundary}:{len(text)}]", output)
        self.assertIn("[b1:0:29]", output)
        self.assertIn("PASSAGE TO ASSIGN — 29 characters\n> " + text[:boundary], output)
        self.assertNotIn("> " + text, output)
        self.assertEqual(document, before)
        self.assertEqual(result["segments"], before["segments"])

    def test_two_neighbors_show_plural_attribution_after_repeated_calls(self):
        document = missing()
        call_text = "“Jasper.”"
        document["source"]["blocks"][0]["text"] = call_text
        document["source"]["blocks"][1]["text"] = call_text
        document["cast"]["characters"][0]["evidence"] = []
        document["cast"]["voices"][1]["traits"] = []
        third = deepcopy(document["source"]["blocks"][1])
        third.update(id="b3", text="Two feminine voices called my name.")
        document["source"]["blocks"].append(third)
        document["source"]["chapters"][0]["block_ids"].append("b3")
        target = deepcopy(document["segments"][1])
        target.update(start=0, end=len(call_text), text=call_text)
        later = deepcopy(document["segments"][2])
        later.update(start=0, end=len(call_text), text=call_text)
        narration = deepcopy(later)
        narration.update(id="s4", block_id="b3", end=len(third["text"]), text=third["text"])
        document["segments"] = [target, later, narration]
        _, output = self.decide(document, "q\n")
        self.assertIn("  [b2]\n  “Jasper.”", output)
        self.assertIn("  [b3]\n  Two feminine voices called my name.", output)

    def test_context_is_bounded_but_passage_is_whole_and_display_is_sanitized(self):
        document = missing((2,))
        text = "ordinary text " * 100
        context = document["source"]["blocks"][0]["text"] + text
        document["source"]["blocks"][0]["text"] = context
        document["segments"][1].update(text=context[document["segments"][1]["start"]:], end=len(context))
        document["source"]["blocks"][1]["text"] = text
        document["segments"][2].update(text=text, end=len(text))
        document["cast"]["characters"][0]["name"] = "Mara\x1b"
        _, output = self.decide(document, "q\n")
        self.assertIn("[context truncated]", output)
        target_lines = [line[2:] for line in output.splitlines() if line.startswith("> ")]
        self.assertEqual(" ".join(target_lines), text.strip())
        self.assertTrue(all(len(line) <= 88 for line in target_lines))
        self.assertIn("\n\nPASSAGE TO ASSIGN", output)
        self.assertNotIn("\x1b", output)

    def test_invalid_numbers_duplicate_roles_narrators_and_duplicate_voices_reprompt(self):
        document = missing()
        add_lena(document)
        duplicate = deepcopy(document["cast"]["roles"][1])
        duplicate["id"] = "mara_other"
        document["cast"]["roles"].append(duplicate)
        result, output = self.decide(document, "o\n0\nx\n2,2\n1,2\n2,4\n2,3\n")
        self.assertEqual(unassigned_segments(result), [])
        self.assertIn("Choose each speaker only once", output)
        self.assertIn("narration cannot be a group member", output)
        self.assertEqual(output.count("Speaker number(s),"), 7)

    def test_all_roles_are_available_even_when_not_in_candidate_list(self):
        document = missing()
        add_lena(document)
        result, output = self.decide(document, "o\n3\n")
        self.assertNotIn("3. Lena", output.split("All voices", 1)[0])
        self.assertIn("3. Lena", output.split("All voices", 1)[1])
        self.assertEqual(result["segments"][1]["role_id"], "lena")

    def test_enter_accepts_best_guess_then_quit_keeps_decisions(self):
        document = missing((1, 2))
        result, output = self.decide(document, "\nq\n")
        self.assertEqual([item["id"] for item in unassigned_segments(result)], ["s3"])
        self.assertEqual(result["segments"][1]["role_id"], "mara")
        self.assertEqual(result["segments"][1]["attribution"]["basis"], "user")
        self.assertEqual(output.count("Speaker number(s),"), 2)
        self.assertIn("Enter accepts the best guess", output)
        self.assertIn("epub-audiobook resolve-speakers 'books/A Book'", output)
        self.assertFalse(result["readiness"]["ready"])

    def test_enter_in_full_cast_does_not_select_and_back_restores_shortlist_mapping(self):
        document = missing()
        add_lena(document)
        result, output = self.decide(document, "o\n\nb\n1\n")
        self.assertEqual(result["segments"][1]["role_id"], "mara")
        self.assertIn("1. Narrator", output.split("All voices", 1)[1])
        self.assertIn("Enter selects a voice only", output)
        self.assertEqual(output.count("Most likely speakers"), 2)

    def test_no_likely_options_requires_other_and_never_accepts_blank(self):
        with patch("epub_audiobook.speaker_decisions.rank_speakers", return_value=[]):
            result, output = self.decide(missing(), "\n1\no\n2\n")
        self.assertEqual(result["segments"][1]["role_id"], "mara")
        self.assertIn("No likely speaker found", output)
        self.assertEqual(output.count("Speaker number(s),"), 4)

    def test_three_ranked_choices_use_ranking_order_not_full_cast_order(self):
        document = missing()
        add_lena(document)
        ranked = [{"role_id": role, "reason": "Plausible in this scene.", "score": score}
                  for score, role in zip((3, 2, 1), ("lena", "mara", "narrator"))]
        with patch("epub_audiobook.speaker_decisions.rank_speakers", return_value=ranked):
            result, output = self.decide(document, "1\n")
        self.assertEqual(result["segments"][1]["role_id"], "lena")
        self.assertIn("1. Lena — best guess", output)
        self.assertIn("2. Mara", output)
        self.assertIn("3. Narrator", output)
        self.assertNotIn("All voices", output)

    def test_full_cast_distinguishes_same_named_characters_without_raw_ids(self):
        document = missing()
        add_lena(document)
        document["cast"]["characters"][1]["name"] = "Mara"
        document["cast"]["roles"][2]["description"] = "Mara the shopkeeper."
        result, output = self.decide(document, "o\n3\n")
        self.assertIn("2. Mara — Mara speaking.", output)
        self.assertIn("3. Mara — Mara the shopkeeper.", output)
        self.assertNotIn("[mara]", output)
        self.assertNotIn("[lena]", output)
        self.assertEqual(result["segments"][1]["role_id"], "lena")

    def test_character_narration_is_distinct_from_their_dialogue_voice(self):
        document = missing()
        document["cast"]["roles"][0].update(character_id="c1", voice_id="v1")
        for segment in document["segments"]:
            if segment["role_id"] == "narrator":
                segment["voice_id"] = "v1"
        result, output = self.decide(document, "o\n1\n")
        self.assertIn("1. Mara (narrator)", output)
        self.assertIn("2. Mara\n", output)
        self.assertEqual(result["segments"][1]["role_id"], "narrator")
        self.assertEqual(result["segments"][1]["kind"], "narration")

    def test_author_narration_is_named_from_book_metadata_in_full_cast(self):
        document = missing()
        document["source"]["book"]["authors"] = ["Nathan Thompson"]
        document["cast"]["roles"][0]["description"] = "Author afterword and acknowledgments."
        result, output = self.decide(document, "o\n1\n")
        self.assertIn("1. Nathan Thompson (author)", output)
        self.assertEqual(result["segments"][1]["role_id"], "narrator")
        self.assertEqual(result["source"], document["source"])

    def test_context_and_options_wrap_to_a_narrow_terminal(self):
        document = missing()
        document["source"]["book"]["title"] = "A long title that should still be readable in a narrow terminal window"
        with patch("epub_audiobook.speaker_decisions.os.get_terminal_size",
                   return_value=os.terminal_size((48, 24))):
            # StringIO lacks fileno; supply a stream with an explicit terminal descriptor.
            class Output(StringIO):
                def fileno(self):
                    return 2
            output = Output()
            prompt_speaker_decisions(document, input_stream=Terminal("q\n"), output_stream=output)
        displayed = output.getvalue().split("Speaker number(s),", 1)[0].splitlines()
        self.assertTrue(all(len(line) <= 48 for line in displayed), displayed)

    def test_eof_preserves_earlier_decisions_and_leaves_remaining_pending(self):
        result, output = self.decide(missing((1, 2)), "1\n")
        self.assertEqual([item["id"] for item in unassigned_segments(result)], ["s3"])
        self.assertEqual(result["segments"][1]["attribution"]["basis"], "user")
        self.assertIn("Input ended", output)

    def test_explicit_nonterminal_never_reads_input_or_falls_back_to_another_terminal(self):
        class Unreadable(StringIO):
            def readline(self):
                raise AssertionError("Noninteractive input must not be read")
        document, output = missing(), StringIO()
        with patch("epub_audiobook.speaker_decisions.open") as terminal_open, \
                patch("epub_audiobook.speaker_decisions.rank_speakers") as rank:
            result = prompt_speaker_decisions(document, book_path="book-01", input_stream=Unreadable("2\n"), output_stream=output)
        terminal_open.assert_not_called()
        rank.assert_not_called()
        self.assertEqual(result, document)
        self.assertIn("Speaker selection incomplete", output.getvalue())
        self.assertIn("epub-audiobook resolve-speakers book-01", output.getvalue())

    def test_redirected_stdin_asks_on_controlling_terminal_for_input_and_output(self):
        class Unreadable(StringIO):
            def readline(self):
                raise AssertionError("The redirected input must not be consumed")
        terminal_input, terminal_output = Terminal("1\n"), StringIO()
        redirected_output = StringIO()
        with patch("sys.stdin", Unreadable()), patch("sys.stderr", redirected_output), \
                patch("epub_audiobook.speaker_decisions.open", side_effect=[
                    nullcontext(terminal_input), nullcontext(terminal_output)]) as terminal_open:
            result = prompt_speaker_decisions(missing(), book_path="book-01")
        self.assertEqual(terminal_open.call_args_list, [
            call("/dev/tty", "r", encoding="utf-8"), call("/dev/tty", "w", encoding="utf-8")])
        self.assertEqual(unassigned_segments(result), [])
        self.assertIn("Speaker number(s)", terminal_output.getvalue())
        self.assertEqual(redirected_output.getvalue(), "")

    def test_controlling_terminal_uses_explicit_output_and_preserves_decisions_at_eof(self):
        terminal_input, output = Terminal("1\n"), StringIO()
        with patch("sys.stdin", StringIO()), \
                patch("epub_audiobook.speaker_decisions.open", return_value=nullcontext(terminal_input)) as terminal_open:
            result = prompt_speaker_decisions(missing((1, 2)), book_path="book-01", output_stream=output)
        terminal_open.assert_called_once_with("/dev/tty", "r", encoding="utf-8")
        self.assertEqual(result["segments"][1]["attribution"]["basis"], "user")
        self.assertEqual([item["id"] for item in unassigned_segments(result)], ["s3"])
        self.assertIn("Input ended", output.getvalue())

    def test_missing_controlling_terminal_saves_without_consuming_redirected_input(self):
        class Unreadable(StringIO):
            def readline(self):
                raise AssertionError("There is no terminal; do not consume redirected input")
        document, output = missing(), StringIO()
        with patch("sys.stdin", Unreadable("2\n")), patch("sys.stderr", output), \
                patch("epub_audiobook.speaker_decisions.open", side_effect=OSError("No controlling terminal")):
            result = prompt_speaker_decisions(document, book_path="book-01")
        self.assertEqual(result, document)
        self.assertIn("no interactive terminal is available", output.getvalue())
        self.assertIn("epub-audiobook resolve-speakers book-01", output.getvalue())
        self.assertNotIn("Speaker number(s)", output.getvalue())

    def test_terminal_output_open_failure_closes_input_and_leaves_passage_unassigned(self):
        terminal_input, output = Terminal("1\n"), StringIO()
        document = missing()
        with patch("sys.stdin", StringIO()), patch("sys.stderr", output), \
                patch("epub_audiobook.speaker_decisions.open", side_effect=[terminal_input, OSError("Output unavailable")]):
            result = prompt_speaker_decisions(document, book_path="book-01")
        self.assertTrue(terminal_input.closed)
        self.assertEqual(result, document)
        self.assertIn("no interactive terminal is available", output.getvalue())

    def test_only_covered_attribution_issues_clear_and_manual_decision_survives_retry(self):
        document = missing()
        document["issues"] = [
            {"id": "a", "kind": "speaker_attribution", "target": "s2", "reason": "Unknown.", "resolved": False},
            {"id": "b", "kind": "attribution_consensus", "target": "b1", "reason": "Unknown.", "resolved": False},
            {"id": "c", "kind": "casting", "target": "b1", "reason": "Review voice.", "resolved": False},
        ]
        result, _ = self.decide(document, "1\n")
        self.assertEqual([issue["resolved"] for issue in result["issues"]], [True, True, False])
        class NoRequests:
            def run(self, *args, **kwargs):
                raise AssertionError("No model call is needed for a resolved user decision")
        retried = complete_speakers(result, NoRequests())
        self.assertEqual(retried["segments"], result["segments"])
        self.assertEqual(retried["provenance"]["speaker_decisions"], result["provenance"]["speaker_decisions"])

    def test_compact_reasons_replace_raw_jury_reports_and_strip_terminal_controls(self):
        document = missing()
        proposal = deepcopy(document["segments"][1])
        proposal.update(role_id="mara", voice_id="v1")
        document["provenance"]["speaker_completion"] = {"cases": [{"block_id": "b1", "rounds": [{"votes": [
            {"reviewer_id": "speaker_completion_1", "proposed_segments": [proposal],
             "raw_response": {"reason": "The tag names Mara.\x1b[31m"}},
        ]}]}]}
        ranked = [{"role_id": "mara", "score": 1, "reason": "The tag names Mara.\x1b[31m"}]
        with patch("epub_audiobook.speaker_decisions.rank_speakers", return_value=ranked):
            result, output = self.decide(document, "q\n")
        self.assertEqual(result["segments"], document["segments"])
        self.assertNotIn("speaker_completion_1", output)
        self.assertIn("The tag names Mara", output)
        self.assertNotIn("\x1b", output)

    def test_jev_client_reaches_lazy_ranker_through_controlling_terminal(self):
        client = object()
        terminal_input, output = Terminal("q\n"), StringIO()
        with patch("sys.stdin", StringIO()), \
                patch("epub_audiobook.speaker_decisions.open", return_value=nullcontext(terminal_input)), \
                patch("epub_audiobook.speaker_decisions.rank_speakers", return_value=[]) as rank:
            result = prompt_speaker_decisions(missing(), jev_client=client, output_stream=output)
        self.assertIs(rank.call_args.kwargs["jev_client"], client)
        self.assertIn("Ranking likely speakers with JEV", output.getvalue())
        self.assertEqual(len(unassigned_segments(result)), 1)

    def analyze_with_input(self, document, input_stream, expected_status=0):
        """Exercise analyze's actual prompt and handoff without model or disk work."""
        from epub_audiobook.cli import main
        config = {"provider": {"model": "test"}, "extraction": {"include_auxiliary": False},
                  "analysis": {"series_registry": None, "cache_dir": None, "chunk_chars": 1000,
                               "repair_attempts": 0, "max_request_chars": 200000, "whole_book_discovery": True,
                               "attribution_workers": 5, "identity_consensus": True, "attribution_review": True,
                               "attribution_review_chunk_chars": 1000, "attribution_review_notes": None,
                               "cast_review_notes": None, "cast_reconciliation_rounds": 1,
                               "cast_reconciliation_notes": None}, "performance_policy": {}}
        published, preparations = [], []

        class Client:
            model, base_url = "test", "https://example.invalid"

            def complete(self, *args, **kwargs):
                raise AssertionError("This CLI decision test must not call a model")

        def prepare(doc, requests, **kwargs):
            preparations.append((deepcopy(doc), requests is None))
            self.assertEqual(kwargs["workers"], 5)
            return deepcopy(doc)

        def publish(doc, registry, output, exporter, **kwargs):
            self.assertEqual(output, Path("fresh-book"))
            self.assertEqual(validate_document(doc), [])
            published.append(deepcopy(doc))
            return readiness(doc)

        with patch("epub_audiobook.cli.load_config", return_value=config), \
                patch("epub_audiobook.cli.source_from", return_value=document["source"]), \
                patch("epub_audiobook.cli.validate_output", side_effect=lambda path: path), \
                patch("epub_audiobook.cli.fresh_registry", return_value={}), \
                patch("epub_audiobook.cli.OpenAICompatibleClient", return_value=Client()), \
                patch("epub_audiobook.cli.analyze", side_effect=lambda *args, **kwargs: deepcopy(document)), \
                patch("epub_audiobook.audition_scripts.prepare_document_samples", side_effect=prepare), \
                patch("epub_audiobook.cli.publish_directory", side_effect=publish), \
                patch("epub_audiobook.speaker_decisions.open", side_effect=OSError("No controlling terminal")), \
                patch("sys.stdin", input_stream), \
                patch("sys.stdout", new_callable=StringIO) as output, \
                patch("sys.stderr", new_callable=StringIO) as diagnostics:
            self.assertEqual(main(["analyze", "input.epub", "-o", "fresh-book"]), expected_status)
        self.assertEqual(len(published), 1)
        # The refresh after user input must have no request client available.
        self.assertEqual([no_requests for _, no_requests in preparations], [False, True])
        return published[0], preparations, output.getvalue(), diagnostics.getvalue()

    def test_cli_analyze_publishes_actual_terminal_choice_and_preserves_source_metadata(self):
        document = missing()
        add_lena(document)
        selected = document["segments"][1]
        selected["attribution"]["evidence"] = [{"block_id": "b1", "quote": "Mara"}]
        selected["delivery"] = {"direction": "Quietly", "evidence": [{"block_id": "b1", "quote": "softly"}]}
        before = deepcopy(document)

        result, preparations, output, diagnostics = self.analyze_with_input(document, Terminal("o\n3\n"))

        self.assertEqual(document, before)
        self.assertEqual(result["source"], before["source"])
        self.assertEqual(result["cast"], before["cast"])
        selected = result["segments"][1]
        # The actual typed choice wins even though the model only suggested Mara.
        self.assertEqual((selected["role_id"], selected["voice_id"]), ("lena", "v2"))
        self.assertEqual(selected["attribution"]["basis"], "user")
        self.assertEqual(selected["attribution"]["evidence"], before["segments"][1]["attribution"]["evidence"])
        for key in ("id", "text", "block_id", "start", "end", "delivery"):
            self.assertEqual(selected[key], before["segments"][1][key])
        self.assertEqual(result["segments"][0], before["segments"][0])
        self.assertEqual(result["segments"][2], before["segments"][2])
        self.assertIsNone(preparations[0][0]["segments"][1]["role_id"])
        self.assertEqual(preparations[1][0]["segments"][1]["role_id"], "lena")
        self.assertEqual(result["provenance"]["speaker_decisions"]["decisions"][0]["selected_roles"], ["lena"])
        self.assertTrue(readiness(result)["ready"])
        self.assertIn("Speaker number(s)", diagnostics)
        self.assertIn("Bundle created", output)
        self.assertIn("No review step is required", output)
        self.assertNotIn("resolve-speakers", diagnostics)

    def test_cli_analyze_without_terminal_saves_incomplete_bundle_and_returns_status_two(self):
        class Unreadable(StringIO):
            def readline(self, *args, **kwargs):
                raise AssertionError("analyze must not read nonterminal input")

        document = missing()
        result, _, output, diagnostics = self.analyze_with_input(document, Unreadable("2\n"), expected_status=2)

        self.assertEqual(result, document)
        self.assertFalse(readiness(result)["ready"])
        self.assertEqual([item["id"] for item in unassigned_segments(result)], ["s2"])
        self.assertIn("no interactive terminal is available", diagnostics)
        self.assertIn("Exiting with status 2", diagnostics)
        self.assertNotIn("Speaker number(s)", diagnostics)
        self.assertIn("Bundle created", output)
        self.assertNotIn("draft", output)

    def test_cli_analyze_quit_or_eof_publishes_choices_already_made(self):
        for answers in ("1\nq\n", "1\n"):
            with self.subTest(answers=answers):
                document = missing((1, 2))
                result, _, output, diagnostics = self.analyze_with_input(document, Terminal(answers), expected_status=2)

                self.assertEqual((result["segments"][1]["role_id"], result["segments"][1]["voice_id"]), ("mara", "v1"))
                self.assertEqual(result["segments"][1]["attribution"]["basis"], "user")
                self.assertEqual(result["segments"][2], document["segments"][2])
                self.assertEqual([item["id"] for item in unassigned_segments(result)], ["s3"])
                self.assertEqual(result["provenance"]["speaker_decisions"]["remaining_unassigned_segments"], 1)
                self.assertFalse(readiness(result)["ready"])
                self.assertIn("Bundle created", output)
                self.assertIn("Speaker selection incomplete", diagnostics)
                self.assertNotIn("draft", output)
                if answers == "1\n":
                    self.assertIn("Input ended", diagnostics)

    def test_cli_analyze_runs_decisions_after_model_and_before_publication(self):
        from epub_audiobook.cli import main
        document = missing()
        events = []
        config = {"provider": {"model": "test"}, "extraction": {"include_auxiliary": False},
                  "analysis": {"series_registry": None, "cache_dir": None, "chunk_chars": 1000,
                               "repair_attempts": 0, "max_request_chars": 200000, "whole_book_discovery": True,
                               "attribution_workers": 5, "identity_consensus": True, "attribution_review": True,
                               "attribution_review_chunk_chars": 1000, "attribution_review_notes": None,
                               "cast_review_notes": None, "cast_reconciliation_rounds": 1,
                               "cast_reconciliation_notes": None}, "performance_policy": {}}
        class Client:
            model, base_url = "test", "https://example.invalid"
        def analyze(*args, **kwargs):
            events.append("automatic")
            return document
        def decide(doc, **kwargs):
            events.append("decisions")
            return doc
        def publish(*args, **kwargs):
            events.append("publish")
            return {"ready": False, "review_count": 1}
        def prepare(doc, *args, **kwargs):
            events.append("auditions")
            return doc
        with patch("epub_audiobook.cli.load_config", return_value=config), \
                patch("epub_audiobook.cli.source_from", return_value=document["source"]), \
                patch("epub_audiobook.cli.validate_output", side_effect=lambda path: path), \
                patch("epub_audiobook.cli.fresh_registry", return_value={}), \
                patch("epub_audiobook.cli.OpenAICompatibleClient", return_value=Client()), \
                patch("epub_audiobook.cli.analyze", side_effect=analyze), \
                patch("epub_audiobook.cli.prompt_speaker_decisions", side_effect=decide), \
                patch("epub_audiobook.audition_scripts.prepare_document_samples", side_effect=prepare), \
                patch("epub_audiobook.cli.publish_directory", side_effect=publish), \
                patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO):
            self.assertEqual(main(["analyze", "input.epub", "-o", "fresh-book"]), 2)
        self.assertEqual(events, ["automatic", "auditions", "decisions", "auditions", "publish"])


if __name__ == "__main__":
    unittest.main()
