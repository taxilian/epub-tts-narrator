"""Suggestions narrow human choices without making or changing assignments."""

from copy import deepcopy
import math
import unittest

from epub_tts_narrator.jev import JevError
from epub_tts_narrator.speaker_suggestions import rank_speakers
from test_contracts import valid_document


def scene(texts, *, people=("Mara", "Lena", "Jasper", "Stranger"), candidates=(), assigned=None):
    document = valid_document()
    document["cast"]["characters"] = [
        {"id": name.lower(), "name": name, "aliases": [], "description": name + " in the story."}
        for name in people]
    document["cast"]["roles"] = [document["cast"]["roles"][0]] + [
        {"id": name.lower(), "character_id": name.lower(), "voice_id": name.lower() + "_voice",
         "kind": "dialogue", "description": name + " speaking."} for name in people]
    template = document["segments"][1]
    document["source"]["blocks"] = []
    document["segments"] = []
    for index, text in enumerate(texts):
        block_id = f"b{index}"
        document["source"]["blocks"].append({"id": block_id, "chapter_id": "ch1", "text": text})
        item = deepcopy(template)
        role_id = (assigned or {}).get(index)
        item.update(id=block_id + ":0:" + str(len(text)), block_id=block_id, start=0, end=len(text), text=text,
                    role_id=role_id, voice_id=role_id + "_voice" if role_id else None,
                    kind="narration" if role_id == "narrator" else "dialogue")
        item["attribution"].update(candidates=list(candidates) if index == 0 else [], basis="explicit" if role_id else "unresolved")
        document["segments"].append(item)
    return document, document["segments"][0]


def jury(document, target, choices):
    votes = []
    for index, choice in enumerate(choices):
        proposed = deepcopy(target)
        proposed.update(role_id=choice if isinstance(choice, str) else None)
        proposed["attribution"]["candidates"] = choice if isinstance(choice, list) else []
        votes.append({"reviewer_id": f"reviewer_{index}", "status": "valid", "proposed_segments": [proposed]})
    document["provenance"]["speaker_completion"] = {"cases": [
        {"block_id": target["block_id"], "rounds": [{"votes": votes}]}]}


def add_narrator(document, role_id, description):
    document["cast"]["roles"].append({"id": role_id, "kind": "narration", "character_id": None,
                                      "voice_id": role_id + "_voice", "description": description})


class Judge:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {f"candidate_{i}": {"type": "noul", "noul": self.scores[row["role_id"]]}
                            for i, row in enumerate(state["candidates"])}}


class SpeakerSuggestionTests(unittest.TestCase):
    def test_original_candidates_and_source_tag_work_without_jury_or_service(self):
        document = valid_document()
        target = document["segments"][1]
        target.update(role_id=None, voice_id=None)
        before = deepcopy(document)
        result = rank_speakers(document, target)
        self.assertEqual([row["role_id"] for row in result], ["mara"])
        self.assertIn("speech tag", result[0]["reason"])
        self.assertEqual(document, before)

    def test_no_signals_produces_no_arbitrary_roles_or_narrator(self):
        document, target = scene(["“Hello.”"])
        self.assertEqual(rank_speakers(document, target), [])

    def test_disagreeing_reviewers_order_supported_alternatives(self):
        document, target = scene(["“Hello.”"])
        jury(document, target, ["lena", "mara", "lena"])
        result = rank_speakers(document, target)
        self.assertEqual([row["role_id"] for row in result], ["lena", "mara"])
        self.assertIn("2 reviewers", result[0]["reason"])
        self.assertGreater(result[0]["score"], result[1]["score"])

    def test_only_latest_round_and_matching_span_votes_are_used(self):
        document, target = scene(["“Hello.”", "“Later.”"])
        jury(document, target, ["lena"])
        case = document["provenance"]["speaker_completion"]["cases"][0]
        old = deepcopy(case["rounds"][0])
        old["votes"][0]["proposed_segments"][0]["role_id"] = "stranger"
        case["rounds"].insert(0, old)
        other = deepcopy(old["votes"][0])
        other["reviewer_id"] = "other"
        other["proposed_segments"][0].update(block_id="b1", role_id="stranger")
        case["rounds"][-1]["votes"].append(other)
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["lena"])

    def test_invalid_role_ids_and_invalid_votes_never_reach_menu(self):
        document, target = scene(["“Hello.”"], candidates=["missing", "mara", {}, None])
        jury(document, target, ["invented", "lena"])
        document["provenance"]["speaker_completion"]["cases"][0]["rounds"][0]["votes"][1]["status"] = "error"
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["mara"])

    def test_book4_calls_offer_nova_vessa_but_not_the_named_addressee(self):
        document, target = scene([
            "“Jasper.”", "“Jasper.”", "Two feminine voices called my name.",
            "“Jasper!” a young woman shouted.", "“He’ll be alright, Nova,” the other woman said sadly.",
            "Vessa leaned closer."], people=("Nova", "Vessa", "Jasper", "Stranger"),
            candidates=["nova", "vessa"], assigned={2: "narrator", 3: "nova", 4: "vessa", 5: "narrator"})
        jury(document, target, [["nova", "vessa"], ["vessa", "nova"], ["nova", "vessa"]])
        result = rank_speakers(document, target)
        self.assertEqual({row["role_id"] for row in result}, {"nova", "vessa"})
        self.assertEqual(len(result), 2)
        self.assertTrue(all(math.isfinite(row["score"]) for row in result))

    def test_nearby_dialogue_is_useful_but_another_chapter_is_excluded(self):
        document, target = scene(["“Who is there?”", "“Me.”", "“A different chapter.”"], assigned={1: "lena", 2: "stranger"})
        document["source"]["blocks"][2]["chapter_id"] = "ch2"
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["lena"])

    def test_narrative_target_can_offer_narrator_without_turning_all_dialogue_into_narration(self):
        document, target = scene(["I walked home."])
        target["kind"] = "narration"
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["narrator"])

    def test_author_section_introduces_author_even_when_target_guessed_dialogue(self):
        for title in ("From the Author", "Author's Afterword", "Afterword", "Acknowledgments"):
            with self.subTest(title=title):
                document, target = scene(["Thank you for reading my book."])
                document["source"]["chapters"][0]["title"] = title
                add_narrator(document, "author", "Author afterword and acknowledgments.")
                judge = Judge({"author": .95})
                result = rank_speakers(document, target, jev_client=judge)
                self.assertEqual([row["role_id"] for row in result], ["author"])
                state, _ = judge.calls[0]
                self.assertEqual(state["chapter"]["title"], title)
                self.assertEqual(state["candidates"][0]["name"], "A. Writer")

    def test_single_generic_narrator_still_covers_afterword_without_dedicated_author_role(self):
        document, target = scene(["Thank you for reading."])
        target["kind"] = "narration"
        document["source"]["chapters"][0]["title"] = "Afterword"
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["narrator"])

    def test_story_narration_does_not_suggest_unrelated_author_voice(self):
        document, target = scene(["I crossed the dark street."])
        target["kind"] = "narration"
        add_narrator(document, "author", "Author afterword and acknowledgments.")
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["narrator"])

    def test_nearby_assigned_author_narration_can_rescue_wrong_target_kind(self):
        document, target = scene(["Thanks for reading.", "My next book will be available soon."], assigned={1: "author"})
        add_narrator(document, "author", "Author afterword and acknowledgments.")
        document["segments"][1]["kind"] = "narration"
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["author"])
        document["source"]["blocks"][1]["chapter_id"] = "ch2"
        self.assertEqual(rank_speakers(document, target), [])

    def test_story_quote_about_author_does_not_create_author_narration_candidate(self):
        document, target = scene(["I handed her the book.", "“Read the author's afterword,” Mara said."], assigned={1: "mara"})
        target["kind"] = "narration"
        add_narrator(document, "author", "Author afterword and acknowledgments.")
        self.assertNotIn("author", [row["role_id"] for row in rank_speakers(document, target)])

    def test_preceding_author_heading_works_but_future_heading_does_not(self):
        document, _ = scene(["From the Author", "Thank you for reading."])
        document["source"]["blocks"][0]["source"] = {"tag": "h1"}
        add_narrator(document, "author", "Author afterword and acknowledgments.")
        self.assertEqual([row["role_id"] for row in rank_speakers(document, document["segments"][1])], ["author"])
        document, target = scene(["The battle ended.", "From the Author"])
        document["source"]["blocks"][1]["source"] = {"tag": "h1"}
        add_narrator(document, "author", "Author afterword and acknowledgments.")
        self.assertEqual(rank_speakers(document, target), [])

    def test_multiple_pov_narrators_follow_nearby_assignments_not_every_role(self):
        document, target = scene(["I walked home.", "The streets were empty."], assigned={1: "second_pov"})
        target["kind"] = "narration"
        document["segments"][1]["kind"] = "narration"
        add_narrator(document, "second_pov", "An alternate narrative viewpoint.")
        add_narrator(document, "third_pov", "Another narrative viewpoint.")
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target)], ["second_pov"])
        document["segments"][1].update(role_id=None, voice_id=None)
        self.assertEqual(rank_speakers(document, target), [])

    def test_user_confirmed_neighbor_is_highlighted_for_joint_scene_reasoning(self):
        document, _ = scene(["“Jasper.”", "“Jasper.”", "Two feminine voices called my name."],
                            people=("Nova", "Vessa", "Jasper"), assigned={0: "nova", 2: "narrator"})
        target = document["segments"][1]
        target["attribution"]["candidates"] = ["nova", "vessa"]
        document["segments"][0]["attribution"]["basis"] = "user"
        judge = Judge({"nova": .2, "vessa": .9})
        result = rank_speakers(document, target, jev_client=judge)
        self.assertEqual([row["role_id"] for row in result], ["vessa"])
        state, questions = judge.calls[0]
        confirmed = state["nearby_confirmed_choices"]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["role_ids"], ["nova"])
        self.assertEqual(confirmed[0]["relative_block_position"], -1)
        self.assertTrue(confirmed[0]["same_words_as_target"])
        self.assertEqual(confirmed[0]["basis"], "user")
        instructions = questions["candidate_0"]["instructions"]
        self.assertIn("explicitly describes distinct voices", instructions)
        self.assertIn("Do not mechanically alternate", instructions)

    def test_confirmed_neighbor_does_not_force_alternation(self):
        document, _ = scene(["“One more thing.”", "“Follow me.”"], assigned={0: "mara"})
        document["segments"][0]["attribution"]["basis"] = "user"
        target = document["segments"][1]
        target["attribution"]["candidates"] = ["mara", "lena"]
        judge = Judge({"mara": .95, "lena": .1})
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target, jev_client=judge)], ["mara"])

    def test_judge_can_rerank_plausible_choices_and_receives_source_not_only_words(self):
        document, target = scene(["“Come here.”", "Lena waved. Mara stayed near the door."], candidates=["mara", "lena"], assigned={1: "narrator"})
        judge = Judge({"mara": .35, "lena": .9})
        before = deepcopy(document)
        result = rank_speakers(document, target, jev_client=judge)
        self.assertEqual([row["role_id"] for row in result], ["lena", "mara"])
        state, questions = judge.calls[0]
        self.assertEqual(state["target"]["text"], "“Come here.”")
        self.assertIn("Lena waved", state["surrounding_source"][1]["text"])
        self.assertEqual(state["known_speakers"][0]["role_ids"], ["narrator"])
        self.assertIn("best guesses", questions["candidate_0"]["instructions"])
        self.assertIn("not itself the speaker", questions["candidate_0"]["instructions"])
        self.assertEqual(document, before)

    def test_judge_can_exclude_merely_mentioned_candidate(self):
        document, target = scene(["“Hello.”"], candidates=["mara", "lena"])
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target, jev_client=Judge({"mara": .9, "lena": .01}))], ["mara"])

    def test_plausible_runner_up_survives_moderate_best_choice_score(self):
        document, target = scene(["“Hello.”"], candidates=["mara", "lena", "stranger"])
        result = rank_speakers(document, target, jev_client=Judge({"mara": .41, "lena": .18, "stranger": .05}))
        self.assertEqual([row["role_id"] for row in result], ["mara", "lena"])

    def test_close_judge_scores_make_ambiguity_visible(self):
        document, target = scene(["“Hello.”"], candidates=["mara", "lena"])
        result = rank_speakers(document, target, jev_client=Judge({"mara": .75, "lena": .73}))
        self.assertEqual(len(result), 2)
        self.assertTrue(all("Similar support to another choice" in row["reason"] for row in result))

    def test_all_rejected_suggestions_offer_no_false_best_guess(self):
        document, target = scene(["“Hello.”"], candidates=["mara", "lena"])
        self.assertEqual(rank_speakers(document, target, jev_client=Judge({"mara": .02, "lena": .05})), [])

    def test_judge_error_falls_back_without_changing_assignments(self):
        document, target = scene(["“Hello.”"], candidates=["mara", "lena"])
        expected = rank_speakers(document, target)
        class Unavailable:
            def evaluate(self, *args):
                raise JevError("Unavailable")
        self.assertEqual(rank_speakers(document, target, jev_client=Unavailable()), expected)

    def test_nonfinite_out_of_range_wrong_type_and_unexpected_answers_fall_back(self):
        document, target = scene(["“Hello.”"], candidates=["mara", "lena"])
        expected = rank_speakers(document, target)
        for score in (math.nan, math.inf, -1, 2, "0.9", True, None):
            with self.subTest(score=score):
                self.assertEqual(rank_speakers(document, target, jev_client=Judge({"mara": score, "lena": .9})), expected)
        class InventedAnswer:
            def evaluate(self, *args):
                return {"answers": {"invented_role": {"type": "noul", "noul": 1}}}
        self.assertEqual(rank_speakers(document, target, jev_client=InventedAnswer()), expected)

    def test_more_than_three_plausible_candidates_yields_top_three(self):
        document, target = scene(["“Hello.”"], candidates=["mara", "lena", "jasper", "stranger"])
        judge = Judge({"mara": .5, "lena": .6, "jasper": .7, "stranger": .8})
        self.assertEqual([row["role_id"] for row in rank_speakers(document, target, jev_client=judge)], ["stranger", "jasper", "lena"])

    def test_remote_context_is_bounded_even_with_many_long_blocks(self):
        document, target = scene(["“Hello.”", *["Mara walked. " * 500 for _ in range(80)]], candidates=["mara"])
        judge = Judge({"mara": .9})
        rank_speakers(document, target, jev_client=judge)
        state, questions = judge.calls[0]
        self.assertLessEqual(sum(len(row["text"]) for row in state["surrounding_source"]), 20000)
        self.assertLessEqual(len(state["surrounding_source"]), 25)
        self.assertLessEqual(len(questions), 32)

    def test_existing_assignment_is_not_reconsidered_or_sent_to_service(self):
        document, target = scene(["“Hello.”"], candidates=["mara"], assigned={0: "lena"})
        judge = Judge({"mara": .99})
        before = deepcopy(document)
        self.assertEqual(rank_speakers(document, target, jev_client=judge), [])
        self.assertEqual(judge.calls, [])
        self.assertEqual(document, before)

    def test_no_candidates_does_not_call_optional_service(self):
        document, target = scene(["“Hello.”"])
        judge = Judge({})
        self.assertEqual(rank_speakers(document, target, jev_client=judge), [])
        self.assertEqual(judge.calls, [])


if __name__ == "__main__":
    unittest.main()
