"""Paragraph synthesis preserves source text, performers, and accepted clip IDs."""

from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.paragraphs import group_paragraphs, paragraph_supported


def passages(texts, *, block="b1", offset=0, voices=None):
    chunks = []
    cursor = offset
    for index, text in enumerate(texts):
        voice = voices[index] if voices else "v1"
        row = {"id": f"{block}:{cursor}", "block_id": block, "chapter_id": "chapter1",
               "start": cursor, "end": cursor + len(text), "text": text,
               "voice_id": voice, "role_id": f"role-{voice}",
               "break_after": "paragraph" if index == len(texts) - 1 else "none",
               "delivery": {"direction": "" if index % 2 else "quietly", "evidence": []}}
        chunks.append({"chunk_id": f"chunk-{block}-{cursor}", "segment": row,
                       "spoken_text": text, "spoken_start": 0, "spoken_end": len(text),
                       "source_ranges": [{"start": cursor, "end": cursor + len(text)}],
                       "silent_control": False})
        cursor += len(text)
    return chunks


def plan(*chapters):
    return {"format": "audiobook-render-plan", "voice_ids": ["v1", "v2"],
            "chapters": [{"id": f"chapter-{number}", "chunks": chunks}
                         for number, chunks in enumerate(chapters, 1)]}


def chunks_of(result, chapter=0):
    return result["chapters"][chapter]["chunks"]


class ParagraphGroupingTests(unittest.TestCase):
    def test_single_voice_capability_accepts_only_uniform_paragraphs(self):
        uniform = chunks_of(group_paragraphs(plan(passages(["Hello. ", "Goodbye."]))))[0]
        mixed = chunks_of(group_paragraphs(plan(passages(["Hello. ", "Goodbye."],
                                   voices=["v1", "v2"])), multi_speaker=True))[0]
        capabilities = {"multi_speaker": False, "single_speaker_paragraph": True}
        self.assertTrue(paragraph_supported(capabilities, uniform))
        self.assertFalse(paragraph_supported(capabilities, mixed))
        self.assertFalse(paragraph_supported({"multi_speaker": False}, uniform))
        self.assertTrue(paragraph_supported({"multi_speaker": True}, mixed))
        self.assertFalse(paragraph_supported({"single_speaker_paragraph": "true"}, uniform))

    def test_explicit_legacy_groups_preserve_original_speakers_and_directions(self):
        members = passages(['“Hello,” ', 'she said. ', '“Come inside.”'], offset=12,
                           voices=["v1", "v2", "v1"])
        source = plan(members)
        original = deepcopy(source)
        grouped = group_paragraphs(source, multi_speaker=True)
        self.assertEqual(len(chunks_of(grouped)), 1)
        chunk = chunks_of(grouped)[0]
        text = ''.join(member["spoken_text"] for member in members)
        self.assertEqual(chunk["paragraph_turns"], members)
        self.assertEqual(chunk["segment"]["text"], text)
        self.assertEqual(chunk["spoken_text"], text)
        self.assertEqual((chunk["segment"]["start"], chunk["segment"]["end"]), (12, 12 + len(text)))
        self.assertEqual((chunk["spoken_start"], chunk["spoken_end"]), (0, len(text)))
        self.assertIsNone(chunk["segment"]["voice_id"])
        self.assertIsNone(chunk["segment"]["role_id"])
        self.assertEqual(chunk["segment"]["delivery"]["direction"], "")
        self.assertEqual(chunk["segment"]["break_after"], "paragraph")
        self.assertEqual(chunk["source_ranges"], [part for member in members for part in member["source_ranges"]])
        self.assertFalse(chunk["silent_control"])
        chunk["paragraph_turns"][0]["segment"]["text"] = "changed copy"
        self.assertEqual(source, original)

    def test_default_keeps_dialogue_narrator_dialogue_as_three_performances(self):
        members = passages(['"Wow, this is amazing", ', 'she said, ',
                            '"Why should we do that again?"'],
                           voices=["character", "narrator", "character"])
        source = plan(members)
        self.assertEqual(group_paragraphs(source), source)

    def test_shared_voice_does_not_merge_character_delivery_into_narration(self):
        members = passages(['"Wait," ', 'I said, ', '"come back."'])
        members[0]["segment"].update(role_id="character", delivery={"direction": "whispered", "evidence": []})
        members[1]["segment"].update(role_id="narrator", delivery={"direction": "", "evidence": []})
        members[2]["segment"].update(role_id="character", delivery={"direction": "whispered", "evidence": []})
        source = plan(members)
        saved = [[member["chunk_id"] for member in members]]
        self.assertEqual(group_paragraphs(source), source)
        self.assertEqual(group_paragraphs(source, preserved_groups=saved), source)

    def test_mixed_saved_group_cannot_override_default_voice_boundaries(self):
        members = passages(["One. ", "Two. ", "Three. ", "Four. ", "Five."],
                           voices=["character", "narrator", "character", "character", "character"])
        saved = [[member["chunk_id"] for member in members[:3]],
                 [member["chunk_id"] for member in members[3:]]]
        grouped = chunks_of(group_paragraphs(plan(members), preserved_groups=saved,
                                            preserved_chunk_ids={members[1]["chunk_id"]}))
        self.assertEqual(grouped[:3], members[:3])
        self.assertEqual(grouped[3]["paragraph_turns"], members[3:])
        restored = [turn for chunk in grouped for turn in chunk.get("paragraph_turns", [chunk])]
        self.assertEqual(restored, members)

    def test_source_blocks_chapters_and_explicit_blank_boundaries_never_join(self):
        first = passages(["One. ", "Two. "])
        second = passages(["Three. ", "Four."], block="b2")
        result = group_paragraphs(plan(first + second, first))
        self.assertEqual([len(chapter["chunks"]) for chapter in result["chapters"]], [2, 1])
        members = passages(["One. ", "Two. ", "Three. ", "Four."])
        members[2]["segment"]["break_before"] = "blank_paragraph"
        groups = chunks_of(group_paragraphs(plan(members)))
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[1]["segment"]["break_before"], "blank_paragraph")

    def test_long_paragraph_splits_at_original_chunk_boundaries(self):
        members = passages(["aaa ", "bbb ", "ccc ", "ddd "])
        groups = chunks_of(group_paragraphs(plan(members), max_chars=8))
        self.assertEqual([chunk["spoken_text"] for chunk in groups], ["aaa bbb ", "ccc ddd "])
        self.assertEqual(groups[0]["segment"]["break_after"], "none")
        self.assertEqual(groups[1]["segment"]["break_after"], "paragraph")
        self.assertEqual([turn for group in groups for turn in group["paragraph_turns"]], members)
        oversized = passages(["a" * 320] * 4)
        groups = chunks_of(group_paragraphs(plan(oversized)))
        self.assertEqual([len(chunk["spoken_text"]) for chunk in groups], [960, 320])
        self.assertEqual(groups[1], oversized[-1])

    def test_partial_chunks_of_one_source_row_do_not_duplicate_source_text(self):
        base = passages(["aaaaa bbbbb ccccc ddddd "], offset=20)[0]
        base["segment"]["break_before"] = "blank_paragraph"
        members = []
        for start in range(0, 24, 6):
            member = deepcopy(base)
            member.update(chunk_id=f"part-{start}", spoken_start=start, spoken_end=start + 6,
                          spoken_text=base["spoken_text"][start:start + 6],
                          source_ranges=[{"start": 20 + start, "end": 26 + start}])
            members.append(member)
        groups = chunks_of(group_paragraphs(plan(members), max_chars=12))
        self.assertEqual([chunk["segment"]["text"] for chunk in groups], ["aaaaa bbbbb ", "ccccc ddddd "])
        self.assertEqual([(chunk["segment"]["start"], chunk["segment"]["end"]) for chunk in groups],
                         [(20, 32), (32, 44)])
        self.assertEqual(groups[0]["segment"]["break_before"], "blank_paragraph")
        self.assertEqual(groups[0]["segment"]["break_after"], "none")
        self.assertNotIn("break_before", groups[1]["segment"])
        self.assertEqual(groups[1]["segment"]["break_after"], "paragraph")
        self.assertEqual(''.join(chunk["segment"]["text"] for chunk in groups), base["segment"]["text"])

    def test_any_accepted_original_clip_preserves_its_entire_paragraph(self):
        members = passages(["One. ", "Two. ", "Three."])
        source = plan(members + passages(["Next. ", "Last."], block="b2"))
        result = chunks_of(group_paragraphs(source, preserved_chunk_ids={members[1]["chunk_id"]}))
        self.assertEqual(result[:3], members)
        self.assertEqual(len(result[3]["paragraph_turns"]), 2)

    def test_accepted_group_id_is_reproduced_and_grouped_plan_is_idempotent(self):
        source = plan(passages(["One. ", "Two. ", "Three."]))
        grouped = group_paragraphs(source)
        group_id = chunks_of(grouped)[0]["chunk_id"]
        self.assertEqual(group_paragraphs(source, preserved_chunk_ids={group_id}), grouped)
        self.assertEqual(group_paragraphs(grouped), grouped)

    def test_saved_groups_survive_an_accepted_leftover_singleton(self):
        members = passages(["aaa ", "bbb ", "ccc ", "ddd ", "eee "])
        source = plan(members)
        initial = group_paragraphs(source, max_chars=8)
        saved_groups = [[turn["chunk_id"] for turn in chunk["paragraph_turns"]]
                        for chunk in chunks_of(initial) if "paragraph_turns" in chunk]
        resumed = group_paragraphs(source, max_chars=8, preserved_groups=saved_groups,
                                   preserved_chunk_ids={members[-1]["chunk_id"]})
        self.assertEqual(resumed, initial)
        # A smaller limit controls only newly synthesized groups. Existing
        # accepted groups remain identical and can reuse their exact recordings.
        self.assertEqual(group_paragraphs(source, max_chars=1, preserved_groups=saved_groups,
                                          preserved_chunk_ids={members[-1]["chunk_id"]}), initial)

    def test_saved_group_anchors_partition_legacy_preservation_into_gaps(self):
        members = passages(["aaa ", "bbb ", "ccc ", "ddd ", "eee ", "fff "])
        grouped = chunks_of(group_paragraphs(plan(members), preserved_groups=[
            [members[2]["chunk_id"], members[3]["chunk_id"]]],
            preserved_chunk_ids={members[0]["chunk_id"]}))
        self.assertEqual(grouped[:2], members[:2])
        self.assertEqual(grouped[2]["paragraph_turns"], members[2:4])
        self.assertEqual(grouped[3]["paragraph_turns"], members[4:])

    def test_saved_members_are_matched_exactly_and_rebuilt_from_current_metadata(self):
        members = passages(["aaa ", "bbb ", "ccc "])
        saved = [[members[0]["chunk_id"], members[1]["chunk_id"]]]
        members[1]["segment"]["delivery"]["direction"] = "new instruction"
        grouped = chunks_of(group_paragraphs(plan(members), max_chars=1, preserved_groups=saved))
        self.assertEqual(grouped[0]["paragraph_turns"][1]["segment"]["delivery"]["direction"],
                         "new instruction")
        members[1]["chunk_id"] = "changed-source-chunk"
        source = plan(members)
        self.assertEqual(group_paragraphs(source, max_chars=1, preserved_groups=saved), source)
        # IDs occurring elsewhere do not authorize nonconsecutive grouping.
        saved = [[members[0]["chunk_id"], members[2]["chunk_id"]]]
        self.assertEqual(group_paragraphs(source, max_chars=1, preserved_groups=saved), source)

    def test_overlapping_saved_groups_choose_source_order_then_longest(self):
        members = passages(["aaa ", "bbb ", "ccc ", "ddd "])
        ids = [chunk["chunk_id"] for chunk in members]
        candidates = [ids[:2], ids[1:], ids[:3]]
        first = group_paragraphs(plan(members), max_chars=1, preserved_groups=candidates)
        self.assertEqual(group_paragraphs(plan(members), max_chars=1,
                                          preserved_groups=list(reversed(candidates))), first)
        self.assertEqual(chunks_of(first)[0]["paragraph_turns"], members[:3])
        self.assertEqual(chunks_of(first)[1], members[3])

    def test_saved_groups_do_not_cross_source_boundaries_or_override_unsafe_content(self):
        members = passages(["aaa ", "bbb "])
        saved = [[member["chunk_id"] for member in members]]
        members[1]["segment"]["break_before"] = "blank_paragraph"
        source = plan(members)
        self.assertEqual(group_paragraphs(source, preserved_groups=saved), source)
        members[1]["segment"].pop("break_before")
        members[1]["silent_control"] = True
        source = plan(members)
        self.assertEqual(group_paragraphs(source, preserved_groups=saved), source)

    def test_bracketed_or_simultaneous_passage_keeps_whole_paragraph_separate(self):
        for extra in ({"performance": {"kind": "bracketed", "enter": True, "exit": True}},
                      {"simultaneous_speakers": [{"role_id": "r1", "voice_id": "v1"},
                                                {"role_id": "r2", "voice_id": "v2"}]}):
            with self.subTest(extra=extra):
                members = passages(["One. ", "Two. ", "Three."])
                members[1]["segment"].update(extra)
                original = plan(members)
                self.assertEqual(group_paragraphs(original), original)

    def test_controls_are_never_spoken_or_included_in_groups(self):
        members = passages(["One. ", "Two. ", "...", "Three. ", "Four."])
        members[2]["silent_control"] = True
        groups = chunks_of(group_paragraphs(plan(members)))
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[1], members[2])
        self.assertNotIn("paragraph_turns", groups[1])
        self.assertEqual([len(groups[i]["paragraph_turns"]) for i in (0, 2)], [2, 2])

    def test_speaker_limit_and_single_voice_engine_keep_valid_groups(self):
        members = passages([f"{i}. " for i in range(17)], voices=[f"v{i}" for i in range(17)])
        groups = chunks_of(group_paragraphs(plan(members), multi_speaker=True))
        self.assertEqual(len(groups[0]["paragraph_turns"]), 16)
        self.assertEqual(groups[1], members[-1])
        members = passages(["One. ", "Two. ", "Three. ", "Four."], voices=["v1", "v1", "v2", "v2"])
        groups = chunks_of(group_paragraphs(plan(members), multi_speaker=False))
        self.assertEqual(len(groups), 2)
        self.assertEqual([{member["segment"]["voice_id"] for member in group["paragraph_turns"]}
                          for group in groups], [{"v1"}, {"v2"}])

    def test_source_gaps_overlap_and_transformed_text_are_not_invented(self):
        for mutate in (lambda chunk: chunk["source_ranges"][0].update(start=6),
                       lambda chunk: chunk["source_ranges"][0].update(start=4),
                       lambda chunk: chunk.update(spoken_text="Other")):
            members = passages(["One. ", "Two."])
            mutate(members[1])
            original = plan(members)
            self.assertEqual(group_paragraphs(original), original)

    def test_single_chunk_and_oversized_individual_chunk_are_unchanged(self):
        source = plan(passages(["Already complete paragraph."]))
        self.assertEqual(group_paragraphs(source), source)
        self.assertEqual(group_paragraphs(source, max_chars=1), source)

    def test_invalid_options_are_rejected(self):
        for value in (0, -1, True, 1000.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_chars"):
                group_paragraphs(plan([]), max_chars=value)
        with self.assertRaisesRegex(ValueError, "multi_speaker"):
            group_paragraphs(plan([]), multi_speaker=1)
        for value in ([[]], [["one"]], [["one", "one"]], [["one", 2]], ["one two"]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "preserved_groups"):
                group_paragraphs(plan([]), preserved_groups=value)


if __name__ == "__main__":
    unittest.main()
