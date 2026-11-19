"""Lane recipes place verified takes on one timeline without changing them."""

from array import array
import math
import unittest

from epub_tts_narrator.renderer.lane_mix import (MAX_LANES, RECIPES, gated_rms, mix_arrays, onset_frame,
                                                 plan_lanes)

RATE = 8000


def tone(frequency, seconds, *, lead=0.0, amplitude=6000):
    silent = [0] * int(lead * RATE)
    voiced = [round(amplitude * math.sin(2 * math.pi * frequency * index / RATE))
              for index in range(int(seconds * RATE))]
    return array("h", silent + voiced)


def lanes(count):
    return [{"audio_sha256": f"{index:064x}"} for index in range(count)]


def decibels(ratio):
    return 20 * math.log10(ratio)


class LaneMixTests(unittest.TestCase):
    def test_first_speech_is_aligned_so_a_long_lead_in_is_not_heard_late(self):
        sources = [tone(220, 0.5), tone(330, 0.5, lead=0.2)]
        plan = plan_lanes(lanes(2), sources, "tight", RATE)
        starts = [offset + onset_frame(source, RATE) for offset, source in zip(plan["offsets"], sources)]
        self.assertLessEqual(plan["offsets"][1], 0.015 * RATE, "the late lane only gets the unison spread")
        self.assertGreater(plan["offsets"][0], 0.19 * RATE)
        self.assertLessEqual(abs(starts[0] - starts[1]), 0.015 * RATE)

    def test_unison_is_as_loud_as_its_lanes_not_quieter_per_voice(self):
        for count in (2, 4, 8):
            sources = [tone(200 + 37 * index, 1.0) for index in range(count)]
            plan = plan_lanes(lanes(count), sources, "loose", RATE)
            mixed, scale, peak = mix_arrays(sources, plan, RATE)
            loss = decibels(gated_rms(mixed, RATE) / gated_rms(sources[0], RATE))
            with self.subTest(count=count):
                self.assertLess(abs(loss), 1.0)
                self.assertLessEqual(peak, 32767)

    def test_peak_ceiling_wins_over_loudness(self):
        sources = [array("h", [30000] * RATE), array("h", [30000] * RATE)]
        plan = plan_lanes(lanes(2), sources, "tight", RATE)
        _, scale, peak = mix_arrays(sources, plan, RATE)
        self.assertLessEqual(peak, 31785)
        self.assertLess(scale, 0.53)

    def test_staggered_recipes_enter_one_after_another(self):
        sources = [tone(200 + 50 * index, 0.5) for index in range(4)]
        for name, low, high in (("swell", 0.25, 0.28), ("babble", 0.15, 0.4), ("echo", 0.3, 0.3)):
            plan = plan_lanes(lanes(4), sources, name, RATE)
            steps = [b - a for a, b in zip(plan["offsets"], plan["offsets"][1:])]
            with self.subTest(recipe=name):
                self.assertTrue(all(low * RATE - 1 <= step <= high * RATE + 1 for step in steps), steps)
        echo = plan_lanes(lanes(3), sources[:3], "echo", RATE)
        self.assertEqual([round(decibels(gain)) for gain in echo["gains"]], [0, -4, -8])
        self.assertEqual([item and item["lowpass_hz"] for item in echo["filters"]], [None, 3200, 1600])

    def test_layered_lead_stays_clear_and_the_others_sit_just_behind_it(self):
        sources = [tone(200 + 50 * index, 0.5) for index in range(5)]
        plan = plan_lanes(lanes(5), sources, "lead", RATE)
        self.assertEqual(plan["offsets"][0], 0)
        self.assertTrue(all(0.02 * RATE <= offset <= 0.06 * RATE for offset in plan["offsets"][1:]))
        self.assertEqual([round(decibels(gain)) for gain in plan["gains"]], [0, -12, -12, -12, -12])

    def test_plans_are_deterministic_and_bound_to_the_verified_takes(self):
        sources = [tone(220, 0.5), tone(330, 0.5)]
        self.assertEqual(plan_lanes(lanes(2), sources, "crowd", RATE), plan_lanes(lanes(2), sources, "crowd", RATE))
        other = [{"audio_sha256": "f" * 64}, {"audio_sha256": "e" * 64}]
        self.assertNotEqual(plan_lanes(lanes(2), sources, "crowd", RATE)["offsets"],
                            plan_lanes(other, sources, "crowd", RATE)["offsets"])

    def test_a_recorded_scale_reproduces_the_same_mix(self):
        sources = [tone(220, 0.5), tone(330, 0.5)]
        plan = plan_lanes(lanes(2), sources, "babble", RATE)
        mixed, scale, _ = mix_arrays(sources, plan, RATE)
        again, same, _ = mix_arrays(sources, plan, RATE, scale=scale)
        self.assertEqual((again, same), (mixed, scale))

    def test_recipes_and_lane_counts_are_a_closed_set(self):
        sources = [tone(220, 0.1)] * 2
        with self.assertRaisesRegex(ValueError, "Unknown lane mix recipe"):
            plan_lanes(lanes(2), sources, "stadium", RATE)
        with self.assertRaisesRegex(ValueError, "2 to 8"):
            plan_lanes(lanes(MAX_LANES + 1), [tone(220, 0.1)] * (MAX_LANES + 1), "crowd", RATE)
        self.assertEqual(set(RECIPES), {"tight", "loose", "crowd", "swell", "babble", "echo", "lead"})


if __name__ == "__main__":
    unittest.main()
