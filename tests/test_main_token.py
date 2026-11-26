"""The open main-token draw for narration (main_token.py), on MLX's CPU device."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mlx.core as mx

from scripts.check_drama3_mlx_batching import random_prompt, tiny_model
from epub_tts_narrator.renderer import main_token
from epub_tts_narrator.renderer.engines import Drama3MLXEngine
from epub_tts_narrator.renderer.mlx_drama3_decode import generate_frames_watched
from epub_tts_narrator.renderer.request_sampling import request_sampling


def draws(probs, count=4000, temperature=1.0, seed=1):
    mx.random.seed(seed)
    logits = mx.log(mx.array([probs] * count, dtype=mx.float32))
    return set(np.array(main_token.sample_open(logits, temperature)).tolist())


class OpenDrawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def test_the_token_crossing_top_p_is_kept_and_improbable_ones_dropped(self):
        # Cumulative .6, .9, .98: the vendored rule drops token 2, the open rule keeps it;
        # token 3 starts after .95 and token 4 is under the 0.001 floor.
        seen = draws([0.6, 0.3, 0.08, 0.0195, 0.0005])
        self.assertEqual(seen, {0, 1, 2})

    def test_there_is_no_rank_cap(self):
        # 50 equal tokens: ranks 0-47 start before .95 (the vendored top-k would keep 30).
        self.assertEqual(draws([1 / 50] * 50, count=20000), set(range(48)))

    def test_the_top_token_is_always_kept(self):
        self.assertEqual(draws([0.9995, 0.0005]), {0})
        self.assertEqual(draws([0.97, 0.03]), {0})  # token 1 starts after .95
        self.assertEqual(draws([0.9, 0.1]), {0, 1})


class DecodeStepTests(unittest.TestCase):
    """Only the main-token draw differs from the vendored step."""

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)
        cls.model = tiny_model()
        cls.prompt = random_prompt(cls.model, 40, np.random.default_rng(3))

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def frames(self, **options):
        mx.random.seed(11)
        frames, _finish = generate_frames_watched(self.model, self.prompt, max_new_tokens=12,
                                                  temperature=0.7, top_k=30, top_p=0.7, **options)
        return frames

    def test_with_the_vendored_draw_in_its_place_the_step_is_the_vendored_one(self):
        from drama3_mlx.sampling import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, sample_top_k_top_p
        calls = []

        def vendored(logits, temperature):
            # The open step draws first then retry, as the vendored step does.
            calls.append(temperature)
            if len(calls) % 2:
                return sample_top_k_top_p(logits, 0.7, 30, 0.7)
            return sample_top_k_top_p(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)

        with patch.object(main_token, "sample_open", vendored):
            ours = self.frames(main_token=main_token.RULE)
        np.testing.assert_array_equal(ours, self.frames())
        self.assertTrue(calls)

    def test_the_open_draw_changes_the_frames_and_unknown_rules_are_refused(self):
        self.assertEqual(self.frames(main_token=main_token.RULE).shape[1], self.frames().shape[1])
        with self.assertRaisesRegex(ValueError, "Unknown main-token rule"):
            self.frames(main_token="hot")


class EngineTests(unittest.TestCase):
    def test_only_the_single_request_loop_offers_the_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Drama3MLXEngine(Path(folder))
        self.assertEqual(engine.main_token_rules, {main_token.RULE})
        engine.batch_rows = 2
        self.assertEqual(engine.main_token_rules, frozenset())

    def test_a_request_names_the_rule_it_asks_for(self):
        self.assertIsNone(main_token.requested())
        with request_sampling({"temperature": 0.95}):
            self.assertIsNone(main_token.requested())
        with request_sampling({"main_token": main_token.RULE}):
            self.assertEqual(main_token.requested(), main_token.RULE)
        with request_sampling({"main_token": "hot"}), self.assertRaisesRegex(ValueError, "Unknown"):
            main_token.requested()


if __name__ == "__main__":
    unittest.main()
