"""CPU-only checks for the opt-in Drama 3 sampling proxy, never importing MLX."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from audiobook_renderer.drama3_sampling import wrap_model

try:
    import numpy as np
except ImportError:
    np = None


class SamplingProxyContractTests(unittest.TestCase):
    def test_wrap_delegates_without_mutating_model_or_loading_runtime(self):
        tower = SimpleNamespace(forced=object(), make_caches=object(), embed_backbone=object())
        model = SimpleNamespace(tower=tower, backbone=object(), config=object())
        original = vars(model).copy()
        with patch("audiobook_renderer.drama3_sampling._runtime") as runtime:
            proxy = wrap_model(model)
            self.assertIs(proxy.backbone, model.backbone)
            self.assertIs(proxy.config, model.config)
            self.assertIs(proxy.tower.forced, tower.forced)
            self.assertIs(proxy.tower.make_caches, tower.make_caches)
            self.assertIs(proxy.tower.embed_backbone, tower.embed_backbone)
            self.assertIsNot(proxy.tower, tower)
            runtime.assert_not_called()
        self.assertEqual(vars(model), original)

    def test_invalid_caps_are_rejected_without_loading_runtime(self):
        model = SimpleNamespace(tower=object())
        for cap in (False, True, 0, -1, 1.5, "30"):
            for name in ("main_top_k", "codebook_top_k"):
                with self.subTest(name=name, cap=cap), self.assertRaises(ValueError):
                    wrap_model(model, **{name: cap})
        with self.assertRaises(ValueError):
            wrap_model(model, codebook_top_k=None)


@unittest.skipIf(np is None, "Numerical proxy tests run in the renderer environment with numpy")
class SamplingProxyNumericalTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.draws = [np.array([2]), np.array([3]), np.array([7]), np.array([8])]
        self.sampling = SimpleNamespace(
            sample_top_k_top_p=self.sample,
            RAS_TEMPERATURE=1.0, RAS_TOP_K=30, RAS_TOP_P=.95, RAS_WINDOW_FRAMES=16,
        )
        self.tower = SimpleNamespace(
            _main_candidates=np.array([0, *range(10, 74)]),
            semantic_begin=10, cfg=SimpleNamespace(vocab_size=64),
            is_semantic=lambda value: (value >= 10) & (value <= 73),
            shared_layers=[object()],
            run_shared=lambda hidden, *args: hidden,
            residual_layers=[lambda hidden, *args: hidden, lambda hidden, *args: hidden],
            codebook_heads=[lambda hidden: np.zeros((1, 64)), lambda hidden: np.ones((1, 64))],
        )
        self.model = SimpleNamespace(tower=self.tower, backbone=object(), config=object())
        self.args = (np.zeros((1, 80)), np.zeros((1, 8)), np.zeros((1, 4)), np.array([11]), [0, 1, 2])

    def sample(self, logits, temperature, top_k, top_p):
        self.calls.append((logits.copy(), temperature, top_k, top_p))
        return self.draws[len(self.calls) - 1]

    def decode(self, main=None, codebooks=30, top_k=30, args=None):
        proxy = wrap_model(self.model, main_top_k=main, codebook_top_k=codebooks)
        with patch("audiobook_renderer.drama3_sampling._runtime", return_value=(np, self.sampling)):
            return proxy.tower.decode_step(*(args or self.args), temperature=.7, top_k=top_k, top_p=.8)

    def test_uncapped_main_fixed_ras_and_residual_caps(self):
        actual = self.decode(codebooks=100, top_k=100)
        self.assertEqual([call[2] for call in self.calls], [65, 30, 30, 30])
        self.assertEqual([call[1:] for call in self.calls], [(.7, 65, .8), (1., 30, .95), (.7, 30, .8), (.7, 30, .8)])
        # First draw is semantic token 11, repeated in history, so RAS token 12 wins.
        np.testing.assert_array_equal(actual, [[12, 2, 7, 8]])

    def test_request_can_lower_residual_cap(self):
        self.decode(main=48, codebooks=20, top_k=12)
        self.assertEqual([call[2] for call in self.calls], [48, 30, 12, 12])

    def test_ras_is_drawn_but_does_not_replace_eos(self):
        self.draws[0] = np.array([0])
        args = (*self.args[:3], np.array([0]), self.args[4])
        actual = self.decode(args=args)
        self.assertEqual(len(self.calls), 4)
        np.testing.assert_array_equal(actual, [[0, 0, 7, 8]])

    def test_only_last_sixteen_frames_trigger_ras(self):
        args = (*self.args[:3], np.array([11, *([20] * 16)]), self.args[4])
        actual = self.decode(args=args)
        np.testing.assert_array_equal(actual, [[11, 1, 7, 8]])

    def test_caps_of_thirty_match_pinned_runtime_operations_and_draws(self):
        # Extract just the installed method's AST; importing its module would
        # initialize MLX. This compares the actual pinned implementation on CPU.
        root = Path(__file__).resolve().parents[1]
        source = root / ".venv-renderer/lib/python3.12/site-packages/drama3_mlx/tower.py"
        if not source.exists():
            self.skipTest("Pinned runtime source is not installed here")
        parsed = ast.parse(source.read_text())
        cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "CodebookTower")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "decode_step")
        for argument in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
            argument.annotation = None
        method.returns = None
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        scope = {"mx": np, "sample_top_k_top_p": self.sample,
                 "RAS_TEMPERATURE": 1., "RAS_TOP_K": 30,
                 "RAS_TOP_P": .95, "RAS_WINDOW_FRAMES": 16}
        exec(compile(module, str(source), "exec"), scope)
        expected = scope["decode_step"](self.tower, *self.args, temperature=.7, top_k=30, top_p=.8)
        expected_calls = self.calls[:]
        self.calls.clear()
        actual = self.decode(main=30, codebooks=30)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(self.calls), len(expected_calls))
        for actual_call, expected_call in zip(self.calls, expected_calls):
            np.testing.assert_array_equal(actual_call[0], expected_call[0])
            self.assertEqual(actual_call[1:], expected_call[1:])


if __name__ == "__main__":
    unittest.main()
