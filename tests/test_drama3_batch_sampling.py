"""Counter-hash sampling noise of the batched Drama 3 decoder (torch on CPU, no Fish)."""

import unittest

import torch

from epub_tts_narrator.renderer.drama3_batch import exponential_noise, sampling_noise


class SamplingNoiseTests(unittest.TestCase):
    def test_every_hash_value_gives_positive_finite_noise(self):
        # The extremes of the hash: all bits set (-1 as int64), none set, and the
        # values whose float32 conversion once rounded the uniform up to 1.0, which
        # made the noise 0 and let argmax pick a dropped token through 0 / 0 = NaN.
        hashes = torch.tensor([-1, 0, (1 << 63) - 1, -(1 << 63), -(1 << 40), (1 << 40) - 1],
                              dtype=torch.int64)
        noise = exponential_noise(hashes)
        self.assertTrue(torch.isfinite(noise).all())
        self.assertTrue((noise > 0).all(), noise.tolist())

    def test_a_dropped_token_is_never_chosen(self):
        # Sampling divides kept probabilities by the noise; dropped tokens have
        # probability 0 and must stay at 0 whatever noise they draw.
        noise = exponential_noise(torch.tensor([[-1, 0, 12345]], dtype=torch.int64))
        probabilities = torch.tensor([[0.0, 0.6, 0.4]])
        self.assertNotEqual(int((probabilities / noise).argmax(-1)), 0)

    def test_stream_is_exponential_and_has_no_zero(self):
        seeds = torch.arange(256, dtype=torch.int64)
        noise = sampling_noise(seeds, torch.zeros_like(seeds), 0, 4097)
        self.assertTrue((noise > 0).all())
        self.assertAlmostEqual(float(noise.mean()), 1.0, delta=0.01)
        self.assertAlmostEqual(float((noise > 1).float().mean()), float(torch.exp(torch.tensor(-1.0))), delta=0.005)

    def test_noise_depends_only_on_seed_step_and_draw(self):
        seeds = torch.tensor([7, 7, 8], dtype=torch.int64)
        steps = torch.tensor([3, 3, 3], dtype=torch.int64)
        noise = sampling_noise(seeds, steps, 2, 64)
        self.assertTrue(torch.equal(noise[0], noise[1]))
        self.assertFalse(torch.equal(noise[0], noise[2]))
        self.assertFalse(torch.equal(noise[0], sampling_noise(seeds, steps, 3, 64)[0]))
        self.assertFalse(torch.equal(noise[0], sampling_noise(seeds, steps + 1, 2, 64)[0]))


if __name__ == "__main__":
    unittest.main()
