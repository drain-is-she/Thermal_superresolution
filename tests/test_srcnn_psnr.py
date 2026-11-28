"""Numerical checks for PSNR with a fixed unit data range."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.srcnn_l1_inference import calculate_psnr, calculate_ssim


class PSNRTests(unittest.TestCase):
    def test_identical_images_have_positive_infinite_psnr(self):
        image = torch.tensor([0.0, 0.25, 0.5, 1.0])
        score = torch.as_tensor(calculate_psnr(image, image))
        self.assertTrue(torch.isposinf(score).item())

    def test_known_errors_for_unit_data_range(self):
        reference = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        for error, expected in [(1.0, 0.0), (0.1, 20.0), (0.01, 40.0)]:
            with self.subTest(error=error):
                score = calculate_psnr(reference + error, reference)
                self.assertAlmostEqual(float(score), expected, places=10)

    def test_both_metrics_reject_broadcastable_shape_mismatches(self):
        prediction = torch.zeros(1, 1, 2, 2)
        targets = [torch.zeros(1, 1, 2, 1), torch.zeros(1, 2, 2), torch.zeros(1, 1, 1, 2)]
        for target in targets:
            for metric in (calculate_psnr, calculate_ssim):
                with self.subTest(metric=metric.__name__, target_shape=target.shape):
                    with self.assertRaisesRegex(ValueError, "shapes must match"):
                        metric(prediction, target)

    def test_perfect_match_ranks_above_tiny_nonzero_error(self):
        reference = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        imperfect = calculate_psnr(reference + 1e-6, reference)
        perfect = calculate_psnr(reference, reference)
        self.assertGreater(float(imperfect), 100.0)
        self.assertGreater(float(perfect), float(imperfect))


if __name__ == '__main__':
    unittest.main()
