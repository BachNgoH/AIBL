"""Analytic regression tests for the paper's AIBL objective."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aibl.aibl_loss import (adapt_weight_map_to_sequence, compute_aibl_loss,
                                compute_instance_weight_map)


class AIBLTests(unittest.TestCase):
    def setUp(self):
        # Two visible pixels for object 0; one for occluded object 1; one background.
        self.owners = np.array([[0, 0], [1, -1]], dtype=np.int32)
        self.masks = [np.array([[1, 1], [0, 0]], dtype=np.float32),
                      np.ones((2, 2), dtype=np.float32)]

    def test_formula_and_gradient(self):
        weights, _ = compute_instance_weight_map(self.owners, self.masks, 2, 2)
        raw = torch.tensor([[1., 1.], [2., .8]])
        expected = raw / raw.mean()
        torch.testing.assert_close(weights, expected)
        pred = torch.tensor([[[1.], [2.], [3.], [4.]]], requires_grad=True)
        loss, _ = compute_aibl_loss(pred, torch.zeros_like(pred),
                                   [self.owners], [self.masks], 2, 2)
        torch.testing.assert_close(loss, (pred.flatten().square() * expected.flatten()).mean())
        loss.backward()
        torch.testing.assert_close(pred.grad.flatten(),
                                   2 * pred.detach().flatten() * expected.flatten() / 4)

    def test_boost_clip_and_gamma_zero(self):
        _, stats = compute_instance_weight_map(self.owners, self.masks, 2, 2,
                                              gamma=1., alpha_max=2.)
        self.assertEqual(stats["boosts"], [1., 2.])
        weight, _ = compute_instance_weight_map(self.owners, self.masks, 2, 2, gamma=0)
        raw = torch.tensor([[1., 1.], [1., .8]])
        torch.testing.assert_close(weight, raw / raw.mean())

    def test_batch_normalization_and_uniform_ablation(self):
        pred = torch.arange(1., 9.).reshape(2, 4, 1)
        loss, _ = compute_aibl_loss(pred, torch.zeros_like(pred),
                                   [self.owners, self.owners], [self.masks, self.masks], 2, 2)
        weights, _ = compute_instance_weight_map(self.owners, self.masks, 2, 2)
        torch.testing.assert_close(loss, (pred.squeeze(-1).square() * weights.flatten()).mean())
        loss, _ = compute_aibl_loss(pred, torch.zeros_like(pred), [], [], 2, 2, mix_ratio=0)
        torch.testing.assert_close(loss, pred.square().mean())

    def test_empty_scene_is_uniform(self):
        owners = np.full((1, 1), -1, dtype=np.int32)
        weight, _ = compute_instance_weight_map(owners, [], 1, 1)
        torch.testing.assert_close(weight, torch.ones(1, 1))

    def test_sequence_packing_and_non_square_expansion(self):
        weight = torch.arange(1., 17.).reshape(4, 4)
        packed = adapt_weight_map_to_sequence(weight, 4, 4, 4, "cpu", torch.float32)
        expected = torch.tensor([3.5, 5.5, 11.5, 13.5])
        torch.testing.assert_close(packed, expected / expected.mean())
        expanded = adapt_weight_map_to_sequence(torch.ones(2, 2), 6, 2, 2, "cpu", torch.float32)
        self.assertEqual(expanded.numel(), 6)

    def test_ablation_modes_preserved(self):
        collapsed, _ = compute_instance_weight_map(self.owners, self.masks, 2, 2,
                                                   weight_mode="collapsed")
        self.assertEqual(collapsed[0, 0], collapsed[1, 0])
        raw, stats = compute_instance_weight_map(self.owners, self.masks, 2, 2,
                                                 norm_mode="dataset_global")
        self.assertAlmostEqual(raw.mean().item(), stats["raw_mean_weight"])

    def test_invalid_supervision_fails(self):
        for kwargs in ({"weight_mode": "oops"}, {"lambda_bg": 0},
                       {"gamma": float("nan")}, {"norm_mode": "dataset_global", "mu_global": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_instance_weight_map(self.owners, self.masks, 2, 2, **kwargs)
        with self.assertRaises(ValueError):
            compute_instance_weight_map(self.owners, [], 2, 2)
        with self.assertRaises(ValueError):
            compute_aibl_loss(torch.ones(2, 4, 1), torch.zeros(2, 4, 1),
                              [self.owners], [self.masks], 2, 2)


if __name__ == "__main__":
    unittest.main()
