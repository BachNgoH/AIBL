"""CPU checks for training schedules, accumulation, batches and safe run metadata."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
from aibl.runtime import get_lr_scheduler, image_to_tensor, save_run_config, select_device_and_dtype
from aibl.aibl_loss import compute_aibl_loss


class TrainingRuntimeTests(unittest.TestCase):
    def test_cpu_auto_precision_is_fp32(self):
        device, dtype = select_device_and_dtype("cpu", "auto")
        self.assertEqual(device, "cpu")
        self.assertEqual(dtype, torch.float32)
        with self.assertRaises(ValueError):
            select_device_and_dtype("mps", "auto")

    def test_scheduler_reaches_floor_after_3000_optimizer_updates(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=5e-5)
        args = argparse.Namespace(warmup_steps=100, lr_scheduler="cosine",
                                  min_lr=1e-6, lr=5e-5)
        scheduler = get_lr_scheduler(optimizer, args, 3000)
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(3000):
            # Actual scalar training loss and four microbatches per update.
            optimizer.zero_grad(set_to_none=True)
            for _ in range(4):
                (parameter.square() / 4).backward()
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertEqual(scheduler.last_epoch, 3000)
        self.assertAlmostEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[100], args.lr)
        self.assertAlmostEqual(rates[-1], args.min_lr)
        self.assertTrue(all(a >= b for a, b in zip(rates[100:], rates[101:])))
        self.assertLess(parameter.item(), 1.0)

    def test_accumulated_aibl_gradients_match_true_batch(self):
        inputs = torch.arange(1.0, 17.0).reshape(2, 4, 2) / 10
        owners = [np.array([[0, 0], [1, -1]], dtype=np.int32),
                  np.array([[-1, 0], [0, 0]], dtype=np.int32)]
        masks = [[np.array([[1, 1], [0, 0]], dtype=np.float32),
                  np.ones((2, 2), dtype=np.float32)],
                 [np.ones((2, 2), dtype=np.float32)]]
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.2, -0.1]]))
        prediction = model(inputs)
        loss, _ = compute_aibl_loss(prediction, torch.zeros_like(prediction),
                                   owners, masks, 2, 2)
        loss.backward()
        batch_gradient = model.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        for index in range(2):
            prediction = model(inputs[index:index + 1])
            loss, _ = compute_aibl_loss(prediction, torch.zeros_like(prediction),
                                       [owners[index]], [masks[index]], 2, 2)
            (loss / 2).backward()
        torch.testing.assert_close(model.weight.grad, batch_gradient)

    def test_saved_configuration_omits_authentication(self):
        args = argparse.Namespace(hf_token="unit-test-authentication-value",
                                  seed=42, steps=3000, mix_ratio=1.0,
                                  use_overlap_ownership=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint"
            save_run_config(args, output)
            content = (output / "config.json").read_text()
            config = json.loads(content)
        self.assertNotIn("hf_token", config)
        self.assertNotIn(args.hf_token, content)
        self.assertEqual(config["steps"], 3000)
        self.assertEqual(config["seed"], 42)
        self.assertTrue(config["use_overlap_ownership"])

    def test_image_batch_preserves_samples_and_normalizes(self):
        black = np.zeros((3, 5, 3), dtype=np.uint8)
        white = np.full((4, 2, 3), 255, dtype=np.uint8)
        images = image_to_tensor([black, white], 8, "cpu")
        self.assertEqual(images.shape, (2, 3, 8, 8))
        self.assertEqual(images.dtype, torch.float32)
        torch.testing.assert_close(images[0], -torch.ones(3, 8, 8))
        torch.testing.assert_close(images[1], torch.ones(3, 8, 8))


if __name__ == "__main__":
    unittest.main()
