"""Equation-level bias checks and real FLUX.2 processor integration, on CPU."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aibl.layout_attention import build_layout_attention_bias, compute_identity_and_spatial_heatmaps
from aibl.inference_config import parse_args, resolve_config, split_indices


class LayoutAttentionTests(unittest.TestCase):
    def setUp(self):
        self.instances = [
            {"bbox": [0, 0, 0.6, 1], "depth": 0.9},
            {"bbox": [0.2, 0, 0.6, 1], "depth": 0.1},
        ]

    def bias(self, **kwargs):
        return build_layout_attention_bias(self.instances, 1, 5, 2, 1, "cpu", torch.float32, **kwargs)

    def test_paper_corrections_require_both_contested_tokens(self):
        difference = (self.bias() - self.bias(use_overlap_ownership=False))[0, 2:, 2:]
        self.assertAlmostEqual(difference[1, 2].item(), 2.0, places=5)
        self.assertAlmostEqual(difference[0, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(difference[0, 0].item(), 0.0, places=5)
        # Separation propagates across the entire winning region, including
        # its exclusive token at 0, to the back object's exclusive token at 3.
        self.assertAlmostEqual(difference[0, 3].item(), -0.5, places=5)
        self.assertAlmostEqual(difference[1, 3].item(), -0.5, places=5)
        torch.testing.assert_close(difference[4], torch.zeros(5))
        torch.testing.assert_close(difference[:, 4], torch.zeros(5))

    def test_text_tokens_unchanged_and_depth_owner_frontmost(self):
        mask = self.bias()[0]
        torch.testing.assert_close(mask[:2], torch.zeros(2, 7))
        torch.testing.assert_close(mask[:, :2], torch.zeros(7, 2))
        ownership, _ = compute_identity_and_spatial_heatmaps(self.instances, 1, 5)
        self.assertEqual(ownership.tolist(), [[1, 1, 1, 2, 0]])
        reverse_owners, _ = compute_identity_and_spatial_heatmaps(self.instances[::-1], 1, 5)
        self.assertEqual(reverse_owners.tolist(), [[2, 2, 2, 1, 0]])

    def test_single_box_receives_gaussian_prior(self):
        result = build_layout_attention_bias(
            [{"bbox": [0, 0, 0.5, 1]}], 1, 4, 0, 1, "cpu", torch.float32
        )[0]
        self.assertGreater(result[0, 0].item(), 4.0)
        self.assertGreater(result[0, 3].item(), 0.0)
        torch.testing.assert_close(result[3], torch.zeros(4))

    def test_no_instances_no_bias(self):
        self.assertIsNone(build_layout_attention_bias([], 2, 2, 2, 1, "cpu", torch.float32))

    def test_real_flux_dual_and_single_attention_accept_batch_head_axis(self):
        from diffusers import Flux2Transformer2DModel

        torch.manual_seed(42)
        model = Flux2Transformer2DModel(
            in_channels=8, num_layers=1, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=12, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), mlp_ratio=2,
            guidance_embeds=False,
        )
        mask = torch.cat([self.bias(), self.bias(use_overlap_ownership=False)], dim=0).unsqueeze(1)
        observed = []

        def capture_mask(module, args, kwargs):
            observed.append(kwargs.get("attention_mask"))

        handles = [block.attn.register_forward_pre_hook(capture_mask, with_kwargs=True)
                   for block in [model.transformer_blocks[0], model.single_transformer_blocks[0]]]
        kwargs = dict(
            hidden_states=torch.randn(2, 5, 8), encoder_hidden_states=torch.randn(2, 2, 12),
            timestep=torch.full((2,), 0.5),
            img_ids=torch.zeros(2, 5, 4), txt_ids=torch.zeros(2, 2, 4),
        )
        output = model(**kwargs, joint_attention_kwargs={"attention_mask": mask}).sample
        self.assertEqual(tuple(output.shape), (2, 5, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(value is mask for value in observed))
        for handle in handles:
            handle.remove()
        unmasked = model(**kwargs).sample
        self.assertFalse(torch.allclose(output, unmasked))
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))


class InferenceConfigurationTests(unittest.TestCase):
    def test_saved_config_used_and_explicit_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "lora_final"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({"omega_intra": 3, "use_overlap_ownership": False}))
            args = parse_args(["--data_root", directory, "--lora_path", str(checkpoint),
                               "--output_dir", directory, "--omega_intra", "4"])
            config = resolve_config(args)
            self.assertEqual(config["omega_intra"], 4)
            self.assertFalse(config["use_overlap_ownership"])
            self.assertEqual(config["beta"], 2.0)

    def test_explicit_paper_split_bounds(self):
        args = parse_args(["--data_root", ".", "--lora_path", ".", "--output_dir", ".",
                           "--start_idx", "44000", "--end_idx", "45000"])
        self.assertEqual(split_indices(args), (44000, 45000))


if __name__ == "__main__":
    unittest.main()
