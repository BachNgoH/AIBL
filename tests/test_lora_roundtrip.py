"""Exercise production LoRA targets, checkpointing, and adapter loading on CPU."""

import tempfile
import unittest

import torch
from diffusers import Flux2Transformer2DModel
from peft import LoraConfig, PeftModel, get_peft_model


def tiny_backbone():
    torch.manual_seed(2026)
    return Flux2Transformer2DModel(
        in_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2,
        joint_attention_dim=12, timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2), mlp_ratio=2, guidance_embeds=False,
    )


class LoraRoundtripTests(unittest.TestCase):
    def test_checkpointed_adapter_trains_and_reload_matches(self):
        backbone = tiny_backbone()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        backbone.enable_gradient_checkpointing()
        model = get_peft_model(backbone, LoraConfig(
            r=2, lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "to_qkv_mlp_proj"],
        ))
        model.train()
        kwargs = dict(
            hidden_states=torch.randn(2, 5, 8),
            encoder_hidden_states=torch.randn(2, 2, 12),
            timestep=torch.full((2,), 0.5),
            img_ids=torch.zeros(2, 5, 4), txt_ids=torch.zeros(2, 2, 4),
            joint_attention_kwargs={"attention_mask": torch.randn(2, 1, 7, 7)},
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=0.001)
        before = [parameter.detach().clone() for parameter in trainable]
        model(**kwargs).sample.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in trainable))
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, parameter) for old, parameter in zip(before, trainable)))
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()
                            if not parameter.requires_grad))
        model.eval()
        with torch.no_grad():
            expected = model(**kwargs).sample
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = PeftModel.from_pretrained(tiny_backbone(), directory, adapter_name="aibl")
            loaded.set_adapter("aibl")
            loaded.eval()
            self.assertEqual(loaded.config.in_channels, 8)
            with torch.no_grad():
                actual = loaded(**kwargs).sample
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
