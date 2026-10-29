"""Train Layout Attention (LA) with Amodal-Aware Instance-Balanced Loss (AIBL)."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aibl.aibl_loss import compute_aibl_loss
from aibl.dataset import OverlapDepth45KDataset, collate_fn
from aibl.runtime import (build_batch_layout_bias, get_lr_scheduler, image_to_tensor,
                          save_run_config, select_device_and_dtype)
from aibl.settings import ATTENTION_DEFAULTS, LORA_TARGETS
from aibl.train_config import parse_args


def _encode_batch(pipe, batch, args, dtype, micro_step):
    """Use the backbone's native prompt and VAE encoders without gradients."""
    device = args.device
    images = image_to_tensor(batch["image"], args.image_size, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + micro_step)
    with torch.no_grad():
        if args.sequential_component_offload:
            pipe.text_encoder.to(device)
        embeddings, text_ids = pipe.encode_prompt(batch["prompt"], device=device,
                                                  max_sequence_length=args.max_sequence_length)
        if args.sequential_component_offload:
            pipe.text_encoder.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            pipe.vae.to(device)
        spatial = torch.cat([pipe._encode_vae_image(image[None].to(dtype), generator)
                             for image in images])
        latents, image_ids = pipe.prepare_latents(
            batch_size=len(images), num_latents_channels=pipe.transformer.config.in_channels // 4,
            height=args.image_size, width=args.image_size, dtype=dtype, device=device,
            generator=generator, latents=spatial)
        if args.sequential_component_offload:
            pipe.vae.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return latents, image_ids, embeddings, text_ids


def main(argv=None):
    args = parse_args(argv)
    args.device, dtype = select_device_and_dtype(args.device, args.precision)
    args.precision = "bf16" if dtype == torch.bfloat16 else "fp32"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = OverlapDepth45KDataset(args.data_root, load_masks=args.use_aibl,
                                     start_idx=args.train_start_idx, end_idx=args.train_end_idx)
    if len(dataset) != args.train_end_idx - args.train_start_idx:
        raise ValueError("Training range exceeds available samples; provide explicit indices for smaller datasets")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    from diffusers import Flux2KleinPipeline
    from peft import LoraConfig, get_peft_model

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id, revision=args.model_revision, torch_dtype=dtype,
        token=args.hf_token or os.environ.get("HF_TOKEN"), low_cpu_mem_usage=False)
    multiple = pipe.vae_scale_factor * 2
    if args.image_size % multiple:
        raise ValueError(f"image_size must be divisible by {multiple} for this VAE")
    grid = args.image_size // multiple
    if args.sequential_component_offload:
        pipe.transformer.to(args.device)
    else:
        pipe.to(args.device)
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    transformer = get_peft_model(pipe.transformer, LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules=LORA_TARGETS))
    transformer.train()
    pipe.transformer = transformer
    parameters = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    scheduler = get_lr_scheduler(optimizer, args, args.steps)
    save_run_config(args, args.output_dir)
    settings = {key: getattr(args, key) for key in ATTENTION_DEFAULTS if key != "use_layout_attention"}
    iterator = iter(loader)
    metrics, accumulated = [], 0.0
    optimizer.zero_grad(set_to_none=True)
    print(f"AIBL + Layout Attention | device={args.device} | samples={len(dataset)} | optimizer_updates={args.steps}")
    for micro_step in range(args.steps * args.grad_accum):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        latents, image_ids, embeddings, text_ids = _encode_batch(pipe, batch, args, dtype, micro_step)
        if latents.shape[1] != grid * grid:
            raise ValueError("Backbone latent token grid is incompatible with Layout Attention")
        timestep = torch.rand(len(latents), device=args.device, dtype=dtype)
        noise = torch.randn_like(latents)
        t = timestep[:, None, None]
        noisy = (1 - t) * latents + t * noise
        target = noise - latents
        attention = None
        if args.use_layout_attention:
            bias = build_batch_layout_bias(batch["instances"], batch["overlap_pairs"], grid, grid,
                                            embeddings.shape[1], args.device, dtype, settings)
            attention = {"attention_mask": bias}
        guidance = torch.full_like(timestep, 3.5) if transformer.config.guidance_embeds else None
        prediction = transformer(
            hidden_states=noisy, timestep=timestep, guidance=guidance,
            encoder_hidden_states=embeddings, txt_ids=text_ids, img_ids=image_ids,
            joint_attention_kwargs=attention, return_dict=False)[0]
        if args.use_aibl:
            loss, _ = compute_aibl_loss(
                prediction, target, batch["ownership_map"], batch["amodal_masks"], grid, grid,
                lambda_bg=args.lambda_bg, gamma=args.gamma, alpha_max=args.alpha_max,
                min_instance_area=args.min_instance_area, mix_ratio=args.mix_ratio)
        else:
            loss = F.mse_loss(prediction.float(), target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at microbatch {micro_step}")
        (loss / args.grad_accum).backward()
        accumulated += float(loss.detach()) / args.grad_accum
        if (micro_step + 1) % args.grad_accum:
            continue
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update = (micro_step + 1) // args.grad_accum
        metrics.append({"optimizer_step": update, "loss": accumulated, "lr": scheduler.get_last_lr()[0]})
        if update % args.log_every == 0:
            print(f"step={update} | AIBL loss={accumulated:.6f} | lr={scheduler.get_last_lr()[0]:.3e}")
        accumulated = 0.0
        if update % args.save_every == 0:
            checkpoint = Path(args.output_dir) / f"lora_step_{update}"
            transformer.save_pretrained(checkpoint)
            save_run_config(args, checkpoint)
    destination = Path(args.output_dir) / "lora_final"
    transformer.save_pretrained(destination)
    save_run_config(args, destination)
    (Path(args.output_dir) / "training_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"Completed {len(metrics)} optimizer updates; AIBL adapter saved to {destination}")


if __name__ == "__main__":
    main()
