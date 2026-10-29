#!/usr/bin/env python3
"""Generate layout-conditioned images from an AIBL checkpoint, without SAM3.

Uses the checkpoint's saved attention configuration, with explicit CLI overrides.
By default layouts come from dataset annotations (an oracle-layout diagnostic).
Use --layout_dir to generate from a fixed set of shared layouts.
Sample bounds are positions in the sorted sample directories, not sample IDs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from aibl.dataset import OverlapDepth45KDataset  # noqa: E402
from aibl.layout_attention import build_layout_attention_bias  # noqa: E402
from aibl.layout_validation import validate_layout  # noqa: E402
from aibl.inference_config import (parse_args, resolve_config, split_indices,
                                   read_generation_metadata, write_generation_metadata)
from aibl.settings import ATTENTION_DEFAULTS
from aibl.runtime import select_device_and_dtype


def main(argv=None):
    args = parse_args(argv)
    effective_config = resolve_config(args)
    args.device, dtype = select_device_and_dtype(args.device, args.precision)
    effective_config["precision"] = "bf16" if dtype == torch.bfloat16 else "fp32"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AIBL-LoRA Image Generation")
    print("=" * 60)
    print(f"LoRA: {args.lora_path}")
    print(f"Output: {out_dir}")
    print(f"Image size: {args.image_size} | Steps: {args.num_inference_steps} | CFG: {args.guidance_scale}")
    print(f"Layout Attention: {args.use_layout_attention} (omega_intra={args.omega_intra}, omega_inter={args.omega_inter}, beta={args.beta})")
    print(f"Split: {args.split} | Samples: {args.num_samples} | start_offset: {args.start_offset}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    start_idx, end_idx = split_indices(args)
    print(f"Sample range: {start_idx}–{end_idx - 1}")

    # Load dataset to get the prompts + bboxes
    dataset = OverlapDepth45KDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        start_idx=start_idx,
        end_idx=end_idx,
        load_masks=False,
    )

    if not len(dataset):
        raise ValueError("No samples in selected range; check explicit split indices")

    meta_path = out_dir / "_generation_metadata.json"
    previous_metadata = read_generation_metadata(meta_path) if args.save_metadata else []
    # Validate selected input before loading any model or encoding prompts.
    todo = []
    skipped = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = int(sample.get("sample_id", start_idx + i))
        out_path = out_dir / f"{sample_id:06d}.png"
        if args.resume and out_path.exists():
            skipped += 1
            continue
        layout = sample
        if args.layout_dir:
            layout_path = args.layout_dir / f"{sample_id:06d}.json"
            layout = json.loads(layout_path.read_text())
        layout = validate_layout(layout, require_instances=args.use_layout_attention)
        todo.append((sample_id, {key: layout[key] for key in ("prompt", "instances", "overlap_pairs")}))
    print(f"\n{len(todo)} samples to generate ({skipped} skipped from existing PNGs)")

    if not todo:
        print("No images to generate; existing provenance preserved.")
        return

    # Load FLUX pipeline
    print("\nLoading FLUX pipeline ...")
    from diffusers import Flux2KleinPipeline
    from peft import PeftModel

    pipe_kwargs = {}
    if args.hf_token or os.environ.get("HF_TOKEN"):
        pipe_kwargs["token"] = args.hf_token or os.environ["HF_TOKEN"]
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        revision=args.model_revision,
        low_cpu_mem_usage=False,
        **pipe_kwargs,
    )

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}")
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path, adapter_name="aibl")
        pipe.transformer.set_adapter("aibl")

    pipe = pipe.to(args.device)
    multiple = pipe.vae_scale_factor * 2
    if args.image_size % multiple:
        raise ValueError(f"image_size must be divisible by {multiple} for this VAE")

    metadata = []
    for sample_id, layout in tqdm(todo, desc="Generating"):
        prompt = layout["prompt"]
        out_path = out_dir / f"{sample_id:06d}.png"

        generator = torch.Generator(device=args.device).manual_seed(
            args.seed + int(sample_id)
        )
        instances = layout["instances"]
        t0 = time.time()
        with torch.inference_mode():
            prompt_embeds, _ = pipe.encode_prompt(prompt, device=args.device,
                                                   max_sequence_length=args.max_sequence_length)
            bias = None
            if args.use_layout_attention:
                settings = {key: getattr(args, key) for key in ATTENTION_DEFAULTS if key != "use_layout_attention"}
                bias = build_layout_attention_bias(
                    instances, args.image_size // multiple, args.image_size // multiple,
                    prompt_embeds.shape[1], 1, args.device, prompt_embeds.dtype,
                    overlap_pairs=layout.get("overlap_pairs"), **settings,
                )
            # Explicit head axis: [B, 1, S, S] broadcasts across attention heads.
            attention_kwargs = {"attention_mask": bias.unsqueeze(1)} if bias is not None else None
            result = pipe(
                prompt_embeds=prompt_embeds,
                max_sequence_length=args.max_sequence_length,
                attention_kwargs=attention_kwargs,
                height=args.image_size,
                width=args.image_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
        elapsed = time.time() - t0
        image: Image.Image = result.images[0]
        image.save(out_path, "PNG")
        if args.save_metadata:
            metadata.append({
                "sample_id": int(sample_id),
                "prompt": prompt,
                "elapsed_s": round(elapsed, 3),
                "image_size": args.image_size,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed + int(sample_id),
                "layout_source": str(args.layout_dir) if args.layout_dir else "dataset_annotations",
                "attention_config": effective_config,
            })
            # Persist each completed image so interruption does not discard
            # the provenance for all PNGs saved earlier in this run.
            write_generation_metadata(meta_path, metadata, previous_metadata)

    if args.save_metadata:
        print(f"Wrote {meta_path}")

    print(f"\nDone. {len(todo)} images written to {out_dir}")


if __name__ == "__main__":
    main()
