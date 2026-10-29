"""Runtime utilities shared by AIBL training and inference."""

import json
from pathlib import Path

import numpy as np
import torch

def get_lr_scheduler(optimizer, args, num_training_steps):
    """Create LR scheduler with warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.lr_scheduler == "cosine":
            progress = (step - args.warmup_steps) / max(1, num_training_steps - args.warmup_steps)
            return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * progress))
        elif args.lr_scheduler == "linear":
            progress = (step - args.warmup_steps) / max(1, num_training_steps - args.warmup_steps)
            return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * (1 - progress)
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


def public_config(args):
    """Persist reproducibility settings without authentication material."""
    from importlib.metadata import PackageNotFoundError, version
    import platform

    config = {key: value for key, value in vars(args).items() if key != "hf_token"}
    runtime = {"python": platform.python_version()}
    for package in ("torch", "diffusers", "transformers", "peft", "numpy", "pillow"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "unavailable"
    config["runtime_versions"] = runtime
    return config


def save_run_config(args, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(public_config(args), indent=2) + "\n")


def image_to_tensor(images, size, device):
    """Resize RGB images and normalize them to [-1, 1]."""
    from PIL import Image

    tensors = []
    for image in images:
        image = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors).to(device)


def select_device_and_dtype(device="auto", precision="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda[:N]")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu for CPU execution")
    supports_bf16 = False
    if selected.type == "cuda":
        with torch.cuda.device(selected):
            supports_bf16 = torch.cuda.is_bf16_supported()
    if precision == "auto":
        precision = "bf16" if supports_bf16 else "fp32"
    if precision == "bf16" and selected.type == "cuda" and not supports_bf16:
        raise ValueError("This CUDA device does not support bf16; use --precision fp32")
    return str(selected), torch.bfloat16 if precision == "bf16" else torch.float32


def build_batch_layout_bias(instances_batch, pairs_batch, latent_h, latent_w,
                            num_txt_tokens, device, dtype, settings):
    from aibl.layout_attention import build_layout_attention_bias

    masks = []
    for instances, pairs in zip(instances_batch, pairs_batch):
        bias = build_layout_attention_bias(instances, latent_h, latent_w, num_txt_tokens,
                                           1, device, dtype, overlap_pairs=pairs, **settings)
        if bias is None:
            sequence = num_txt_tokens + latent_h * latent_w
            bias = torch.zeros(1, sequence, sequence, device=device, dtype=dtype)
        masks.append(bias)
    return torch.cat(masks).unsqueeze(1)
