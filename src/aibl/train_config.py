"""Command-line recipe for Layout Attention and AIBL training."""

import argparse
import math

from aibl.settings import ATTENTION_DEFAULTS, MODEL_ID, validate_attention_settings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Layout Attention (LA) and AIBL on OverlapDepth-45K")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="outputs/aibl")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_revision")
    parser.add_argument("--train_start_idx", type=int, default=0)
    parser.add_argument("--train_end_idx", type=int, default=44000)
    parser.add_argument("--steps", type=int, default=3000, help="Number of optimizer updates")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine", "linear"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--sequential_component_offload", action="store_true")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    for key, default in ATTENTION_DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(f"--{key}", type=float, default=default)
    parser.add_argument("--use_aibl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma", type=float, default=0.5, help="Occlusion boost exponent gamma")
    parser.add_argument("--lambda_bg", type=float, default=0.8)
    parser.add_argument("--alpha_max", type=float, default=5.0, help="Maximum occlusion boost alpha_max")
    parser.add_argument("--mix_ratio", type=float, default=1.0, help="1=AIBL objective, 0=uniform MSE")
    parser.add_argument("--min_instance_area", type=int, default=1)
    parser.add_argument("--hf_token", help="Prefer setting HF_TOKEN in the environment")
    args = parser.parse_args(argv)
    for key in ("steps", "batch_size", "grad_accum", "lora_rank", "save_every", "log_every",
                "max_sequence_length", "min_instance_area"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.image_size < 64 or args.image_size % 16:
        parser.error("image_size must be a multiple of 16 and at least 64")
    if args.num_workers < 0 or args.warmup_steps < 0:
        parser.error("num_workers and warmup_steps must be nonnegative")
    if args.train_start_idx < 0 or args.train_end_idx <= args.train_start_idx:
        parser.error("training indices must define a nonempty half-open range")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        parser.error("Require lr>0 and 0<=min_lr<=lr")
    if not 0 <= args.mix_ratio <= 1 or args.gamma < 0 or args.lambda_bg <= 0 or args.alpha_max < 1:
        parser.error("Invalid AIBL parameters")
    try:
        validate_attention_settings(vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    for key in ("min_lr", "grad_clip", "gamma", "lambda_bg", "alpha_max", "mix_ratio"):
        if not math.isfinite(getattr(args, key)):
            parser.error(f"{key} must be finite")
    return args
