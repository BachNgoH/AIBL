"""Configuration and selection helpers for the public generation command."""

import argparse
import json
import os
from pathlib import Path
import tempfile


from aibl.settings import ATTENTION_DEFAULTS, MODEL_ID, migrate_checkpoint_config, validate_attention_settings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate images with AIBL LoRA and layout attention")
    for name in ("data_root", "lora_path", "output_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--config", type=Path, help="Training configuration (auto-discovered beside LoRA)")
    parser.add_argument("--model_id")
    parser.add_argument("--model_revision")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--max_sequence_length", type=int)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--start_idx", type=int, help="Inclusive position in sorted sample directories")
    parser.add_argument("--end_idx", type=int, help="Exclusive position in sorted sample directories")
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"],
                        help="OverlapDepth-45K: train=0:44000, test=44000:45000; verify ordering")
    parser.add_argument("--layout_dir", type=Path,
                        help="Shared layouts: SAMPLE_ID.json with prompt and normalized xywh instances")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf_token", help="Prefer HF_TOKEN environment variable")
    for key, value in ATTENTION_DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(f"--{key}", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_metadata", action="store_true")
    args = parser.parse_args(argv)
    if args.num_samples <= 0 or args.start_offset < 0 or args.num_inference_steps <= 0:
        parser.error("sample/step counts must be positive and start_offset nonnegative")
    for name in ("start_idx", "end_idx"):
        if getattr(args, name) is not None and getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")
    if args.start_idx is not None and args.end_idx is not None and args.end_idx <= args.start_idx:
        parser.error("end_idx must exceed start_idx")
    return args


def resolve_config(args):
    """Resolve CLI overrides against settings saved with the checkpoint."""
    checkpoint = Path(args.lora_path)
    candidates = [args.config] if args.config else [checkpoint / "config.json", checkpoint.parent / "config.json"]
    config_path = next((path for path in candidates if path.is_file()), None)
    if args.config and config_path is None:
        raise FileNotFoundError(args.config)
    config = migrate_checkpoint_config(json.loads(config_path.read_text())) if config_path else {}
    defaults = dict(ATTENTION_DEFAULTS, model_id=MODEL_ID, model_revision=None, image_size=512,
                    precision="auto", max_sequence_length=512)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, value))
    validate_attention_settings(vars(args))
    if type(args.image_size) is not int or args.image_size < 64 or args.image_size % 16:
        raise ValueError("image_size must be a multiple of 16 and at least 64")
    if type(args.max_sequence_length) is not int or args.max_sequence_length < 1 or args.precision not in {"auto", "bf16", "fp32"}:
        raise ValueError("Invalid max_sequence_length or precision in checkpoint configuration")
    return {key: getattr(args, key) for key in defaults}


def split_indices(args):
    """Resolve the selected positional range within the dataset."""
    ranges = {"all": (0, None), "train": (0, 44000), "test": (44000, 45000)}
    if args.start_idx is not None or args.end_idx is not None:
        start, end = args.start_idx or 0, args.end_idx
    else:
        start, end = ranges[args.split]
    start += args.start_offset
    end = min(start + args.num_samples, end) if end is not None else start + args.num_samples
    if end <= start:
        raise ValueError("selected sample range is empty")
    return start, end


def read_generation_metadata(path):
    """Load prior provenance before generation, rejecting corrupt/ambiguous records."""
    path = Path(path)
    if not path.exists():
        return []
    def reject_nonfinite(value):
        raise ValueError(f"generation metadata contains invalid JSON constant {value}")

    rows = json.loads(path.read_text(), parse_constant=reject_nonfinite)
    if not isinstance(rows, list):
        raise ValueError("generation metadata must be a list of sample records")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("generation metadata records must be objects")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id in seen:
            raise ValueError("generation metadata needs unique integer sample_id fields")
        seen.add(sample_id)
    return rows


def write_generation_metadata(path, rows, previous_rows):
    """Merge by sample ID and atomically replace the metadata sidecar."""
    path = Path(path)
    merged = {row["sample_id"]: row for row in previous_rows}
    merged.update({row["sample_id"]: row for row in rows})
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".generation-", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump([merged[key] for key in sorted(merged)], handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
