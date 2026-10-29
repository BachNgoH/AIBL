"""
OverlapDepth-45K dataset loader.

Dataset structure:
    data_root/
    └── samples/
        ├── 000000/
        │   ├── image.png       # RGB image
        │   ├── meta.json       # Instance metadata
        │   └── masks.npz       # amodal_0, modal_0, depth_map, ownership_map
        ├── 000001/
        └── ...

Each sample contains:
    - image: RGB image array [H, W, 3]
    - prompt: Text description
    - instances: List of instance dicts with bbox, category, depth, etc.
    - overlap_pairs: List of {front, back} pairs indicating occlusion
    - amodal_masks: List of amodal (full object) masks
    - modal_masks: List of modal (visible only) masks
    - depth_map: Scene depth map
    - ownership_map: Per-pixel instance ownership
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from aibl.layout_validation import normalize_overlap_pairs, validate_instances


def _load_json(path: Path, max_retries: int = 3) -> dict:
    """Load JSON file with retry for transient NFS read truncation."""
    import time
    for attempt in range(max_retries):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                raise


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load NPZ file as dict of arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _validate_mask_data(out: Dict[str, Any]) -> None:
    """Fail fast when supervision arrays do not match the sample metadata."""
    n = out["n_objects"]
    expected_shape = (out["height"], out["width"])
    if len(out["amodal_masks"]) != n or len(out["modal_masks"]) != n:
        raise ValueError(
            f"expected {n} amodal/modal masks, got "
            f"{len(out['amodal_masks'])}/{len(out['modal_masks'])}"
        )

    for name in ("amodal_masks", "modal_masks"):
        for index, mask in enumerate(out[name]):
            if not np.isfinite(mask).all():
                raise ValueError(f"{name}[{index}] contains nonfinite values")
            if (np.asarray(mask) < 0).any() or (np.asarray(mask) > 1).any():
                raise ValueError(f"{name}[{index}] values must lie in [0, 1]")
            if np.asarray(mask).shape != expected_shape:
                raise ValueError(
                    f"{name}[{index}] has shape {np.asarray(mask).shape}; "
                    f"expected {expected_shape}"
                )

    for name in ("depth_map", "ownership_map"):
        if np.asarray(out[name]).shape != expected_shape:
            raise ValueError(
                f"{name} has shape {np.asarray(out[name]).shape}; "
                f"expected {expected_shape}"
            )

    ownership = np.asarray(out["ownership_map"])
    if not np.issubdtype(ownership.dtype, np.integer):
        raise ValueError("ownership_map must contain integer instance labels")
    if ownership.size and (ownership.min() < -1 or ownership.max() >= n):
        raise ValueError(
            f"ownership_map values must be -1 or instance indices 0..{n - 1}; "
            f"got range {ownership.min()}..{ownership.max()}"
        )


class OverlapDepth45KDataset(Dataset):
    """
    Dataset for OverlapDepth-45K layout-conditioned image generation.

    Loads images with per-instance masks (amodal + modal), depth maps,
    and overlap pair annotations for training depth-aware image generation.

    Args:
        data_root: Root directory containing samples/ subdirectory
        split: Optional dataset split: train=0:44000, test=44000:45000.
               Explicit start/end indices take precedence; verify dataset ordering.
        max_objects: Maximum accepted object count; larger samples raise an error
        image_size: Image size (int or tuple)
        load_masks: Whether to load mask data from NPZ files
    """

    def __init__(
        self,
        data_root: str,
        split: Optional[str] = None,
        max_objects: int = 50,
        image_size: int = 512,
        load_masks: bool = True,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.samples_dir = self.data_root / "samples"
        self.load_masks = load_masks
        self.max_objects = max_objects
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size

        if split not in (None, "train", "test"):
            raise ValueError("split must be train, test, or None")
        if start_idx is not None and start_idx < 0:
            raise ValueError("start_idx must be nonnegative")
        if end_idx is not None and end_idx < (start_idx or 0):
            raise ValueError("end_idx must be at least start_idx")

        # Find all sample directories
        self.sample_dirs = sorted([
            d for d in self.samples_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        ])

        # Apply start/end indices first (takes precedence over split)
        if start_idx is not None or end_idx is not None:
            start = start_idx or 0
            end = len(self.sample_dirs) if end_idx is None else end_idx
            self.sample_dirs = self.sample_dirs[start:end]
        # Apply split if specified
        elif split is not None:
            if len(self.sample_dirs) != 45000:
                raise ValueError("Named splits require exactly 45,000 ordered samples; use explicit indices otherwise")
            self.sample_dirs = self.sample_dirs[:44000] if split == "train" else self.sample_dirs[44000:45000]

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single sample."""
        sample_dir = self.sample_dirs[idx]
        meta = _load_json(sample_dir / "meta.json")
        image = np.array(Image.open(sample_dir / "image.png").convert("RGB"))

        # Limit objects
        declared_objects = int(meta["n_objects"])
        image_height = int(meta.get("height", image.shape[0]))
        image_width = int(meta.get("width", image.shape[1]))
        if (image_height, image_width) != image.shape[:2]:
            raise ValueError(
                f"{sample_dir / 'meta.json'} declares image size "
                f"{image_height}x{image_width}, but image.png is "
                f"{image.shape[0]}x{image.shape[1]}"
            )
        if declared_objects != len(meta["instances"]):
            raise ValueError(
                f"{sample_dir / 'meta.json'} declares {declared_objects} objects "
                f"but contains {len(meta['instances'])} instances"
            )
        if declared_objects > self.max_objects:
            raise ValueError(
                f"{sample_dir / 'meta.json'} has {declared_objects} objects, "
                f"exceeding max_objects={self.max_objects}"
            )
        n = declared_objects
        instances = validate_instances(meta["instances"])

        # Shared contract: positions and explicit metadata IDs are never conflated.
        overlap_pairs = normalize_overlap_pairs(instances, meta.get("overlap_pairs"))

        out = {
            "image": image,
            "prompt": meta.get("prompt") or meta.get("global_caption", ""),
            "instances": instances,
            "overlap_pairs": overlap_pairs,
            "n_objects": n,
            "sample_id": meta.get("sample_id", idx),
            "height": image_height,
            "width": image_width,
        }

        # Load masks if available
        if self.load_masks:
            masks_path = sample_dir / "masks.npz"
            if not masks_path.exists():
                raise FileNotFoundError(f"Missing required mask data: {masks_path}")
            try:
                data = _load_npz(masks_path)

                # Handle both old format (amodal_0, amodal_1, ...) and new format (amodal_masks array)
                if "amodal_masks" in data:
                    # New format: masks stored as arrays
                    amodal_arr = data["amodal_masks"]
                    if "modal_masks" in data:
                        modal_arr = data["modal_masks"]
                    elif "masks" in data:
                        modal_arr = data["masks"]
                    else:
                        raise KeyError("modal_masks")
                    if len(amodal_arr) != n or len(modal_arr) != n:
                        raise ValueError(
                            f"n_objects={n}, but mask arrays contain "
                            f"{len(amodal_arr)}/{len(modal_arr)} entries"
                        )
                    out["amodal_masks"] = [amodal_arr[i] for i in range(n)]
                    out["modal_masks"] = [modal_arr[i] for i in range(n)]
                else:
                    # Old format: masks stored as individual keys
                    amodal_keys = {
                        key for key in data if key.startswith("amodal_")
                    }
                    modal_keys = {
                        key for key in data if key.startswith("modal_")
                    }
                    if len(amodal_keys) != n or len(modal_keys) != n:
                        raise ValueError(
                            f"n_objects={n}, but individual masks contain "
                            f"{len(amodal_keys)}/{len(modal_keys)} entries"
                        )
                    out["amodal_masks"] = [data[f"amodal_{i}"] for i in range(n)]
                    out["modal_masks"] = [data[f"modal_{i}"] for i in range(n)]

                out["depth_map"] = data["depth_map"]
                out["ownership_map"] = data["ownership_map"]
                _validate_mask_data(out)
            except Exception as exc:
                raise ValueError(f"Invalid mask data in {masks_path}: {exc}") from exc

        return out


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """
    Collate function for DataLoader.

    Preserves list structure for variable-length fields (instances, masks, etc.)
    while stacking fixed-size tensors.

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Collated batch dict with lists preserved
    """
    has_masks = "amodal_masks" in batch[0]

    out = {
        "image": [b["image"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
        "instances": [b["instances"] for b in batch],
        "overlap_pairs": [b["overlap_pairs"] for b in batch],
        "n_objects": [b["n_objects"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
    }

    if has_masks:
        out["amodal_masks"] = [b["amodal_masks"] for b in batch]
        out["modal_masks"] = [b["modal_masks"] for b in batch]
        out["depth_map"] = [b["depth_map"] for b in batch]
        out["ownership_map"] = [b["ownership_map"] for b in batch]

    return out
