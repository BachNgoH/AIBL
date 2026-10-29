"""Input contracts and spatial sequence geometry for AIBL."""

import math

import numpy as np


def sequence_grid(seq_len, latent_h, latent_w):
    """Choose an exact token grid, preserving the latent aspect ratio if possible."""
    if min(seq_len, latent_h, latent_w) < 1:
        raise ValueError("Sequence and latent dimensions must be positive")
    ratio = latent_h / latent_w
    pairs = []
    for height in range(1, math.isqrt(seq_len) + 1):
        if seq_len % height == 0:
            width = seq_len // height
            pairs.extend(((height, width), (width, height)))
    return min(pairs, key=lambda pair: abs(math.log((pair[0] / pair[1]) / ratio)))


def validate_weight_inputs(ownership, masks, height, width, lambda_bg, gamma,
                           alpha_max, weight_mode, norm_mode, mu_global):
    if min(height, width) < 1:
        raise ValueError("Latent dimensions must be positive")
    if not math.isfinite(lambda_bg) or lambda_bg <= 0:
        raise ValueError("lambda_bg must be finite and positive")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if not math.isfinite(alpha_max) or alpha_max < 1:
        raise ValueError("alpha_max must be finite and at least 1")
    if weight_mode not in {"ownership", "collapsed"}:
        raise ValueError("weight_mode must be ownership or collapsed")
    if norm_mode not in {"per_image", "dataset_global"}:
        raise ValueError("norm_mode must be per_image or dataset_global")
    if norm_mode == "dataset_global" and (not math.isfinite(mu_global) or mu_global <= 0):
        raise ValueError("mu_global must be finite and positive")
    if not isinstance(ownership, np.ndarray) or ownership.ndim != 2 or not ownership.size:
        raise ValueError("ownership_map must be a nonempty 2D numpy array")
    if not np.issubdtype(ownership.dtype, np.integer):
        raise ValueError("ownership_map must contain integer labels")
    if ownership.min() < -1 or ownership.max() >= len(masks):
        raise ValueError("ownership labels must be -1 or zero-based indices into amodal_masks")
    for mask in masks:
        mask = np.asarray(mask)
        if mask.shape != ownership.shape or not np.isfinite(mask).all():
            raise ValueError("Amodal masks must match ownership shape and contain finite values")
        if (mask < 0).any() or (mask > 1).any():
            raise ValueError("Amodal mask values must lie in [0, 1]")
