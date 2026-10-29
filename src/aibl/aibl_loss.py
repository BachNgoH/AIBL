"""Amodal-Aware Instance-Balanced Loss (AIBL).

alpha_i=min(alpha_max,(A_amodal/A_visible)**gamma); owned pixels receive
alpha_i, background receives lambda_bg, and each image is mean-normalized.
The objective applies these weights to rectified-flow matching MSE.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from aibl.aibl_validation import sequence_grid, validate_weight_inputs


def compute_instance_weight_map(ownership_map, amodal_masks, latent_h, latent_w,
                                lambda_bg=0.8, gamma=0.5, min_instance_area=1, alpha_max=5.0,
                                weight_mode="ownership", norm_mode="per_image", mu_global=1.0):
    """Construct W_norm on the latent grid; owner IDs are zero-based, background=-1.

    Collapsed weights and dataset_global normalization provide alternate
    weighting and normalization modes. Defaults use per-instance weights.
    """
    validate_weight_inputs(ownership_map, amodal_masks, latent_h, latent_w,
                           lambda_bg, gamma, alpha_max, weight_mode, norm_mode, mu_global)
    if min_instance_area < 1:
        raise ValueError("min_instance_area must be positive")
    own = torch.from_numpy(ownership_map.astype(np.float32))[None, None]
    own = F.interpolate(own, size=(latent_h, latent_w), mode="nearest")[0, 0].long()
    weights = torch.full((latent_h, latent_w), lambda_bg, dtype=torch.float32)
    weights[own >= 0] = 1.0
    alphas, visible_areas, amodal_areas, labels = [], [], [], []
    skipped = 0
    for label in own.unique().tolist():
        if label < 0:
            continue
        visible = int((own == label).sum())
        if visible < min_instance_area:
            skipped += 1
            continue
        mask = torch.from_numpy(np.asarray(amodal_masks[label], dtype=np.float32))[None, None]
        mask = F.interpolate(mask, size=(latent_h, latent_w), mode="bilinear", align_corners=False)[0, 0]
        amodal = max(int((mask > 0.3).sum()), visible)
        alpha = min(alpha_max, (amodal / visible) ** gamma)
        labels.append(label)
        alphas.append(alpha)
        visible_areas.append(visible)
        amodal_areas.append(amodal)
        weights[own == label] = alpha
    if weight_mode == "collapsed":
        alpha_bar = sum(a * v for a, v in zip(alphas, visible_areas)) / max(sum(visible_areas), 1)
        weights[own >= 0] = alpha_bar if visible_areas else 1.0
    raw_mean = float(weights.mean())
    if norm_mode == "per_image":
        weights = weights / weights.mean()
    stats = dict(n_instances=len(labels), boosts=alphas, visible_areas=visible_areas,
                 amodal_areas=amodal_areas, skipped_instances=skipped,
                 raw_mean_weight=raw_mean, final_mean_weight=float(weights.mean()),
                 weight_mode=weight_mode, norm_mode=norm_mode)
    return weights, stats


def adapt_weight_map_to_sequence(weight_map, seq_len, latent_h, latent_w,
                                  device, dtype, norm_mode="per_image"):
    """Average weights over packed spatial tokens, preserving per-image scale."""
    if tuple(weight_map.shape) != (latent_h, latent_w):
        raise ValueError("weight_map shape must equal the latent grid")
    if norm_mode not in {"per_image", "dataset_global"}:
        raise ValueError("Unknown norm_mode")
    height, width = sequence_grid(seq_len, latent_h, latent_w)
    spatial = weight_map.reshape(1, 1, latent_h, latent_w)
    if seq_len <= latent_h * latent_w:
        weights = F.adaptive_avg_pool2d(spatial, (height, width)).flatten()
    else:
        weights = F.interpolate(spatial, size=(height, width), mode="nearest").flatten()
    if norm_mode == "per_image":
        weights = weights / weights.mean()
    return weights.to(device=device, dtype=dtype)


def compute_aibl_loss(model_pred, target, ownership_maps, amodal_masks_batch,
                       latent_h, latent_w, lambda_bg=0.8, gamma=0.5, min_instance_area=1,
                       alpha_max=5.0, mix_ratio=1.0, weight_mode="ownership",
                       norm_mode="per_image", mu_global=1.0):
    """Return AIBL-weighted MSE and diagnostics; mix_ratio=0 gives uniform MSE."""
    if model_pred.ndim != 3 or model_pred.shape != target.shape or min(model_pred.shape) < 1:
        raise ValueError("model_pred and target require matching nonempty [B,tokens,channels] shapes")
    if not math.isfinite(mix_ratio) or not 0 <= mix_ratio <= 1:
        raise ValueError("mix_ratio must lie in [0, 1]")
    residual = (model_pred.float() - target.float()).square().mean(dim=-1)
    uniform = residual.mean()
    batch, sequence = residual.shape
    if mix_ratio == 0:
        return uniform, {"mean_boost": 1.0, "mix_ratio": 0.0}
    if len(ownership_maps) != batch or len(amodal_masks_batch) != batch:
        raise ValueError("AIBL requires one ownership map and mask list per batch sample")
    losses, boosts, stats_batch = [], [], []
    for index in range(batch):
        weights, stats = compute_instance_weight_map(
            ownership_maps[index], amodal_masks_batch[index], latent_h, latent_w,
            lambda_bg, gamma, min_instance_area, alpha_max, weight_mode, norm_mode, mu_global)
        weights = adapt_weight_map_to_sequence(weights, sequence, latent_h, latent_w,
                                               model_pred.device, torch.float32, norm_mode)
        if norm_mode == "dataset_global":
            weights = weights / mu_global
        losses.append((residual[index] * weights).mean())
        boosts.extend(stats["boosts"])
        stats_batch.append(stats)
    weighted = torch.stack(losses).mean()
    diagnostics = {"mean_boost": float(np.mean(boosts)) if boosts else 1.0,
                   "alpha_max": max(boosts, default=1.0), "mix_ratio": mix_ratio,
                   "n_instances_avg": sum(s["n_instances"] for s in stats_batch) / batch,
                   "raw_mean_weight_avg": sum(s["raw_mean_weight"] for s in stats_batch) / batch,
                   "final_mean_weight_avg": sum(s["final_mean_weight"] for s in stats_batch) / batch}
    return mix_ratio * weighted + (1 - mix_ratio) * uniform, diagnostics
