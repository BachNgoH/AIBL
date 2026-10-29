"""Layout Attention (LA) with region identity, overlap ownership, and spatial priors."""

import math

import torch

from aibl.layout_validation import normalize_overlap_pairs, validate_grid, validate_instances


def build_region_identity_map(instances, latent_h, latent_w, device="cpu", overlap_pairs=None):
    """Return R, box coverage c(p), and rasterized box centers (background R=0)."""
    validate_grid(latent_h, latent_w)
    instances = validate_instances(instances)
    pairs = normalize_overlap_pairs(instances, overlap_pairs)
    region = torch.zeros(latent_h, latent_w, dtype=torch.long, device=device)
    coverage = torch.zeros_like(region)
    boxes, centers, scores = [], [], []
    for index, instance in enumerate(instances):
        x, y, width, height = instance["bbox"]
        x1, y1 = min(latent_w - 1, int(x * latent_w)), min(latent_h - 1, int(y * latent_h))
        x2 = max(x1 + 1, min(latent_w, int((x + width) * latent_w)))
        y2 = max(y1 + 1, min(latent_h, int((y + height) * latent_h)))
        box = torch.zeros_like(region, dtype=torch.bool)
        box[y1:y2, x1:x2] = True
        boxes.append(box)
        coverage += box.long()
        centers.append(((x1 + x2) / (2 * latent_w), (y1 + y2) / (2 * latent_h)))
        if instance.get("depth") is not None:
            score = instance["depth"]
        elif instance.get("depth_rank") is not None:
            score = -instance["depth_rank"]
        else:
            score = -index
        scores.append(score)
    for index in sorted(range(len(instances)), key=lambda index: scores[index]):
        region[boxes[index]] = index + 1
    for pair in pairs:
        front, back = pair["front"], pair["back"]
        # Preserve a third instance that already owns pixels in front of this pair.
        region[boxes[front] & boxes[back] & (region == back + 1)] = front + 1
    return region, coverage, centers


def _gaussian(latent_h, latent_w, center, sigma, device):
    y, x = torch.meshgrid(torch.linspace(0, 1, latent_h, device=device),
                          torch.linspace(0, 1, latent_w, device=device), indexing="ij")
    return torch.exp(-((x - center[0]).square() + (y - center[1]).square()) / (2 * sigma**2))


def compute_identity_and_spatial_heatmaps(instances, latent_h, latent_w,
                                         bandwidth=0.1, device=None, overlap_pairs=None):
    """Visualize region identity R and owner-masked spatial heatmaps."""
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    device = device or "cpu"
    region, _, centers = build_region_identity_map(instances, latent_h, latent_w, device, overlap_pairs)
    if not centers:
        return None, None
    heatmaps = [_gaussian(latent_h, latent_w, center, math.sqrt(bandwidth / 2), device)
                * (region == index + 1) for index, center in enumerate(centers)]
    return region, torch.stack(heatmaps)


def build_layout_attention_bias(instances, latent_h, latent_w, num_txt_tokens, batch_size,
                                device, dtype, omega_intra=2.0, omega_inter=-1.0, beta=2.0,
                                use_overlap_ownership=True, sigma=math.sqrt(0.05),
                                overlap_pairs=None):
    """Return the layout attention bias as [B,S,S].

    The caller adds the singleton attention-head axis. Text blocks remain zero.
    Gaussian rows follow each query's owner and use scale beta/2.
    """
    if (not isinstance(num_txt_tokens, int) or num_txt_tokens < 0
            or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("num_txt_tokens must be nonnegative and batch_size positive")
    if not all(math.isfinite(value) for value in (omega_intra, omega_inter, beta, sigma)):
        raise ValueError("Layout Attention parameters must be finite")
    if omega_intra <= 0 or omega_inter >= 0 or beta <= 1 or sigma <= 0:
        raise ValueError("Require omega_intra>0, omega_inter<0, beta>1, and sigma>0")
    region, coverage, centers = build_region_identity_map(instances, latent_h, latent_w,
                                                         device, overlap_pairs)
    if not centers:
        return None
    owners = region.flatten()
    foreground = (owners[:, None] > 0) & (owners[None, :] > 0)
    same = (owners[:, None] == owners[None, :]) & foreground
    different = (owners[:, None] != owners[None, :]) & foreground
    size = latent_h * latent_w
    image_bias = torch.zeros(size, size, device=device, dtype=dtype)
    image_bias[same] = beta * omega_intra
    image_bias[different] = omega_inter
    if use_overlap_ownership:
        contested = coverage.flatten() >= 2
        image_bias[same & contested[:, None] & contested[None, :]] += 0.5 * beta * omega_intra
        contested_regions = torch.unique(owners[contested])
        winner_region = torch.isin(owners, contested_regions) & (owners > 0)
        image_bias[different & (winner_region[:, None] | winner_region[None, :])] += 0.5 * omega_inter
    for index, center in enumerate(centers):
        prior = _gaussian(latent_h, latent_w, center, sigma, device).flatten()
        image_bias[owners == index + 1, :] += prior[None, :] * (0.5 * beta)
    sequence = num_txt_tokens + size
    bias = torch.zeros(sequence, sequence, device=device, dtype=dtype)
    bias[num_txt_tokens:, num_txt_tokens:] = image_bias
    return bias.unsqueeze(0).expand(batch_size, -1, -1)
