"""Shared Layout Attention settings and migration of recorded checkpoint names."""

import math

ATTENTION_DEFAULTS = {
    "use_layout_attention": True,
    "use_overlap_ownership": True,
    "omega_intra": 2.0,
    "omega_inter": -1.0,
    "beta": 2.0,
    "sigma": 0.05**0.5,
}
MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "to_qkv_mlp_proj"]


def migrate_checkpoint_config(config):
    """Map legacy setting names and reject unsupported checkpoint variants."""
    if not isinstance(config, dict):
        raise ValueError("Checkpoint configuration must be a JSON object")
    result = dict(config)
    aliases = {
        "use_gligen_layout": "use_layout_attention", "use_overlap_zones": "use_overlap_ownership",
        "intra_bbox_bias": "omega_intra", "inter_bbox_bias": "omega_inter", "gligen_boost": "beta",
        "aibl_max_boost": "alpha_max",
    }
    for old, new in aliases.items():
        if old in result:
            if new in result and result[new] != result[old]:
                raise ValueError(f"Conflicting checkpoint fields: {old} and {new}")
            result[new] = result.pop(old)
    supported = dict(omega_winner=0.5, omega_contested=0.5, omega_loser=0.0,
                     depth_cohesion_lambda=0.0, depth_separation_mu=0.0,
                     asym_delta=0.0, use_owner_binding=False)
    for key, default in supported.items():
        active = result.get("use_overlap_ownership", True) if key in {"omega_winner", "omega_contested", "omega_loser"} else True
        if active and key in result and result[key] != default:
            raise ValueError(f"Unsupported checkpoint setting {key}={result[key]!r}")
        result.pop(key, None)
    return result


def validate_attention_settings(settings):
    for key in ("use_layout_attention", "use_overlap_ownership"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    for key in ("omega_intra", "omega_inter", "beta", "sigma"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if (settings["omega_intra"] <= 0 or settings["omega_inter"] >= 0
            or settings["beta"] <= 1 or settings["sigma"] <= 0):
        raise ValueError("Require omega_intra>0, omega_inter<0, beta>1, and sigma>0")
