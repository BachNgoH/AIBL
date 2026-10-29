"""Validated layout contracts shared by attention and public inference.

Boxes use normalized xywh. Pair front/back fields are positional indices;
front_id/back_id fields explicitly refer to instance metadata IDs.
"""

import math
from numbers import Integral


def validate_bbox(bbox, context="bbox"):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{context} must contain normalized [x, y, width, height]")
    try:
        if any(isinstance(value, (bool, str)) for value in bbox):
            raise ValueError("coordinates must be numbers")
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must contain finite numeric coordinates") from exc
    if (not all(math.isfinite(value) for value in (x, y, width, height))
            or x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > 1.000001 or y + height > 1.000001):
        raise ValueError(f"{context} must have positive size and lie within normalized [0, 1]")
    return x, y, width, height


def validate_instances(instances):
    if not isinstance(instances, (list, tuple)):
        raise ValueError("instances must be a list of instance objects")
    validated = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise ValueError(f"instance {index} must be an object")
        instance = dict(instance, bbox=validate_bbox(instance.get("bbox"), f"instance {index} bbox"))
        for field in ("depth", "depth_rank"):
            if instance.get(field) is not None:
                try:
                    value = float(instance[field])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"instance {index} {field} must be finite") from exc
                if not math.isfinite(value):
                    raise ValueError(f"instance {index} {field} must be finite")
                instance[field] = value
        validated.append(instance)
    return validated


def normalize_overlap_pairs(instances, pairs):
    """Return positional pair references; never guess whether a number is an ID."""
    if pairs is None:
        return []
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("overlap_pairs must be a list")
    normalized = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("each overlap pair must be an object")
        if "front" in pair or "back" in pair:
            if "front_id" in pair or "back_id" in pair:
                raise ValueError("overlap pair cannot mix positional and ID reference fields")
            front, back = pair.get("front"), pair.get("back")
        else:
            id_to_index = {}
            for index, instance in enumerate(instances):
                identifier = instance.get("id", index)
                if not isinstance(identifier, (int, str)) or isinstance(identifier, bool):
                    raise ValueError("instance IDs must be integers or strings")
                if identifier in id_to_index:
                    raise ValueError("instance IDs must be unique for ID-based overlap pairs")
                id_to_index[identifier] = index
            try:
                front = id_to_index[pair["front_id"]]
                back = id_to_index[pair["back_id"]]
            except (KeyError, TypeError) as exc:
                raise ValueError("overlap pair references a missing instance ID") from exc
        if (not isinstance(front, Integral) or isinstance(front, bool)
                or not isinstance(back, Integral) or isinstance(back, bool)
                or not 0 <= front < len(instances) or not 0 <= back < len(instances)
                or front == back):
            raise ValueError("overlap front/back must be distinct valid instance positions")
        normalized.append({"front": int(front), "back": int(back)})
    return normalized


def validate_layout(layout, require_instances=True):
    if not isinstance(layout, dict):
        raise ValueError("layout must be an object containing prompt and instances")
    prompt = layout.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("layout prompt must be a nonempty string")
    instances = validate_instances(layout.get("instances"))
    if require_instances and not instances:
        raise ValueError("layout-conditioned generation requires at least one instance")
    return dict(layout, instances=instances,
                overlap_pairs=normalize_overlap_pairs(instances, layout.get("overlap_pairs")))


def validate_grid(latent_h, latent_w):
    if any(not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
           for value in (latent_h, latent_w)):
        raise ValueError("latent grid dimensions must be positive integers")
