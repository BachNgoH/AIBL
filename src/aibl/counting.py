"""
Counting evaluation metrics.

Metrics:
- Exact accuracy: Exact count match
- Mean absolute error: Average |predicted - ground_truth|
- Accuracy within tolerance: Count within ±k of ground truth
"""

from dataclasses import dataclass, asdict, field
from typing import List
from numbers import Integral

import numpy as np


@dataclass
class CountingMetrics:
    """Counting evaluation results."""
    exact_accuracy: float
    mean_absolute_error: float
    mean_relative_error: float
    accuracy_within_1: float
    accuracy_within_2: float
    accuracy_within_3: float
    per_sample_errors: List[int] = field(default_factory=list)
    root_mean_squared_error: float = 0.0
    accuracy_within_10_percent: float = 0.0
    accuracy_within_20_percent: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def count_accuracy(predicted: int, ground_truth: int) -> float:
    """Exact count match (1.0 if equal, 0.0 otherwise)."""
    return 1.0 if predicted == ground_truth else 0.0


def count_absolute_error(predicted: int, ground_truth: int) -> int:
    """Absolute count error."""
    return abs(predicted - ground_truth)


def count_relative_error(predicted: int, ground_truth: int) -> float:
    """Relative count error (normalized by ground truth)."""
    if ground_truth == 0:
        return 0.0 if predicted == 0 else 1.0
    return abs(predicted - ground_truth) / ground_truth


def count_within_tolerance(predicted: int, ground_truth: int, k: int = 2) -> bool:
    """Check if count is within tolerance k."""
    return abs(predicted - ground_truth) <= k


def evaluate_counting(
    predicted_counts: List[int],
    ground_truth_counts: List[int],
) -> CountingMetrics:
    """
    Evaluate counting performance.

    Args:
        predicted_counts: List of predicted counts per sample
        ground_truth_counts: List of ground truth counts per sample

    Returns:
        CountingMetrics with all counting metrics
    """
    if len(predicted_counts) != len(ground_truth_counts):
        raise ValueError("Predicted and ground-truth counts must have equal lengths")
    for count in list(predicted_counts) + list(ground_truth_counts):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Counts must be nonnegative integers")
    n = len(predicted_counts)
    if n == 0:
        return CountingMetrics(0, 0, 0, 0, 0, 0, [])

    exact_matches = sum(1 for p, g in zip(predicted_counts, ground_truth_counts) if p == g)
    within_1 = sum(1 for p, g in zip(predicted_counts, ground_truth_counts) if abs(p - g) <= 1)
    within_2 = sum(1 for p, g in zip(predicted_counts, ground_truth_counts) if abs(p - g) <= 2)
    within_3 = sum(1 for p, g in zip(predicted_counts, ground_truth_counts) if abs(p - g) <= 3)

    errors = [abs(p - g) for p, g in zip(predicted_counts, ground_truth_counts)]
    relative_errors = [
        count_relative_error(p, g)
        for p, g in zip(predicted_counts, ground_truth_counts)
    ]

    return CountingMetrics(
        exact_accuracy=exact_matches / n,
        mean_absolute_error=float(np.mean(errors)),
        mean_relative_error=float(np.mean(relative_errors)),
        accuracy_within_1=within_1 / n,
        accuracy_within_2=within_2 / n,
        accuracy_within_3=within_3 / n,
        per_sample_errors=errors,
        root_mean_squared_error=float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=np.float64))))),
        accuracy_within_10_percent=sum(10 * abs(p - g) <= g for p, g in zip(predicted_counts, ground_truth_counts)) / n,
        accuracy_within_20_percent=sum(5 * abs(p - g) <= g for p, g in zip(predicted_counts, ground_truth_counts)) / n,
    )
