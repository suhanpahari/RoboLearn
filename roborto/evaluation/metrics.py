"""Evaluation metrics. Aggregate across seeds with roborto.analysis."""

from __future__ import annotations

import numpy as np


def success_rate(successes) -> float:
    """Fraction of successful episodes. `successes`: iterable of 0/1."""
    s = np.asarray(successes, dtype=np.float64)
    return float(s.mean()) if s.size else 0.0


def spl(success, shortest_path, actual_path) -> float:
    """Success weighted by Path Length (Anderson et al., 2018).

    SPL = mean_i  S_i * l_i / max(p_i, l_i)
    """
    s = np.asarray(success, dtype=np.float64)
    shortest = np.asarray(shortest_path, dtype=np.float64)
    actual = np.asarray(actual_path, dtype=np.float64)
    denom = np.maximum(actual, shortest)
    denom = np.where(denom > 0, denom, 1.0)
    return float(np.mean(s * shortest / denom))


# TODO(claude-code): add soft_spl, distance_to_goal_reduction and collision_rate
# for the Habitat navigation track (Phase 2).
