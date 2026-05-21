"""Multi-seed aggregation.

Reports use IQM + stratified-bootstrap confidence intervals (via `rliable`),
never single-seed point estimates - see docs/PLAN.md section 6.3.
"""

from __future__ import annotations

import numpy as np


def iqm_with_ci(scores: np.ndarray, reps: int = 5000, seed: int = 0):
    """Interquartile mean and 95% bootstrap CI of final performance.

    Args:
        scores: array of shape (num_runs, num_tasks).
    Returns:
        (iqm, (ci_low, ci_high)).
    """
    # TODO(claude-code): implement with rliable.metrics.aggregate_iqm and
    # rliable.library.get_interval_estimates. Left as a stub so the scaffold
    # installs without the optional `analysis` extra.
    raise NotImplementedError("wire rliable here - see docstring and docs/PLAN.md 6.3")
