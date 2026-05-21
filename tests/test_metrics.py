"""Evaluation metric correctness."""

import numpy as np

from roborto.evaluation.metrics import spl, success_rate


def test_success_rate():
    assert success_rate([1, 0, 1, 1]) == 0.75
    assert success_rate([]) == 0.0


def test_spl_equals_success_rate_on_optimal_paths():
    success = np.array([1, 1, 0])
    shortest = np.array([10.0, 5.0, 8.0])
    actual = np.array([10.0, 5.0, 20.0])  # optimal on the two successes
    assert abs(spl(success, shortest, actual) - (2 / 3)) < 1e-9


def test_spl_penalizes_long_paths():
    assert spl(np.array([1]), np.array([10.0]), np.array([20.0])) == 0.5
