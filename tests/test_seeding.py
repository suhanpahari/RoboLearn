"""set_global_seed must make runs reproducible."""

import numpy as np

from roborto.utils.seeding import set_global_seed


def test_same_seed_is_reproducible():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    set_global_seed(1)
    a = np.random.rand(5)
    set_global_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
