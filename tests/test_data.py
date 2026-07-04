from __future__ import annotations

import numpy as np

from bvs.data import materialize, stream_vectors


def test_stream_bounded_batches_and_count():
    seen = 0
    max_batch = 0
    for v, ids in stream_vectors(23_000, dim=32, batch_size=5_000, seed=7):
        assert v.dtype == np.float32
        assert v.shape[1] == 32
        assert np.array_equal(ids, np.arange(seen, seen + v.shape[0]))
        max_batch = max(max_batch, v.shape[0])
        seen += v.shape[0]
    assert seen == 23_000
    assert max_batch <= 5_000  # peak memory is bounded by batch_size


def test_stream_is_reproducible():
    a, ida = materialize(4000, dim=16, seed=11)
    b, idb = materialize(4000, dim=16, seed=11)
    assert np.array_equal(a, b)
    assert np.array_equal(ida, idb)


def test_different_seeds_differ():
    a, _ = materialize(2000, dim=16, seed=1)
    b, _ = materialize(2000, dim=16, seed=2)
    assert not np.array_equal(a, b)
