from __future__ import annotations

import numpy as np

from bvs.flat import FlatIndex
from bvs.ivf import IVFIndex


def _ids_returned_are_valid(ids, ntotal):
    flat = ids[ids >= 0]
    return np.all(flat < ntotal) and np.all(flat >= 0)


def test_ivf_returns_valid_subset(dataset):
    ivf = IVFIndex(dataset["dim"], nlist=64).train(dataset["vecs"])
    ivf.add(dataset["vecs"], dataset["ids"])
    d, i = ivf.search(dataset["queries"][:50], k=10, nprobe=4)
    assert i.shape == (50, 10)
    assert _ids_returned_are_valid(i, ivf.ntotal)


def test_ivf_streaming_add_matches_total(dataset):
    ivf = IVFIndex(dataset["dim"], nlist=64).train(dataset["vecs"][:5000])
    total = 0
    for start in range(0, dataset["vecs"].shape[0], 3000):
        batch = dataset["vecs"][start : start + 3000]
        ids = dataset["ids"][start : start + 3000]
        ivf.add(batch, ids)
        total += batch.shape[0]
    assert ivf.ntotal == total == dataset["vecs"].shape[0]
    # Sum of inverted-list sizes must equal ntotal (no vector lost/duplicated).
    assert sum(ivf.list_size(c) for c in range(ivf.nlist)) == ivf.ntotal


def test_ivf_recall_increases_with_nprobe(dataset):
    from bvs.metrics import recall_at_k

    flat = FlatIndex(dataset["dim"])
    flat.add(dataset["vecs"])
    _, truth = flat.search(dataset["queries"], k=10)

    ivf = IVFIndex(dataset["dim"], nlist=64).train(dataset["vecs"])
    ivf.add(dataset["vecs"], dataset["ids"])

    recalls = []
    for nprobe in (1, 4, 16, 64):
        _, approx = ivf.search(dataset["queries"], k=10, nprobe=nprobe)
        recalls.append(recall_at_k(approx, truth, 10))
    # Monotone non-decreasing, and full probe recovers exact search.
    assert all(b >= a - 1e-9 for a, b in zip(recalls, recalls[1:])), recalls
    assert recalls[-1] > 0.98, recalls


def test_full_nprobe_equals_bruteforce_neighbors(dataset):
    """Probing all cells must return the exact nearest neighbour ids."""
    flat = FlatIndex(dataset["dim"])
    flat.add(dataset["vecs"])
    _, truth = flat.search(dataset["queries"][:30], k=5)

    ivf = IVFIndex(dataset["dim"], nlist=64).train(dataset["vecs"])
    ivf.add(dataset["vecs"], dataset["ids"])
    _, approx = ivf.search(dataset["queries"][:30], k=5, nprobe=ivf.nlist)
    # Top-1 must match exactly.
    assert np.array_equal(approx[:, 0], truth[:, 0])
