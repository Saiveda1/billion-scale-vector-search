from __future__ import annotations

import numpy as np

from bvs.flat import FlatIndex
from bvs.ivfpq import IVFPQIndex
from bvs.metrics import recall_at_k


def _truth(dataset, k=10):
    flat = FlatIndex(dataset["dim"])
    flat.add(dataset["vecs"])
    _, truth = flat.search(dataset["queries"], k=k)
    return truth


def test_ivfpq_build_and_search_shapes(dataset):
    idx = IVFPQIndex(dataset["dim"], nlist=64, m=16, ksub=256).train(dataset["vecs"])
    idx.add(dataset["vecs"], dataset["ids"])
    d, i = idx.search(dataset["queries"][:20], k=10, nprobe=8)
    assert d.shape == (20, 10)
    assert i.shape == (20, 10)
    assert np.all(i[i >= 0] < idx.ntotal)


def test_ivfpq_streaming_build(dataset):
    idx = IVFPQIndex(dataset["dim"], nlist=64, m=16).train(dataset["vecs"][:6000])
    total = 0
    for start in range(0, dataset["vecs"].shape[0], 2500):
        b = dataset["vecs"][start : start + 2500]
        ids = dataset["ids"][start : start + 2500]
        idx.add(b, ids)
        total += b.shape[0]
    assert idx.ntotal == total
    assert sum(idx.list_size(c) for c in range(idx.nlist)) == idx.ntotal


def test_ivfpq_recall_increases_with_nprobe(dataset):
    truth = _truth(dataset)
    idx = IVFPQIndex(dataset["dim"], nlist=64, m=16, ksub=256).train(dataset["vecs"])
    idx.add(dataset["vecs"], dataset["ids"])
    recalls = []
    for nprobe in (1, 4, 16, 64):
        _, approx = idx.search(dataset["queries"], k=10, nprobe=nprobe)
        recalls.append(recall_at_k(approx, truth, 10))
    assert all(b >= a - 0.02 for a, b in zip(recalls, recalls[1:])), recalls
    # PQ is lossy, but with clustered data recall should be substantial.
    assert recalls[-1] > 0.5, recalls


def test_ivfpq_code_storage_bytes(dataset):
    m = 16
    idx = IVFPQIndex(dataset["dim"], nlist=64, m=m).train(dataset["vecs"])
    idx.add(dataset["vecs"], dataset["ids"])
    assert idx.code_storage_bytes() == idx.ntotal * (m + 8)
