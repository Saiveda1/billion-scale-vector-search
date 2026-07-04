from __future__ import annotations

from bvs.ivfpq import IVFPQIndex
from bvs.memory import (
    MemoryModel,
    codebook_bytes,
    fp32_bytes_per_vector,
    pq_bytes_per_vector,
)


def test_fp32_bytes():
    assert fp32_bytes_per_vector(128) == 512


def test_pq_bytes_per_vector():
    assert pq_bytes_per_vector(16) == 24  # 16 code bytes + 8 id bytes
    assert pq_bytes_per_vector(16, store_ids=False) == 16


def test_memory_model_matches_actual_index(dataset):
    """Analytic per-vector accounting must equal what the index actually stores."""
    m = 16
    idx = IVFPQIndex(dataset["dim"], nlist=64, m=m).train(dataset["vecs"])
    idx.add(dataset["vecs"], dataset["ids"])
    model = MemoryModel(dim=dataset["dim"], nlist=idx.nlist, m=m)
    # Codes + ids exactly; codebooks are fixed overhead on top.
    assert idx.code_storage_bytes() == idx.ntotal * pq_bytes_per_vector(m)
    assert model.ivfpq_total(idx.ntotal) == (
        idx.code_storage_bytes()
        + codebook_bytes(dataset["dim"], idx.nlist, m)
    )


def test_billion_extrapolation():
    model = MemoryModel(dim=128, nlist=4096, m=16)
    info = model.extrapolate(1_000_000_000)
    # fp32: 128*4 = 512 B/vec -> 512 GB ; PQ: 24 B/vec -> 24 GB.
    assert info["fp32_bytes_per_vec"] == 512
    assert info["pq_bytes_per_vec"] == 24
    assert abs(info["flat_gb"] - 512.0) < 1.0
    assert 23.9 < info["ivfpq_gb"] < 24.2
    assert info["compression_ratio"] > 20


def test_compression_ratio_positive():
    model = MemoryModel(dim=128, nlist=1024, m=8)
    assert model.compression_ratio(1_000_000) > 1.0
