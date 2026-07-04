"""Streaming synthetic vector generator (bounded memory, deterministic).

Vectors are drawn from a Gaussian mixture with a fixed set of latent cluster
centres. Clustered data (rather than uniform noise) is what makes ANN recall a
meaningful quantity — there are genuine near neighbours to find or miss.

The generator yields fixed-size batches from a per-batch seeded RNG, so it can
emit an arbitrary number of vectors (up to 1B) while holding only one batch in
memory at a time. Given the same ``seed`` and ``n`` it is byte-for-byte
reproducible, and a global vector id is attached to every row.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from . import SEED


def _latent_centers(dim: int, n_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Spread centres on a scaled sphere-ish cloud so clusters are separable.
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers *= 6.0
    return centers


# Fixed logical block size for generation. Vector content is keyed to the block
# a global row falls in — NOT to the caller's streaming batch_size — so row i is
# byte-identical no matter how the stream is chunked. This is what lets the index
# (built at one batch_size) and the brute-force ground truth (built at another)
# agree on what "vector i" is; keying the RNG on a running batch counter instead
# silently generates different data per chunking and destroys recall at scale.
_GEN_BLOCK = 100_000


def _generate_block(
    block_idx: int, dim: int, n_clusters: int, cluster_std: float,
    centers: np.ndarray, seed: int,
) -> np.ndarray:
    """Deterministically generate one fixed-size logical block of vectors."""
    rng = np.random.default_rng((seed, block_idx))
    assign = rng.integers(0, n_clusters, size=_GEN_BLOCK)
    noise = rng.standard_normal((_GEN_BLOCK, dim)).astype(np.float32) * cluster_std
    return (centers[assign] + noise).astype(np.float32)


def stream_vectors(
    n: int,
    dim: int = 128,
    *,
    batch_size: int = 50_000,
    n_clusters: int = 1024,
    cluster_std: float = 1.0,
    seed: int = SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(vectors, ids)`` batches summing to ``n`` rows.

    Content is a pure function of the global row index (via fixed logical blocks),
    so the stream is byte-for-byte identical for any ``batch_size``. Peak memory is
    ``O((batch_size + _GEN_BLOCK) * dim)`` regardless of ``n`` — the mechanism that
    lets index construction scale to a billion vectors.
    """
    centers = _latent_centers(dim, n_clusters, seed)
    produced = 0
    while produced < n:
        b = min(batch_size, n - produced)
        need_end = produced + b
        parts: list[np.ndarray] = []
        pos = produced
        while pos < need_end:
            blk = pos // _GEN_BLOCK
            blk_start = blk * _GEN_BLOCK
            block = _generate_block(blk, dim, n_clusters, cluster_std, centers, seed)
            lo = pos - blk_start
            hi = min(_GEN_BLOCK, need_end - blk_start)
            parts.append(block[lo:hi])
            pos = blk_start + hi
        vecs = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        ids = np.arange(produced, produced + b, dtype=np.int64)
        yield np.ascontiguousarray(vecs, dtype=np.float32), ids
        produced += b


def materialize(
    n: int, dim: int = 128, *, seed: int = SEED, **kw
) -> tuple[np.ndarray, np.ndarray]:
    """Collect a stream into memory (for small/medium sets and tests)."""
    vecs, ids = [], []
    for v, i in stream_vectors(n, dim, seed=seed, **kw):
        vecs.append(v)
        ids.append(i)
    return np.vstack(vecs), np.concatenate(ids)


def make_queries(
    nq: int, dim: int = 128, *, n_clusters: int = 1024, seed: int = SEED
) -> np.ndarray:
    """Query vectors drawn from the same distribution (disjoint RNG stream)."""
    centers = _latent_centers(dim, n_clusters, seed)
    # Disjoint from the database stream (which keys RNG on small batch indices).
    rng = np.random.default_rng((seed, 2_000_000_001))
    assign = rng.integers(0, n_clusters, size=nq)
    noise = rng.standard_normal((nq, dim)).astype(np.float32)
    return np.ascontiguousarray(centers[assign] + noise, dtype=np.float32)
