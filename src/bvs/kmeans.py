"""Deterministic k-means helpers used by the coarse quantizer and PQ codebooks.

We wrap scikit-learn's ``MiniBatchKMeans`` so that training a coarse quantizer
with thousands of centroids, or 256-way PQ sub-codebooks, stays fast and memory
bounded on large samples. A tiny exact-assignment helper (``assign``) is provided
because assignment is on the hot path of both index build and search and we want a
chunked, allocation-friendly implementation rather than sklearn's ``predict``.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from . import SEED


def train_kmeans(
    x: np.ndarray,
    k: int,
    *,
    seed: int = SEED,
    max_iter: int = 25,
    batch_size: int | None = None,
    n_init: int = 1,
) -> np.ndarray:
    """Train ``k`` centroids on ``x`` and return them as float32 ``(k, d)``.

    Uses MiniBatchKMeans for speed. If there are fewer samples than requested
    centroids we simply return the unique samples padded by random points, which
    keeps small-data unit tests well defined.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    if k >= n:
        # Degenerate: not enough data to fit k clusters. Use the points we have
        # and pad by sampling with replacement so downstream shapes are valid.
        rng = np.random.default_rng(seed)
        pad = rng.integers(0, n, size=k - n) if k > n else np.empty(0, dtype=int)
        idx = np.concatenate([np.arange(n), pad]).astype(int)
        return x[idx].copy()

    if batch_size is None:
        batch_size = min(n, max(256, 10 * k))
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        max_iter=max_iter,
        batch_size=batch_size,
        n_init=3,
        reassignment_ratio=0.0,
    )
    km.fit(x)
    return km.cluster_centers_.astype(np.float32)


def assign(x: np.ndarray, centroids: np.ndarray, *, chunk: int = 16384) -> np.ndarray:
    """Assign each row of ``x`` to its nearest centroid (L2), chunked.

    Returns an int32 array of centroid indices. Uses the expansion
    ``||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2`` and drops the constant ``||x||^2``
    term so the argmin is computed from ``||c||^2 - 2 x·c`` only.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    c_sq = np.einsum("ij,ij->i", centroids, centroids)  # (k,)
    out = np.empty(x.shape[0], dtype=np.int32)
    for start in range(0, x.shape[0], chunk):
        xb = x[start : start + chunk]
        # (b, k) = ||c||^2 - 2 xb·c ; argmin equals argmin of full L2 distance.
        d = c_sq[None, :] - 2.0 * (xb @ centroids.T)
        out[start : start + chunk] = np.argmin(d, axis=1).astype(np.int32)
    return out
