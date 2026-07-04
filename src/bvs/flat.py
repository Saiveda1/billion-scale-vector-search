"""Exact brute-force (flat) index — the ground truth for recall measurement.

Distances are computed in chunks over the database so that a query batch against
millions of vectors never materialises a full (nq x N) matrix in memory.
"""
from __future__ import annotations

import numpy as np


def _topk_l2(
    queries: np.ndarray,
    db: np.ndarray,
    k: int,
    *,
    db_chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) of the exact ``k`` nearest db rows per query.

    Squared-L2 distances. Computed by streaming the database in row chunks and
    keeping a running top-k, so peak memory is ``O(nq * db_chunk)`` not
    ``O(nq * N)``.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    nq = queries.shape[0]
    q_sq = np.einsum("ij,ij->i", queries, queries)[:, None]  # (nq, 1)

    best_d = np.full((nq, k), np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)

    for start in range(0, db.shape[0], db_chunk):
        block = np.ascontiguousarray(db[start : start + db_chunk], dtype=np.float32)
        b_sq = np.einsum("ij,ij->i", block, block)[None, :]  # (1, b)
        # (nq, b) squared distances.
        d = q_sq + b_sq - 2.0 * (queries @ block.T)
        idx = np.arange(start, start + block.shape[0], dtype=np.int64)

        # Merge this block's candidates with the running best, then re-select top-k.
        cat_d = np.concatenate([best_d, d], axis=1)
        cat_i = np.concatenate([best_i, np.broadcast_to(idx, (nq, idx.size))], axis=1)
        part = np.argpartition(cat_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(cat_d, part, axis=1)
        best_i = np.take_along_axis(cat_i, part, axis=1)

    # Final sort within the retained k for stable, ranked output.
    order = np.argsort(best_d, axis=1)
    best_d = np.take_along_axis(best_d, order, axis=1)
    best_i = np.take_along_axis(best_i, order, axis=1)
    return best_d, best_i


class FlatIndex:
    """Exact nearest-neighbour search by full L2 scan.

    Serves as the recall oracle for the approximate indexes and as a latency
    baseline. Holds the database in memory (fp32), so it is only used at the
    scales where that is affordable.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._db: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        return 0 if self._db is None else self._db.shape[0]

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {x.shape[1]}")
        self._db = x if self._db is None else np.vstack([self._db, x])

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self._db is None:
            raise RuntimeError("index is empty; call add() first")
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        k = min(k, self.ntotal)
        return _topk_l2(queries, self._db, k)
