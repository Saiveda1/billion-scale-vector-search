"""Inverted File (IVF) index with a k-means coarse quantizer.

The vector space is partitioned into ``nlist`` Voronoi cells by a k-means coarse
quantizer. Each database vector is assigned to its nearest cell and stored in that
cell's inverted list. A query is compared only against the ``nprobe`` nearest
cells, turning an O(N) scan into O(N * nprobe / nlist) on average.

This variant keeps full fp32 vectors inside each list (exact re-ranking within the
probed cells). It is the stepping stone to :class:`~bvs.ivfpq.IVFPQIndex`, which
replaces the stored vectors with PQ codes for billion-scale compression.
"""
from __future__ import annotations

import numpy as np

from . import SEED
from .kmeans import assign, train_kmeans


class IVFIndex:
    def __init__(self, dim: int, nlist: int = 256, *, seed: int = SEED) -> None:
        self.dim = dim
        self.nlist = nlist
        self.seed = seed
        self.centroids: np.ndarray | None = None
        # Per-cell storage.
        self._list_vectors: list[list[np.ndarray]] = [[] for _ in range(nlist)]
        self._list_ids: list[list[np.ndarray]] = [[] for _ in range(nlist)]
        self._ntotal = 0

    @property
    def ntotal(self) -> int:
        return self._ntotal

    @property
    def is_trained(self) -> bool:
        return self.centroids is not None

    def train(self, x: np.ndarray) -> "IVFIndex":
        self.centroids = train_kmeans(x, self.nlist, seed=self.seed)
        self.nlist = self.centroids.shape[0]
        return self

    def add(self, x: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Assign a batch to cells and append to inverted lists (streaming-safe)."""
        if self.centroids is None:
            raise RuntimeError("call train() before add()")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if ids is None:
            ids = np.arange(self._ntotal, self._ntotal + x.shape[0], dtype=np.int64)
        cells = assign(x, self.centroids)
        order = np.argsort(cells, kind="stable")
        cells_s, x_s, ids_s = cells[order], x[order], ids[order]
        # Group contiguous runs of the same cell and extend that list once.
        boundaries = np.flatnonzero(np.diff(cells_s)) + 1
        for grp_x, grp_ids, cell in zip(
            np.split(x_s, boundaries),
            np.split(ids_s, boundaries),
            cells_s[np.concatenate([[0], boundaries])],
        ):
            self._list_vectors[cell].append(grp_x)
            self._list_ids[cell].append(grp_ids)
        self._ntotal += x.shape[0]

    def _consolidate(self) -> None:
        """Collapse per-cell chunk lists into single arrays (idempotent)."""
        for c in range(self.nlist):
            if len(self._list_vectors[c]) > 1:
                self._list_vectors[c] = [np.vstack(self._list_vectors[c])]
                self._list_ids[c] = [np.concatenate(self._list_ids[c])]

    def list_size(self, cell: int) -> int:
        return int(sum(v.shape[0] for v in self._list_vectors[cell]))

    def search(
        self, queries: np.ndarray, k: int = 10, nprobe: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.centroids is None:
            raise RuntimeError("index not trained")
        self._consolidate()
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        nq = queries.shape[0]
        nprobe = min(nprobe, self.nlist)

        # Rank cells per query by distance to coarse centroids.
        c_sq = np.einsum("ij,ij->i", self.centroids, self.centroids)
        coarse = c_sq[None, :] - 2.0 * (queries @ self.centroids.T)  # (nq, nlist)
        probe_cells = np.argpartition(coarse, nprobe - 1, axis=1)[:, :nprobe]

        out_d = np.full((nq, k), np.inf, dtype=np.float32)
        out_i = np.full((nq, k), -1, dtype=np.int64)

        for qi in range(nq):
            q = queries[qi]
            cand_v, cand_id = [], []
            for cell in probe_cells[qi]:
                if self._list_vectors[cell]:
                    cand_v.append(self._list_vectors[cell][0])
                    cand_id.append(self._list_ids[cell][0])
            if not cand_v:
                continue
            vecs = np.vstack(cand_v)
            cids = np.concatenate(cand_id)
            d = np.einsum("ij,ij->i", vecs, vecs) - 2.0 * (vecs @ q)  # drop ||q||^2
            kk = min(k, d.shape[0])
            top = np.argpartition(d, kk - 1)[:kk]
            top = top[np.argsort(d[top])]
            out_d[qi, :kk] = d[top] + float(q @ q)  # restore constant for true L2^2
            out_i[qi, :kk] = cids[top]
        return out_d, out_i
