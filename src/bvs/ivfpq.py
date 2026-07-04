"""IVF-PQ — the flagship billion-scale index.

Combines a k-means **coarse quantizer** (IVF, ``nlist`` cells) with **Product
Quantization of the residuals**. For each database vector we:

1. find its nearest coarse centroid ``c`` (the cell it lives in);
2. compute the residual ``r = x - c``;
3. PQ-encode ``r`` into ``M`` bytes and append it to cell ``c``'s inverted list.

Encoding residuals rather than raw vectors concentrates PQ's precision where it
matters (residuals have far smaller variance), which is the standard FAISS recipe.

**Search.** For a query ``q`` we rank the coarse cells, take the ``nprobe`` closest,
and for each probed cell ``c`` form the query residual ``q - c`` and a per-subspace
ADC lookup table. Scanning a cell is then ``M`` table lookups per stored code — no
decompression. Candidates from all probed cells are merged and the top-``k`` byapproximate distance are returned.

Peak build memory is bounded by the codebooks + one batch + the compact codes, so
the same code path scales from 10k to 1B vectors (see ``bvs.memory``).
"""
from __future__ import annotations

import numpy as np

from . import SEED
from .kmeans import assign, train_kmeans
from .pq import ProductQuantizer


class IVFPQIndex:
    def __init__(
        self,
        dim: int,
        nlist: int = 256,
        m: int = 16,
        ksub: int = 256,
        *,
        seed: int = SEED,
    ) -> None:
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.ksub = ksub
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.pq = ProductQuantizer(dim, m, ksub, seed=seed)
        self._list_codes: list[list[np.ndarray]] = [[] for _ in range(nlist)]
        self._list_ids: list[list[np.ndarray]] = [[] for _ in range(nlist)]
        self._ntotal = 0

    @property
    def ntotal(self) -> int:
        return self._ntotal

    @property
    def is_trained(self) -> bool:
        return self.centroids is not None and self.pq.codebooks is not None

    # --- build -------------------------------------------------------------
    def train(self, x: np.ndarray) -> "IVFPQIndex":
        """Train coarse quantizer, then PQ on the coarse residuals of ``x``."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.centroids = train_kmeans(x, self.nlist, seed=self.seed)
        self.nlist = self.centroids.shape[0]
        while len(self._list_codes) < self.nlist:
            self._list_codes.append([])
            self._list_ids.append([])
        cells = assign(x, self.centroids)
        residuals = x - self.centroids[cells]
        self.pq.train(residuals)
        return self

    def add(self, x: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Assign, residual-encode and append a batch (streaming-safe)."""
        if not self.is_trained:
            raise RuntimeError("call train() before add()")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if ids is None:
            ids = np.arange(self._ntotal, self._ntotal + x.shape[0], dtype=np.int64)
        cells = assign(x, self.centroids)
        residuals = x - self.centroids[cells]
        codes = self.pq.encode(residuals)  # (b, m) uint8

        order = np.argsort(cells, kind="stable")
        cells_s, codes_s, ids_s = cells[order], codes[order], ids[order]
        boundaries = np.flatnonzero(np.diff(cells_s)) + 1
        for grp_codes, grp_ids, cell in zip(
            np.split(codes_s, boundaries),
            np.split(ids_s, boundaries),
            cells_s[np.concatenate([[0], boundaries])],
        ):
            self._list_codes[cell].append(grp_codes)
            self._list_ids[cell].append(grp_ids)
        self._ntotal += x.shape[0]

    def _consolidate(self) -> None:
        for c in range(self.nlist):
            if len(self._list_codes[c]) > 1:
                self._list_codes[c] = [np.vstack(self._list_codes[c])]
                self._list_ids[c] = [np.concatenate(self._list_ids[c])]

    def list_size(self, cell: int) -> int:
        return int(sum(cd.shape[0] for cd in self._list_codes[cell]))

    def code_storage_bytes(self) -> int:
        """Bytes held in inverted lists: PQ codes (M/vec) + int64 ids (8/vec)."""
        return self._ntotal * (self.m + 8)

    # --- search ------------------------------------------------------------
    def search(
        self, queries: np.ndarray, k: int = 10, nprobe: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("index not trained")
        self._consolidate()
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        nq = queries.shape[0]
        nprobe = min(nprobe, self.nlist)

        c_sq = np.einsum("ij,ij->i", self.centroids, self.centroids)
        coarse = c_sq[None, :] - 2.0 * (queries @ self.centroids.T)  # (nq, nlist)
        probe_cells = np.argpartition(coarse, nprobe - 1, axis=1)[:, :nprobe]

        out_d = np.full((nq, k), np.inf, dtype=np.float32)
        out_i = np.full((nq, k), -1, dtype=np.int64)

        for qi in range(nq):
            cells = probe_cells[qi]
            # Query residuals against each probed centroid, then ADC tables.
            resid = queries[qi][None, :] - self.centroids[cells]     # (nprobe, dim)
            tables = self.pq.compute_distance_tables(resid)          # (nprobe, m, ksub)

            cand_d, cand_id = [], []
            for j, cell in enumerate(cells):
                if not self._list_codes[cell]:
                    continue
                codes = self._list_codes[cell][0]                    # (L, m)
                cand_d.append(ProductQuantizer.adc_distances(tables[j], codes))
                cand_id.append(self._list_ids[cell][0])
            if not cand_d:
                continue
            d = np.concatenate(cand_d)
            ids = np.concatenate(cand_id)
            kk = min(k, d.shape[0])
            top = np.argpartition(d, kk - 1)[:kk]
            top = top[np.argsort(d[top])]
            out_d[qi, :kk] = d[top]
            out_i[qi, :kk] = ids[top]
        return out_d, out_i
