"""Product Quantization (PQ) implemented from scratch in NumPy.

A D-dimensional vector is split into ``M`` contiguous sub-vectors of dimension
``Ds = D / M``. Each subspace gets its own codebook of ``ksub`` centroids learned
by k-means. A vector is then stored as ``M`` centroid indices — one byte each when
``ksub <= 256`` — giving a ``4 * D / M``-fold compression over fp32.

Search uses **Asymmetric Distance Computation (ADC)**: the query stays in full
precision. For a query we precompute, per subspace, a lookup table of squared
distances from the query sub-vector to every sub-centroid. The (approximate)
squared distance to any encoded database vector is then just ``M`` table lookups
summed — no decompression required.
"""
from __future__ import annotations

import numpy as np

from . import SEED
from .kmeans import train_kmeans


class ProductQuantizer:
    """Learns per-subspace codebooks and encodes vectors into compact byte codes.

    Parameters
    ----------
    m:
        Number of subspaces. ``dim`` must be divisible by ``m``.
    ksub:
        Centroids per subspace (<= 256 keeps codes at 1 byte/subspace).
    """

    def __init__(self, dim: int, m: int, ksub: int = 256, *, seed: int = SEED) -> None:
        if dim % m != 0:
            raise ValueError(f"dim {dim} not divisible by m {m}")
        if ksub > 256:
            raise ValueError("ksub > 256 would exceed 1 byte/subspace")
        self.dim = dim
        self.m = m
        self.ksub = ksub
        self.dsub = dim // m
        self.seed = seed
        # codebooks[m] : (ksub, dsub) float32
        self.codebooks: np.ndarray | None = None

    @property
    def code_dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    def _subspace(self, x: np.ndarray, i: int) -> np.ndarray:
        return x[:, i * self.dsub : (i + 1) * self.dsub]

    def train(self, x: np.ndarray) -> "ProductQuantizer":
        """Learn the ``m`` sub-codebooks from a training sample ``x``."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        books = np.empty((self.m, self.ksub, self.dsub), dtype=np.float32)
        for i in range(self.m):
            sub = self._subspace(x, i)
            books[i] = train_kmeans(sub, self.ksub, seed=self.seed + i)
        self.codebooks = books
        return self

    def encode(self, x: np.ndarray, *, chunk: int = 65536) -> np.ndarray:
        """Encode ``x`` into ``(n, m)`` uint8 codes (nearest sub-centroid ids)."""
        if self.codebooks is None:
            raise RuntimeError("call train() before encode()")
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = x.shape[0]
        codes = np.empty((n, self.m), dtype=np.uint8)
        for i in range(self.m):
            book = self.codebooks[i]  # (ksub, dsub)
            b_sq = np.einsum("ij,ij->i", book, book)  # (ksub,)
            for start in range(0, n, chunk):
                sub = self._subspace(x[start : start + chunk], i)
                # argmin over ||c||^2 - 2 x·c
                d = b_sq[None, :] - 2.0 * (sub @ book.T)
                codes[start : start + chunk, i] = np.argmin(d, axis=1).astype(np.uint8)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate fp32 vectors from codes ``(n, m)``."""
        if self.codebooks is None:
            raise RuntimeError("call train() before decode()")
        codes = np.asarray(codes, dtype=np.uint8)
        n = codes.shape[0]
        out = np.empty((n, self.dim), dtype=np.float32)
        for i in range(self.m):
            out[:, i * self.dsub : (i + 1) * self.dsub] = self.codebooks[i][codes[:, i]]
        return out

    def compute_distance_tables(self, queries: np.ndarray) -> np.ndarray:
        """Build ADC lookup tables for a batch of queries.

        Returns ``(nq, m, ksub)`` float32 of squared distances from each query
        sub-vector to every sub-centroid. Summing ``m`` looked-up entries yields
        the asymmetric squared distance to an encoded vector.
        """
        if self.codebooks is None:
            raise RuntimeError("call train() before compute_distance_tables()")
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        nq = queries.shape[0]
        tables = np.empty((nq, self.m, self.ksub), dtype=np.float32)
        for i in range(self.m):
            qsub = self._subspace(queries, i)          # (nq, dsub)
            book = self.codebooks[i]                    # (ksub, dsub)
            q_sq = np.einsum("ij,ij->i", qsub, qsub)[:, None]   # (nq, 1)
            b_sq = np.einsum("ij,ij->i", book, book)[None, :]   # (1, ksub)
            tables[:, i, :] = q_sq + b_sq - 2.0 * (qsub @ book.T)
        return tables

    @staticmethod
    def adc_distances(table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Asymmetric squared distances for one query.

        ``table`` : ``(m, ksub)`` for a single query.
        ``codes`` : ``(n, m)`` uint8 database codes.
        Returns ``(n,)`` summed lookup distances.
        """
        m = table.shape[0]
        # Gather table[i, codes[:, i]] for each subspace and sum.
        return table[np.arange(m), codes].sum(axis=1)

    # --- diagnostics -------------------------------------------------------
    def reconstruction_error(self, x: np.ndarray) -> float:
        """Mean squared reconstruction error over ``x`` (encode->decode)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        recon = self.decode(self.encode(x))
        return float(np.mean(np.sum((x - recon) ** 2, axis=1)))

    def code_bytes_per_vector(self) -> int:
        return self.m  # 1 byte per subspace
