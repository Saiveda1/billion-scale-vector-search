"""Byte-level memory accounting and 1-billion-vector extrapolation.

The whole point of IVF-PQ is that the *per-vector* footprint is a small constant
that does not depend on N, so we can compute the 1B footprint exactly from the
index parameters. These helpers are unit-tested against the actual bytes the
index allocates.
"""
from __future__ import annotations

from dataclasses import dataclass

GiB = 1024 ** 3
GB = 1000 ** 3

ID_BYTES = 8      # int64 external id stored per vector
FP32_BYTES = 4


def fp32_bytes_per_vector(dim: int) -> int:
    """Raw storage for one float32 vector (the flat baseline)."""
    return dim * FP32_BYTES


def pq_bytes_per_vector(m: int, *, store_ids: bool = True) -> int:
    """IVF-PQ per-vector storage: ``m`` code bytes + optional int64 id.

    The coarse cell assignment is implicit in the inverted-list layout, so it
    costs no per-vector bytes beyond the id.
    """
    return m + (ID_BYTES if store_ids else 0)


def codebook_bytes(dim: int, nlist: int, m: int, ksub: int = 256) -> int:
    """Fixed (N-independent) overhead: coarse centroids + PQ sub-codebooks."""
    coarse = nlist * dim * FP32_BYTES
    dsub = dim // m
    pq = m * ksub * dsub * FP32_BYTES
    return coarse + pq


@dataclass(frozen=True)
class MemoryModel:
    dim: int
    nlist: int
    m: int
    ksub: int = 256

    def flat_total(self, n: int) -> int:
        return n * fp32_bytes_per_vector(self.dim)

    def ivfpq_total(self, n: int) -> int:
        return n * pq_bytes_per_vector(self.m) + codebook_bytes(
            self.dim, self.nlist, self.m, self.ksub
        )

    def compression_ratio(self, n: int) -> float:
        return self.flat_total(n) / self.ivfpq_total(n)

    def extrapolate(self, n: int) -> dict[str, float]:
        """Human-readable footprint summary at scale ``n``."""
        flat = self.flat_total(n)
        pq = self.ivfpq_total(n)
        return {
            "n": n,
            "flat_bytes": flat,
            "ivfpq_bytes": pq,
            "flat_gib": flat / GiB,
            "ivfpq_gib": pq / GiB,
            "flat_gb": flat / GB,
            "ivfpq_gb": pq / GB,
            "compression_ratio": flat / pq,
            "fp32_bytes_per_vec": fp32_bytes_per_vector(self.dim),
            "pq_bytes_per_vec": pq_bytes_per_vector(self.m),
        }


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"
