#!/usr/bin/env python3
"""Build an IVF-PQ index from a bounded-memory stream and report build stats.

Demonstrates the streaming construction path: a small training sample is drawn
from the head of the stream to fit the coarse quantizer and PQ codebooks, then
the *entire* dataset is streamed batch-by-batch into the inverted lists. Peak RAM
is bounded by (codebooks + one batch + compact codes) — never the raw fp32 corpus.

Example:
    python scripts/build_index.py --rows 1000000 --dim 128 --nlist 1024 --m 16
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np

from bvs import SEED, IVFPQIndex
from bvs.data import stream_vectors
from bvs.memory import MemoryModel, human_bytes


def build_ivfpq_streaming(
    n: int,
    dim: int,
    *,
    nlist: int,
    m: int,
    ksub: int = 256,
    batch_size: int = 50_000,
    n_clusters: int = 1024,
    cluster_std: float = 1.0,
    train_sample: int = 200_000,
    seed: int = SEED,
    verbose: bool = False,
) -> tuple[IVFPQIndex, dict]:
    """Return a trained, populated IVF-PQ index and a stats dict."""
    # --- 1. Draw a bounded training sample from the head of the stream. ------
    sample_parts: list[np.ndarray] = []
    got = 0
    for vecs, _ in stream_vectors(
        min(train_sample, n), dim, batch_size=batch_size,
        n_clusters=n_clusters, cluster_std=cluster_std, seed=seed,
    ):
        sample_parts.append(vecs)
        got += vecs.shape[0]
        if got >= train_sample:
            break
    sample = np.vstack(sample_parts)

    idx = IVFPQIndex(dim, nlist=nlist, m=m, ksub=ksub, seed=seed)
    t_train = time.perf_counter()
    idx.train(sample)
    train_s = time.perf_counter() - t_train
    del sample, sample_parts

    # --- 2. Stream the full corpus into the inverted lists. ------------------
    t_add = time.perf_counter()
    for vecs, ids in stream_vectors(
        n, dim, batch_size=batch_size, n_clusters=n_clusters,
        cluster_std=cluster_std, seed=seed,
    ):
        idx.add(vecs, ids)
    idx._consolidate()
    add_s = time.perf_counter() - t_add

    model = MemoryModel(dim=dim, nlist=idx.nlist, m=m, ksub=ksub)
    stats = {
        "n": n,
        "dim": dim,
        "nlist": idx.nlist,
        "m": m,
        "train_s": train_s,
        "add_s": add_s,
        "build_s": train_s + add_s,
        "index_bytes": model.ivfpq_total(n),
        "flat_bytes": model.flat_total(n),
        "compression": model.compression_ratio(n),
    }
    if verbose:
        print(
            f"built IVF-PQ n={n:,} nlist={idx.nlist} m={m}: "
            f"train {train_s:.1f}s + add {add_s:.1f}s = {stats['build_s']:.1f}s | "
            f"index {human_bytes(stats['index_bytes'])} vs "
            f"flat {human_bytes(stats['flat_bytes'])} "
            f"({stats['compression']:.1f}x smaller)"
        )
    return idx, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()
    build_ivfpq_streaming(
        args.rows, args.dim, nlist=args.nlist, m=args.m,
        batch_size=args.batch_size, verbose=True,
    )


if __name__ == "__main__":
    main()
