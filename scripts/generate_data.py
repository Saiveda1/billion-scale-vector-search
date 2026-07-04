#!/usr/bin/env python3
"""Stream synthetic vectors to disk in bounded-memory shards.

Writes a directory of ``.npy`` shards plus a ``queries.npy`` file. The generator
never holds more than one batch in RAM, so ``--rows`` can be arbitrarily large
(the same mechanism that lets the index build scale to 1B vectors).

Example:
    python scripts/generate_data.py --rows 1000000 --dim 128 --out data/base
"""
from __future__ import annotations

import argparse
import os
import time

import _bootstrap  # noqa: F401
import numpy as np

from bvs import SEED
from bvs.data import make_queries, stream_vectors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--n-clusters", type=int, default=1024)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--out", type=str, default="data/base")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    written = 0
    peak_batch_mb = 0.0
    for i, (vecs, ids) in enumerate(
        stream_vectors(
            args.rows,
            args.dim,
            batch_size=args.batch_size,
            n_clusters=args.n_clusters,
            seed=args.seed,
        )
    ):
        np.save(os.path.join(args.out, f"shard_{i:05d}.npy"), vecs)
        written += vecs.shape[0]
        peak_batch_mb = max(peak_batch_mb, vecs.nbytes / 1e6)
    q = make_queries(args.queries, args.dim, n_clusters=args.n_clusters, seed=args.seed)
    np.save(os.path.join(args.out, "queries.npy"), q)

    dt = time.perf_counter() - t0
    print(f"wrote {written:,} vectors (dim={args.dim}) to {args.out} in {dt:.1f}s")
    print(f"peak in-RAM batch: {peak_batch_mb:.1f} MB  (rows fully streamed)")
    print(f"queries: {q.shape[0]:,} -> {args.out}/queries.npy")


if __name__ == "__main__":
    main()
