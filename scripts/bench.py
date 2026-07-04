#!/usr/bin/env python3
"""End-to-end benchmark harness — produces benchmarks/results.csv + RESULTS.md.

Runs REAL measurements:

  * scaling    : build time / recall / latency / memory across N = 1e4, 1e5, 1e6.
  * nprobe     : recall@10 & latency vs nprobe (the accuracy/speed dial).
  * pq_m       : recall & compression vs number of PQ subspaces M.
  * baseline   : exact brute-force latency & memory for reference.

Ground truth for recall is exact brute force (bvs.flat), computed in DB chunks so
it never materialises an (nq x N) matrix.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import _bootstrap  # noqa: F401
import numpy as np

from bvs import SEED, FlatIndex
from bvs.data import make_queries, stream_vectors
from bvs.memory import MemoryModel
from bvs.metrics import latency_percentiles, per_query_latencies_ms, recall_at_k
from build_index import build_ivfpq_streaming

RESULTS_DIR = os.path.join(_bootstrap.ROOT, "benchmarks")
DIM = 128
N_CLUSTERS = 1024


def ground_truth(n: int, dim: int, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k ids via chunked brute force over a streamed database."""
    flat = FlatIndex(dim)
    for vecs, _ in stream_vectors(n, dim, batch_size=100_000, n_clusters=N_CLUSTERS, seed=SEED):
        flat.add(vecs)
    _, truth = flat.search(queries, k=k)
    return truth, flat


def nlist_for(n: int) -> int:
    """Heuristic nlist ~ 2*sqrt(N) (FAISS rule of thumb), clamped to sane bounds."""
    import math
    return int(min(2048, max(64, 2 * round(math.sqrt(n)))))


def run(args) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    k = args.k
    rows: list[dict] = []

    # ---- 1. Scaling sweep across dataset sizes ---------------------------
    print("== scaling sweep ==")
    for n in args.sizes:
        nq = args.queries if n < 1_000_000 else max(200, args.queries // 2)
        queries = make_queries(nq, DIM, n_clusters=N_CLUSTERS, seed=SEED)
        nlist = nlist_for(n)
        idx, bstats = build_ivfpq_streaming(
            n, DIM, nlist=nlist, m=args.m, batch_size=args.batch_size, verbose=True
        )
        truth, flat = ground_truth(n, DIM, queries, k)

        # Probe a fraction of cells (~10%) so coarse recall stays high as nlist
        # grows (nlist ~ 2*sqrt(N)); cap it because coarse recall saturates well
        # before then and the PQ code precision (m) is the real recall ceiling.
        nprobe = min(48, max(16, round(nlist * 0.10)))
        _, approx = idx.search(queries, k=k, nprobe=nprobe)
        rec = recall_at_k(approx, truth, k)
        lat = per_query_latencies_ms(lambda q: idx.search(q, k=k, nprobe=nprobe), queries)
        lp = latency_percentiles(lat)
        # Brute-force latency on the same queries (subset for the big set).
        flat_lat = per_query_latencies_ms(
            lambda q: flat.search(q, k=k), queries[: min(nq, 200)]
        )
        flp = latency_percentiles(flat_lat)

        rows.append({
            "experiment": "scaling", "n": n, "dim": DIM, "nlist": idx.nlist,
            "m": args.m, "nprobe": nprobe, "recall@10": round(rec, 4),
            "p50_ms": round(lp["p50_ms"], 4), "p95_ms": round(lp["p95_ms"], 4),
            "qps": round(lp["qps"], 1), "build_s": round(bstats["build_s"], 3),
            "index_bytes": bstats["index_bytes"], "flat_bytes": bstats["flat_bytes"],
            "compression": round(bstats["compression"], 2),
            "flat_p50_ms": round(flp["p50_ms"], 4), "flat_p95_ms": round(flp["p95_ms"], 4),
        })
        del flat, truth

    # ---- 2. nprobe sweep (accuracy/latency dial) at a fixed medium N ------
    print("== nprobe sweep ==")
    n = args.sweep_n
    nq = args.queries
    queries = make_queries(nq, DIM, n_clusters=N_CLUSTERS, seed=SEED)
    nlist = nlist_for(n)
    idx, bstats = build_ivfpq_streaming(n, DIM, nlist=nlist, m=args.m,
                                        batch_size=args.batch_size, verbose=True)
    truth, flat = ground_truth(n, DIM, queries, k)
    for nprobe in [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= idx.nlist]:
        _, approx = idx.search(queries, k=k, nprobe=nprobe)
        rec = recall_at_k(approx, truth, k)
        lat = per_query_latencies_ms(lambda q: idx.search(q, k=k, nprobe=nprobe), queries)
        lp = latency_percentiles(lat)
        rows.append({
            "experiment": "nprobe", "n": n, "dim": DIM, "nlist": idx.nlist,
            "m": args.m, "nprobe": nprobe, "recall@10": round(rec, 4),
            "p50_ms": round(lp["p50_ms"], 4), "p95_ms": round(lp["p95_ms"], 4),
            "qps": round(lp["qps"], 1), "build_s": round(bstats["build_s"], 3),
            "index_bytes": bstats["index_bytes"], "flat_bytes": bstats["flat_bytes"],
            "compression": round(bstats["compression"], 2),
            "flat_p50_ms": "", "flat_p95_ms": "",
        })
    # Brute-force reference point at this N.
    flat_lat = per_query_latencies_ms(lambda q: flat.search(q, k=k), queries[:200])
    flp = latency_percentiles(flat_lat)
    rows.append({
        "experiment": "baseline", "n": n, "dim": DIM, "nlist": "", "m": "",
        "nprobe": "", "recall@10": 1.0, "p50_ms": round(flp["p50_ms"], 4),
        "p95_ms": round(flp["p95_ms"], 4), "qps": round(flp["qps"], 1),
        "build_s": 0.0, "index_bytes": MemoryModel(DIM, 1, args.m).flat_total(n),
        "flat_bytes": MemoryModel(DIM, 1, args.m).flat_total(n), "compression": 1.0,
        "flat_p50_ms": "", "flat_p95_ms": "",
    })

    # ---- 3. PQ M sweep (compression vs recall) ---------------------------
    print("== pq_m sweep ==")
    for m in args.m_values:
        idxm, bm = build_ivfpq_streaming(n, DIM, nlist=nlist, m=m,
                                         batch_size=args.batch_size, verbose=True)
        _, approx = idxm.search(queries, k=k, nprobe=min(16, idxm.nlist))
        rec = recall_at_k(approx, truth, k)
        lat = per_query_latencies_ms(
            lambda q: idxm.search(q, k=k, nprobe=min(16, idxm.nlist)), queries
        )
        lp = latency_percentiles(lat)
        rows.append({
            "experiment": "pq_m", "n": n, "dim": DIM, "nlist": idxm.nlist,
            "m": m, "nprobe": min(16, idxm.nlist), "recall@10": round(rec, 4),
            "p50_ms": round(lp["p50_ms"], 4), "p95_ms": round(lp["p95_ms"], 4),
            "qps": round(lp["qps"], 1), "build_s": round(bm["build_s"], 3),
            "index_bytes": bm["index_bytes"], "flat_bytes": bm["flat_bytes"],
            "compression": round(bm["compression"], 2),
            "flat_p50_ms": "", "flat_p95_ms": "",
        })
        del idxm

    write_outputs(rows)


def write_outputs(rows: list[dict]) -> None:
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    write_markdown(rows)


def write_markdown(rows: list[dict]) -> None:
    md = os.path.join(RESULTS_DIR, "RESULTS.md")
    from bvs.memory import human_bytes

    def sec(title):
        return f"\n### {title}\n\n"

    lines = ["# Benchmark Results\n",
             "All numbers produced by `python scripts/bench.py` on synthetic "
             "clustered vectors (dim=128, float32). Recall@10 is measured against "
             "exact brute-force ground truth.\n"]

    scaling = [r for r in rows if r["experiment"] == "scaling"]
    _m = scaling[0]["m"] if scaling else 64
    _np = scaling[0]["nprobe"] if scaling else 48
    lines.append(sec(f"Scaling (IVF-PQ, M={_m}, nprobe≈{_np}, ~10% of cells)"))
    lines.append("| N | nlist | build (s) | recall@10 | p50 (ms) | p95 (ms) | QPS | index | vs fp32 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in scaling:
        lines.append(
            f"| {int(r['n']):,} | {r['nlist']} | {r['build_s']} | {r['recall@10']} | "
            f"{r['p50_ms']} | {r['p95_ms']} | {r['qps']} | "
            f"{human_bytes(r['index_bytes'])} | {r['compression']}x |"
        )

    nprobe = [r for r in rows if r["experiment"] == "nprobe"]
    base = [r for r in rows if r["experiment"] == "baseline"]
    if nprobe:
        n = nprobe[0]["n"]
        lines.append(sec(f"nprobe sweep (N={int(n):,}, M={nprobe[0]['m']})"))
        lines.append("| nprobe | recall@10 | p50 (ms) | p95 (ms) | QPS |")
        lines.append("|---|---|---|---|---|")
        for r in nprobe:
            lines.append(f"| {r['nprobe']} | {r['recall@10']} | {r['p50_ms']} | {r['p95_ms']} | {r['qps']} |")
        if base:
            b = base[0]
            lines.append(f"| brute-force | 1.0 | {b['p50_ms']} | {b['p95_ms']} | {b['qps']} |")

    pqm = [r for r in rows if r["experiment"] == "pq_m"]
    if pqm:
        lines.append(sec(f"PQ M sweep (N={int(pqm[0]['n']):,}, nprobe=16)"))
        lines.append("| M (subspaces) | bytes/vec | recall@10 | p50 (ms) | compression |")
        lines.append("|---|---|---|---|---|")
        for r in pqm:
            lines.append(f"| {r['m']} | {int(r['m']) + 8} | {r['recall@10']} | {r['p50_ms']} | {r['compression']}x |")

    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    ap.add_argument("--sweep-n", type=int, default=100_000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--m-values", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()
    t0 = time.perf_counter()
    run(args)
    print(f"total bench wall time: {time.perf_counter() - t0:.1f}s")
