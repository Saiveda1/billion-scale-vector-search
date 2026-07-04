"""Recall and latency metrics for ANN evaluation."""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np


def recall_at_k(
    approx_ids: np.ndarray, truth_ids: np.ndarray, k: int | None = None
) -> float:
    """Mean recall@k = |approx_topk ∩ truth_topk| / k, averaged over queries.

    Both inputs are ``(nq, >=k)`` arrays of neighbour ids (rank order). This is the
    standard ANN recall: what fraction of the true k nearest neighbours the
    approximate index retrieved in its own top-k.
    """
    approx_ids = np.asarray(approx_ids)
    truth_ids = np.asarray(truth_ids)
    if k is None:
        k = truth_ids.shape[1]
    hits = 0
    for a_row, t_row in zip(approx_ids[:, :k], truth_ids[:, :k]):
        hits += np.intersect1d(a_row, t_row).size
    return hits / (approx_ids.shape[0] * k)


def latency_percentiles(latencies_ms: np.ndarray) -> dict[str, float]:
    lat = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(np.mean(lat)),
        "qps": float(1000.0 / np.mean(lat)) if np.mean(lat) > 0 else float("inf"),
    }


def per_query_latencies_ms(
    search_fn, queries: np.ndarray, *, warmup: int = 3
) -> np.ndarray:
    """Time ``search_fn(query_2d)`` one query at a time; return ms array.

    A few warmup queries prime caches / JIT-less numpy buffers so percentiles
    reflect steady state.
    """
    queries = np.atleast_2d(queries)
    for i in range(min(warmup, queries.shape[0])):
        search_fn(queries[i : i + 1])
    lat = np.empty(queries.shape[0], dtype=np.float64)
    for i in range(queries.shape[0]):
        t0 = time.perf_counter()
        search_fn(queries[i : i + 1])
        lat[i] = (time.perf_counter() - t0) * 1000.0
    return lat


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = {"s": 0.0}

    def read() -> float:
        return time.perf_counter() - start

    try:
        yield read
    finally:
        elapsed["s"] = time.perf_counter() - start
