from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bvs.data import make_queries, materialize  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    dim = 64
    vecs, ids = materialize(8_000, dim, batch_size=4_000, n_clusters=48, seed=42)
    queries = make_queries(100, dim, n_clusters=48, seed=42)
    return {"vecs": vecs, "ids": ids, "queries": queries, "dim": dim}


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
