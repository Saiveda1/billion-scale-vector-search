"""Billion-Scale Vector Search — approximate nearest neighbor (ANN) from scratch.

Offline, deterministic, dependency-light. Every component is real and implemented
in NumPy / scikit-learn:

- ``flat``   : exact brute-force search (ground truth for recall).
- ``pq``     : Product Quantization with asymmetric distance computation (ADC).
- ``ivf``    : Inverted File index with a k-means coarse quantizer.
- ``ivfpq``  : IVF + PQ on coarse residuals — the flagship billion-scale index.
- ``data``   : streaming, sharded synthetic vector generator (bounded memory).
- ``memory`` : byte-level memory accounting and 1B-vector extrapolation.
- ``metrics``: recall@k, latency percentiles, timing utilities.

The design mirrors FAISS' IVFPQ but is written from first principles so every
step (coarse assignment, residual PQ encoding, ADC lookup tables) is inspectable.
"""
from __future__ import annotations

__version__ = "1.0.0"

SEED = 42

from .flat import FlatIndex  # noqa: E402
from .pq import ProductQuantizer  # noqa: E402
from .ivf import IVFIndex  # noqa: E402
from .ivfpq import IVFPQIndex  # noqa: E402

__all__ = [
    "SEED",
    "FlatIndex",
    "ProductQuantizer",
    "IVFIndex",
    "IVFPQIndex",
]
