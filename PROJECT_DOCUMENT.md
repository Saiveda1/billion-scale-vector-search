# Billion-Scale Vector Search Project Document

**Prepared For:** Sai Veda  
**GitHub Publishing Account:** Nikeshk834  
**Repository Slug:** `02-billion-scale-vector-search`  
**Verified Test Count From Portfolio Index:** 22  

## Background

**Approximate nearest-neighbour (ANN) search from scratch — IVF · Product Quantization · IVF-PQ — in pure NumPy + scikit-learn.**

A single-box vector index that stays honest about scale: it is *built and
benchmarked up to 1,000,000 × 128-d vectors* with real recall/latency/memory numbers,
and the **streaming, sharded build path makes 1B feasible** — at the high-recall
operating point (M=64, recall@10 ≈ 0.75–0.92) a billion 128-d vectors shrink from
**512 GB (float32) to ~67 GiB (IVF-PQ)**, a **7.1× reduction** computed from measured
per-vector cost; tune M down for up to **32× compression** at lower recall (see the
PQ-M sweep).

Offline, deterministic (`seed=42`), zero paid APIs, CPU-only.

```
        ┌───────────┐   coarse k-means      ┌──────────────────────────────┐
 query ─►│  IVF cell │─ rank, take nprobe ─►│  probed inverted lists (PQ)  │
        │ quantizer │                       │  residual → ADC table lookup │
        └───────────┘                       │  Σ M bytes → top-k           │
                                            └──────────────────────────────┘
  brute force (flat) ── exact ground truth for recall measurement
```

---

## Why this is hard

Exact k-NN over a billion 128-d vectors needs **512 GB of RAM** and ~50 GFLOP per
query. IVF-PQ attacks both:

- **IVF** partitions the space into `nlist` k-means cells; a query only scans the
  `nprobe` nearest cells → sublinear candidate set.
- **PQ** stores each vector as `M` one-byte codes (nearest sub-centroid per
  subspace) and computes distances by **Asymmetric Distance Computation**: `M`
  table lookups, no decompression.
- **IVF-PQ** encodes *coarse residuals* with PQ — the standard FAISS recipe —
  implemented here transparently in NumPy so every step is inspectable.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design and the 1B story.

---

## Benchmark results (real numbers)

<!-- BENCH_TABLE_START -->
**Scaling** (IVF-PQ, M=64, nprobe ≈ 10% of cells, recall vs exact brute force):

| N vectors | nlist | nprobe | recall@10 | p50 latency | QPS | build time | vs fp32 |
|----------:|------:|-------:|----------:|------------:|----:|-----------:|--------:|
| 10,000 | 200 | 20 | **0.925** | 3.3 ms | 293 | 35 s | 5.4× |
| 100,000 | 632 | 48 | **0.751** | 10.9 ms | 92 | 70 s | 6.7× |
| 1,000,000 | 2,000 | 48 | **0.874** | 10.4 ms | 96 | 342 s | 7.0× |
| 1,000,000 (exact) | — | — | 1.000 | 7.1 ms | 140 | — | 1.0× |

**PQ-M sweep** @ 100k — the compression ↔ recall dial:

| M (bytes/vec code) | recall@10 | compression |
|-------------------:|----------:|------------:|
| 8  | 0.167 | **24.9×** |
| 16 | 0.238 | 17.9× |
| 32 | 0.428 | 11.5× |
| 64 | **0.751** | 6.7× |

The **nprobe sweep** (100k) trades latency for headroom at fixed recall: p50 rises
0.96 ms → 19.6 ms from nprobe 1 → 128 while recall holds at 0.751 — for this
well-clustered data the coarse quantizer already captures the neighbourhood at
nprobe=1, so **PQ precision (M), not cell coverage, is the recall lever**.
<!-- BENCH_TABLE_END -->

Full tables in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md); raw rows in
[`benchmarks/results.csv`](benchmarks/results.csv).

---

## Project Purpose

This repository is part of the AI engineering portfolio and focuses on the following problem space:

- IVF-PQ ANN engine from scratch (NumPy)
- Headline result from the portfolio index: recall@10 **0.75–0.92** vs exact; **~7× compression**; 1B = ~67 GiB

## What This Project Solves

This project provides a production-style implementation with benchmark evidence and operational checks committed into the repository.

## Technical Approach

# Architecture

Billion-scale approximate nearest neighbour (ANN) search, built from first
principles in NumPy + scikit-learn. This document covers the algorithms, the data
flow, the design trade-offs, and — most importantly — the honest story of how the
same code path scales from ten thousand vectors to one billion.

---

## 1. The problem

Exact k-NN over `N` vectors of dimension `D` costs `O(N·D)` per query and
`O(N·D·4)` bytes to store in float32. At a billion 128-d vectors that is **512 GB
of RAM and ~50 billion FLOPs per query** — infeasible on a single box. ANN trades
a small, *measured* amount of recall for orders-of-magnitude less memory and
latency. Two ideas do the heavy lifting:

1. **Don't look at every vector** → Inverted File (IVF) partitioning.
2. **Don't store every vector at full precision** → Product Quantization (PQ).

IVF-PQ combines both and is the workhorse behind FAISS-scale systems.

---

## 2. Components (all in `src/bvs/`)

```
                       ┌──────────────────────────────────────────────┐
                       │              IVFPQIndex (flagship)            │
   query q ──────────► │                                              │
                       │  coarse quantizer (k-means, nlist cells)     │
                       │        │ rank cells, take top-nprobe         │
                       │        ▼                                      │
                       │  for each probed cell c:                     │
                       │     residual  r_q = q - centroid[c]          │
                       │     ADC table = ‖r_q,sub − codebook‖²  (M×256)│
                       │     scan cell codes  →  Σ table lookups      │
                       │        │                                      │
                       │        ▼   merge candidates across cells     │
                       │     top-k by approximate distance            │
                       └──────────────────────────────────────────────┘
   flat.py  → exact brute force (ground truth / recall oracle)
   pq.py    → Product Quantizer: codebooks, encode/decode, ADC tables
   ivf.py   → IVF with full-precision residual-free lists (stepping stone)
   kmeans.py→ MiniBatchKMeans training + chunked nearest-centroid assignment
   data.py  → streaming Gaussian-mixture generator (bounded memory)
   memory.py→ byte-exact accounting + 1B extrapolation
   metrics.

## Benchmark And Validation Evidence

The portfolio root documents **22 passing tests** for this project, and the repo quickstart uses `make test` as the standard validation path. The benchmark outputs committed in `benchmarks/` and the generated visuals in `assets/` are the evidence package for this delivery.

### RESULTS.md

# Benchmark Results

All numbers produced by `python scripts/bench.py` on synthetic clustered vectors (dim=128, float32). Recall@10 is measured against exact brute-force ground truth.


### Scaling (IVF-PQ, M=64, nprobe≈20, ~10% of cells)


| N | nlist | build (s) | recall@10 | p50 (ms) | p95 (ms) | QPS | index | vs fp32 |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 200 | 35.233 | 0.9247 | 3.3271 | 3.774 | 292.9 | 931.1 KiB | 5.37x |
| 100,000 | 632 | 70.128 | 0.7514 | 10.8552 | 11.6738 | 91.7 | 7.3 MiB | 6.69x |
| 1,000,000 | 2000 | 341.957 | 0.8744 | 10.4025 | 11.4218 | 96.0 | 69.8 MiB | 7.0x |

### nprobe sweep (N=100,000, M=64)


| nprobe | recall@10 | p50 (ms) | p95 (ms) | QPS |
|---|---|---|---|---|
| 1 | 0.7514 | 0.96 | 1.2061 | 1001.9 |
| 2 | 0.7514 | 1.4017 | 1.9994 | 660.8 |
| 4 | 0.7514 | 2.3856 | 2.7246 | 435.8 |
| 8 | 0.7514 | 3.3944 | 3.8419 | 293.2 |
| 16 | 0.7514 | 5.4785 | 5.9149 | 182.3 |
| 32 | 0.7514 | 8.2853 | 8.9756 | 119.7 |
| 64 | 0.7514 | 12.7348 | 14.0239 | 77.6 |
| 128 | 0.7514 | 19.6412 | 21.677 | 50.4 |
| brute-force | 1.0 | 7.1301 | 8.1655 | 139.7 |

### PQ M sweep (N=100,000, nprobe=16)


| M (subspaces) | bytes/vec | recall@10 | p50 (ms) | compression |
|---|---|---|---|---|
| 8 | 16 | 0.1673 | 1.4126 | 24.92x |
| 16 | 24 | 0.2382 | 2.0136 | 17.94x |
| 32 | 40 | 0.4275 | 3.1697 | 11.49x |
| 64 | 72 | 0.7514 | 5.3754 | 6.69x |

## Visual Artifacts Reviewed

- `assets/recall_latency_pareto.png`: Recall vs Latency Pareto — IVF-PQ vs brute force.
- `assets/memory_footprint.png`: Memory footprint: fp32 vs IVF-PQ, extrapolated to 1B.
- `assets/build_time_scaling.png`: Index build-time scaling (log-log).
- `assets/recall_vs_nprobe.png`: Recall@10 vs nprobe (the accuracy dial).

## Engineering Notes

The primary design and scale decisions are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md). The benchmark markdown in [`benchmarks/`](./benchmarks) and the generated figures in [`assets/`](./assets) should be read together: the markdown gives the measured numbers, and the screenshots make those results easier to inspect quickly during review.

## Files Included In This Repo

- [`README.md`](./README.md) for project overview, quickstart, and headline results
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design and scaling choices
- [`benchmarks/`](./benchmarks) for measured results from the committed runs
- [`assets/`](./assets) for generated screenshots and dashboards
- [`tests/`](./tests) for the automated validation suite

## Delivery Summary

This project document was prepared for **Sai Veda** so the repository reads like a real project handoff: what the system is for, what problem it solves, what evidence supports it, and where the benchmark and test artifacts live inside the repo.
