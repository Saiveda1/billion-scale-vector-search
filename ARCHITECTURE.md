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
   metrics.py→ recall@k, latency percentiles, timers
```

### 2.1 Coarse quantizer (IVF)

A k-means model with `nlist` centroids partitions the space into Voronoi cells.
Each database vector is assigned to its nearest centroid and lives in that cell's
inverted list. A query only scans the `nprobe` nearest cells, so the expected work
is `N · nprobe / nlist` distance evaluations instead of `N`. We pick
`nlist ≈ 4·√N` (a standard rule of thumb) so average list length stays roughly
constant as `N` grows.

`nprobe` is the **accuracy dial**: `nprobe=1` is fastest/lowest recall; scanning
all cells reproduces exact search (verified in `test_ivf.py`).

### 2.2 Product Quantization (PQ)

A `D`-dim vector is split into `M` contiguous sub-vectors of dim `D/M`. Each
subspace has its own k-means codebook of `ksub=256` centroids, so a sub-vector is
encoded by the **1-byte** id of its nearest sub-centroid. A whole vector becomes
`M` bytes — a `4·D/M`× compression over float32 (for `D=128, M=16` that is 32×
on the raw vector, before ids).

**Asymmetric Distance Computation (ADC).** At query time the query stays in full
precision. For each subspace we precompute a lookup table of squared distances
from the query sub-vector to all 256 sub-centroids (`M × 256` floats). The
approximate squared distance to *any* encoded database vector is then just `M`
table lookups summed — no decompression, and the expensive distance math is done
once per query instead of once per database vector. `pq.adc_distances` vectorizes
this over an entire inverted list.

### 2.3 IVF-PQ: PQ on residuals

Encoding raw vectors with PQ is wasteful — most of a vector's energy is explained
by its coarse centroid. Instead we PQ-encode the **residual** `r = x − centroid`.
Residuals have far smaller variance, so the same 16 bytes buy much lower
reconstruction error. Search mirrors this: for each probed cell the query residual
`q − centroid[c]` gets its own ADC table. This is exactly the FAISS `IVFPQ`
recipe, implemented transparently here.

---

## 3. Streaming / sharded build — how it reaches 1B

The build **never holds the corpus in RAM**. Two bounded-memory passes:

| Pass | What | Peak memory |
|------|------|-------------|
| Train | Draw a fixed sample (default 200k) from the head of the stream; fit coarse k-means + PQ codebooks on residuals | `sample × D × 4` (bounded, N-independent) |
| Add | Stream the full corpus in `batch_size` chunks; assign → residual → PQ-encode → append **compact codes** to inverted lists | `batch × D × 4` + growing codes (`N × (M+8)`) |

`scripts/generate_data.py` and `bvs.data.stream_vectors` are generators that yield
one batch at a time keyed by `(seed, batch_index)`, so an arbitrary `N` is
reproducible and never materialised. `scripts/build_index.py` consumes the stream.

Because the training set is a fixed-size sample and the per-vector stored state is
a small constant (`M` code bytes + an 8-byte id), **the only thing that grows with
N is the compact code array** — which is the whole point.

### The 1B footprint (computed, not guessed)

`bvs.memory.MemoryModel` gives byte-exact per-vector costs, unit-tested against
what the index actually allocates:

- float32 flat: `D·4 = 512` bytes/vector → **512 GB** at 1B.
- IVF-PQ (`M=16`): `16 + 8 = 24` bytes/vector → **~24 GB** at 1B (+ a few MB of
  codebooks).

That ~21× shrink is what turns an impossible single-box index into a feasible one.
To actually *store and serve* 1B, the inverted lists shard cleanly by cell across
machines (each shard is an independent set of cells); a query fans out only to the
shards owning its `nprobe` cells. Nothing in the algorithm requires global state
beyond the (tiny) shared codebooks.

### What we actually ran

We build and benchmark up to **1,000,000 vectors × 128-d** end to end with real
recall/latency/build numbers (see `benchmarks/RESULTS.md`). 1B is reported as an
**extrapolation from measured per-vector cost**, exactly as the portfolio
"impressive but truthful" rule requires — the streaming build path that makes it
possible is the same code exercised at 1M.

---

## 4. Design trade-offs & choices

- **MiniBatchKMeans** for the coarse quantizer and PQ codebooks: near-Lloyd
  quality at a fraction of the time/memory, which matters when `nlist` is in the
  thousands. Wrapped in `kmeans.train_kmeans` with a fixed `random_state` for
  determinism.
- **Chunked assignment / distance** everywhere (`kmeans.assign`, `flat._topk_l2`,
  `pq.encode`) so no operation materialises an `O(nq·N)` or `O(N·nlist)` matrix.
- **int64 external ids** stored per vector so results map back to caller ids after
  sharding; counted honestly in the memory model.
- **Squared-L2** throughout (monotonic with L2, avoids sqrt on the hot path). The
  `‖x‖² − 2x·c` expansion drops constants that don't affect argmin.
- **IVF (full-precision) kept as a separate class** — it's the pedagogical
  stepping stone and a useful exact-within-cell re-ranker, and it lets tests prove
  the coarse quantizer independently of PQ error.

### Accuracy knobs (measured in the benchmark)

| Knob | ↑ increases | Cost |
|------|-------------|------|
| `nprobe` | recall | latency (∝ cells scanned) |
| `M` (PQ subspaces) | recall / precision | memory (`M` bytes/vec), build time |
| `nlist` | selectivity (shorter lists) | more coarse-centroid comparisons |

---

## 5. Limitations & honest gaps

- Single-process, in-RAM inverted lists. Production 1B needs on-disk/mmap lists
  and multi-node sharding; the code is structured for it (cells are independent)
  but that orchestration layer is out of scope here.
- No graph index (HNSW) — IVF-PQ was chosen for its clean memory story at
  billion-scale. HNSW gives lower latency at higher memory and is complementary.
- Synthetic Gaussian-mixture data. Real embeddings have more structure; the recipe
  is identical but absolute recall numbers would differ.
- No SIMD/GPU; ADC scans are vectorized NumPy. That is enough to demonstrate the
  algorithm and the scaling laws, not to set latency records.
