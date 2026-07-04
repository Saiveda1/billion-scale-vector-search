#!/usr/bin/env python3
"""Render the portfolio screenshots from REAL benchmark data.

Reads ``benchmarks/results.csv`` (produced by ``scripts/bench.py``) and renders
four professional panels into ``assets/``:

  1. recall_latency_pareto.png : Recall@10 vs p50 latency (IVF-PQ nprobe sweep)
                                 with the brute-force reference point.
  2. memory_footprint.png      : fp32 vs IVF-PQ bytes/vector and the 1B footprint.
  3. build_time_scaling.png    : index build time vs N (log-log).
  4. recall_vs_nprobe.png      : Recall@10 vs nprobe accuracy dial.
"""
from __future__ import annotations

import os

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bvs.memory import MemoryModel
from bvs.viztheme import (
    ACCENT,
    BAD,
    GOOD,
    MUTED,
    PALETTE,
    TEXT,
    WARN,
    apply_theme,
    save_panel,
)

ROOT = _bootstrap.ROOT
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "benchmarks", "results.csv")


def _load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    return df


def chart_pareto(df: pd.DataFrame) -> None:
    nprobe = df[df.experiment == "nprobe"].sort_values("p50_ms")
    base = df[df.experiment == "baseline"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nprobe["p50_ms"], nprobe["recall@10"], "-o", color=ACCENT,
            lw=2.2, ms=7, label="IVF-PQ (varying nprobe)")
    for _, r in nprobe.iterrows():
        ax.annotate(f"np={int(r['nprobe'])}", (r["p50_ms"], r["recall@10"]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=7.5, color=MUTED)
    if len(base):
        b = base.iloc[0]
        ax.scatter([b["p50_ms"]], [1.0], color=BAD, s=140, marker="*",
                   zorder=5, label="Brute force (exact)")
        ax.annotate("exact\n100% recall", (b["p50_ms"], 1.0),
                    textcoords="offset points", xytext=(-10, -34),
                    fontsize=8, color=BAD, ha="right")
    n = int(nprobe.iloc[0]["n"])
    ax.set_xscale("log")
    ax.set_xlabel("Query latency p50 (ms, log scale)")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Recall vs Latency Pareto — IVF-PQ vs brute force  (N={n:,}, dim=128)")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.03, "up-and-left is better", transform=ax.transAxes,
            fontsize=8, color=MUTED, style="italic")
    save_panel(fig, os.path.join(ASSETS, "recall_latency_pareto.png"))


def chart_memory(df: pd.DataFrame) -> None:
    m = int(df[df.experiment == "scaling"]["m"].iloc[0])
    nlist = int(df[df.experiment == "scaling"]["nlist"].max())
    model = MemoryModel(dim=128, nlist=nlist, m=m)
    fp32_bpv = model.flat_total(1)
    pq_bpv = m + 8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: bytes per vector.
    labels = ["fp32 flat", f"IVF-PQ (M={m})"]
    vals = [fp32_bpv, pq_bpv]
    bars = ax1.bar(labels, vals, color=[MUTED, GOOD], width=0.6)
    ax1.set_ylabel("Bytes per vector")
    ax1.set_title("Per-vector footprint")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + fp32_bpv * 0.02,
                 f"{v} B", ha="center", color=TEXT, fontweight="bold")
    ax1.text(1, pq_bpv + fp32_bpv * 0.10,
             f"{fp32_bpv / pq_bpv:.1f}x\nsmaller", ha="center",
             color=GOOD, fontweight="bold", fontsize=11)

    # Right: total footprint at scale (log), highlight 1B.
    ns = np.array([1e6, 1e7, 1e8, 1e9])
    flat_gib = np.array([model.flat_total(int(n)) / 1024**3 for n in ns])
    pq_gib = np.array([model.ivfpq_total(int(n)) / 1024**3 for n in ns])
    x = np.arange(len(ns))
    w = 0.38
    ax2.bar(x - w / 2, flat_gib, w, color=MUTED, label="fp32 flat")
    ax2.bar(x + w / 2, pq_gib, w, color=GOOD, label="IVF-PQ")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["1M", "10M", "100M", "1B"])
    ax2.set_ylabel("Total index size (GiB, log)")
    ax2.set_title("Footprint scaling — the 1B story")
    ax2.legend(loc="upper left")
    info = model.extrapolate(1_000_000_000)
    ax2.annotate(
        f"1B vectors:\nfp32 = {info['flat_gib']:.0f} GiB\nIVF-PQ = {info['ivfpq_gib']:.0f} GiB",
        (3, pq_gib[-1]), textcoords="offset points", xytext=(-4, 18),
        fontsize=8.5, color=WARN, ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="#161b22", ec=WARN, lw=1),
    )
    save_panel(fig, os.path.join(ASSETS, "memory_footprint.png"),
               suptitle="Memory: fp32 vs IVF-PQ (extrapolated to 1B vectors)")


def chart_build_scaling(df: pd.DataFrame) -> None:
    s = df[df.experiment == "scaling"].sort_values("n")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s["n"], s["build_s"], "-o", color=PALETTE[4], lw=2.2, ms=8,
            label="IVF-PQ build (train + stream-add)")
    # Linear-in-N reference line anchored at the first point.
    n0, t0 = s["n"].iloc[0], s["build_s"].iloc[0]
    ref_n = np.array(s["n"], dtype=float)
    ax.plot(ref_n, t0 * ref_n / n0, "--", color=MUTED, lw=1.3,
            label="linear O(N) reference")
    for _, r in s.iterrows():
        ax.annotate(f"{r['build_s']:.1f}s", (r["n"], r["build_s"]),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=8, color=TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of vectors N (log)")
    ax.set_ylabel("Build time (s, log)")
    ax.set_title("Index build-time scaling (log-log)")
    ax.legend(loc="upper left")
    save_panel(fig, os.path.join(ASSETS, "build_time_scaling.png"))


def chart_recall_nprobe(df: pd.DataFrame) -> None:
    np_df = df[df.experiment == "nprobe"].sort_values("nprobe")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np_df["nprobe"], np_df["recall@10"], "-o", color=GOOD, lw=2.4, ms=7)
    ax.axhline(1.0, color=BAD, ls="--", lw=1.2, alpha=0.8)
    ax.text(np_df["nprobe"].max(), 1.0, " exact = 1.0", color=BAD,
            va="bottom", ha="right", fontsize=8)
    ax.set_xscale("log", base=2)
    n = int(np_df.iloc[0]["n"])
    nlist = int(np_df.iloc[0]["nlist"])
    ax.set_xlabel("nprobe (cells scanned, log2)")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Recall@10 vs nprobe  (N={n:,}, nlist={nlist}, M={int(np_df.iloc[0]['m'])})")
    ax.set_ylim(min(0.0, np_df["recall@10"].min() - 0.05), 1.03)
    for _, r in np_df.iterrows():
        ax.annotate(f"{r['recall@10']:.2f}", (r["nprobe"], r["recall@10"]),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=7.5, color=MUTED, ha="center")
    save_panel(fig, os.path.join(ASSETS, "recall_vs_nprobe.png"))


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    apply_theme()
    df = _load()
    chart_pareto(df)
    chart_memory(df)
    chart_build_scaling(df)
    chart_recall_nprobe(df)
    print(f"wrote 4 PNGs to {ASSETS}")


if __name__ == "__main__":
    main()
