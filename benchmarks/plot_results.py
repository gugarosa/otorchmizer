"""Generate benchmark comparison plots.

Reads results.json produced by run_benchmarks.py and generates:
1. Speedup bar charts per optimizer
2. Scaling analysis (speedup vs population size)
3. Convergence quality scatter (numpy fitness vs torch fitness)
4. Memory usage comparison
5. Combined summary dashboard

Usage:
    python benchmarks/plot_results.py                      # default
    python benchmarks/plot_results.py --input results.json # custom input
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def pair_results(results):
    """Pair numpy and torch_cpu results by (optimizer, function, n_agents, n_variables)."""

    index = {}
    for r in results:
        key = (r["optimizer"], r["function"], r["n_agents"], r["n_variables"])
        index.setdefault(key, {})[r["backend"]] = r

    pairs = []
    for key, backends in index.items():
        if "numpy" in backends and "torch_cpu" in backends:
            pairs.append({
                "key": key,
                "numpy": backends["numpy"],
                "torch_cpu": backends["torch_cpu"],
                "torch_gpu": backends.get("torch_gpu"),
            })
    return pairs


# ============================================================================
# Plot 1: Speedup bar chart per optimizer (aggregated across functions)
# ============================================================================

def plot_speedup_bars(pairs, outdir):
    """Bar chart showing average speedup per optimizer across all configurations."""

    opt_speedups = defaultdict(list)
    for p in pairs:
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        speedup = np_t / tc_t if tc_t > 0 else 1.0
        opt_speedups[p["key"][0]].append(speedup)

    opts = sorted(opt_speedups.keys())
    means = [np.mean(opt_speedups[o]) for o in opts]
    stds = [np.std(opt_speedups[o]) for o in opts]
    mins_ = [np.min(opt_speedups[o]) for o in opts]
    maxs_ = [np.max(opt_speedups[o]) for o in opts]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(opts))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#2196F3", edgecolor="#1565C0",
                  alpha=0.85, zorder=3)

    # Add value labels
    for i, (bar, mn, mx) in enumerate(zip(bars, mins_, maxs_)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.1,
                f"{means[i]:.1f}×", ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"[{mn:.1f}×–{mx:.1f}×]", ha="center", va="center", fontsize=9, color="white")

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="parity (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels(opts, fontsize=12)
    ax.set_ylabel("Speedup (×)", fontsize=13)
    ax.set_title("Otorchmizer vs Opytimizer — Speedup by Algorithm (CPU)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_speedup_bars.png"), dpi=150)
    plt.close(fig)
    print("  → 01_speedup_bars.png")


# ============================================================================
# Plot 2: Speedup scaling with population size
# ============================================================================

def plot_scaling(pairs, outdir):
    """Line plot: speedup vs population size, one line per optimizer."""

    data = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        opt = p["key"][0]
        n = p["key"][2]  # n_agents
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        speedup = np_t / tc_t if tc_t > 0 else 1.0
        data[opt][n].append(speedup)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))

    for (opt, sizes), color in zip(sorted(data.items()), colors):
        ns = sorted(sizes.keys())
        means = [np.mean(sizes[n]) for n in ns]
        ax.plot(ns, means, "o-", color=color, label=opt, linewidth=2, markersize=8)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="parity")
    ax.set_xlabel("Population Size (n_agents)", fontsize=13)
    ax.set_ylabel("Speedup (×)", fontsize=13)
    ax.set_title("Speedup Scaling with Population Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_scaling_population.png"), dpi=150)
    plt.close(fig)
    print("  → 02_scaling_population.png")


# ============================================================================
# Plot 3: Speedup scaling with dimensions
# ============================================================================

def plot_dimension_scaling(pairs, outdir):
    """Line plot: speedup vs number of variables, one line per optimizer."""

    data = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        opt = p["key"][0]
        d = p["key"][3]  # n_variables
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        speedup = np_t / tc_t if tc_t > 0 else 1.0
        data[opt][d].append(speedup)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))

    for (opt, dims_data), color in zip(sorted(data.items()), colors):
        ds = sorted(dims_data.keys())
        means = [np.mean(dims_data[d]) for d in ds]
        ax.plot(ds, means, "s-", color=color, label=opt, linewidth=2, markersize=8)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="parity")
    ax.set_xlabel("Number of Variables (dimensions)", fontsize=13)
    ax.set_ylabel("Speedup (×)", fontsize=13)
    ax.set_title("Speedup Scaling with Problem Dimensionality", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_scaling_dimensions.png"), dpi=150)
    plt.close(fig)
    print("  → 03_scaling_dimensions.png")


# ============================================================================
# Plot 4: Convergence quality scatter
# ============================================================================

def plot_quality_scatter(pairs, outdir):
    """Scatter plot: numpy best fitness vs torch best fitness.

    Points on the diagonal = identical quality. Below = torch is better.
    Near-zero pairs (both < 1e-4) are treated as converged and plotted at a
    baseline to avoid float32/float64 precision artifacts dominating the plot.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    markers = {"PSO": "o", "WOA": "s", "FA": "D", "GA": "^", "HC": "v"}
    colors = {"sphere": "#2196F3", "rastrigin": "#4CAF50", "ackley": "#FF9800", "rosenbrock": "#F44336"}

    NEAR_ZERO = 1e-4

    for p in pairs:
        opt = p["key"][0]
        fn = p["key"][1]
        np_f = max(p["numpy"]["best_fitness"], NEAR_ZERO)
        tc_f = max(p["torch_cpu"]["best_fitness"], NEAR_ZERO)
        ax.scatter(np_f, tc_f,
                   marker=markers.get(opt, "o"),
                   color=colors.get(fn, "gray"),
                   alpha=0.6, s=60, edgecolors="white", linewidth=0.5)

    # Diagonal reference
    all_fits = []
    for p in pairs:
        all_fits.extend([max(p["numpy"]["best_fitness"], NEAR_ZERO),
                         max(p["torch_cpu"]["best_fitness"], NEAR_ZERO)])
    valid_fits = [f for f in all_fits if np.isfinite(f) and f > 0]
    if valid_fits:
        lo = min(valid_fits) * 0.5
        hi = max(valid_fits) * 2.0
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="parity line")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Build legend
    from matplotlib.lines import Line2D
    opt_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="None", markersize=8, label=o)
                   for o, m in markers.items()]
    fn_handles = [Line2D([0], [0], marker="o", color=c, linestyle="None", markersize=8, label=f)
                  for f, c in colors.items()]
    ax.legend(handles=opt_handles + fn_handles, fontsize=9, loc="upper left", ncol=2)

    ax.set_xlabel("Opytimizer (NumPy) — Best Fitness", fontsize=13)
    ax.set_ylabel("Otorchmizer (PyTorch) — Best Fitness", fontsize=13)
    ax.set_title("Convergence Quality Comparison", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_quality_scatter.png"), dpi=150)
    plt.close(fig)
    print("  → 04_quality_scatter.png")


# ============================================================================
# Plot 5: Memory usage comparison
# ============================================================================

def plot_memory(pairs, outdir):
    """Grouped bar chart: memory usage numpy vs torch, grouped by optimizer."""

    opt_mem = defaultdict(lambda: {"numpy": [], "torch": []})
    for p in pairs:
        opt = p["key"][0]
        opt_mem[opt]["numpy"].append(p["numpy"]["peak_memory_mb"])
        opt_mem[opt]["torch"].append(p["torch_cpu"]["peak_memory_mb"])

    opts = sorted(opt_mem.keys())
    np_means = [np.mean(opt_mem[o]["numpy"]) for o in opts]
    tc_means = [np.mean(opt_mem[o]["torch"]) for o in opts]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(opts))
    w = 0.35

    ax.bar(x - w / 2, np_means, w, label="Opytimizer (NumPy)", color="#FF7043", edgecolor="#D84315", alpha=0.85)
    ax.bar(x + w / 2, tc_means, w, label="Otorchmizer (PyTorch)", color="#42A5F5", edgecolor="#1565C0", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(opts, fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=13)
    ax.set_title("Memory Usage Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_memory_usage.png"), dpi=150)
    plt.close(fig)
    print("  → 05_memory_usage.png")


# ============================================================================
# Plot 6: Timing heatmap (optimizer × population size)
# ============================================================================

def plot_timing_heatmap(pairs, outdir):
    """Heatmap of speedup for each (optimizer, n_agents) combination."""

    data = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        opt = p["key"][0]
        n = p["key"][2]
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        speedup = np_t / tc_t if tc_t > 0 else 1.0
        data[opt][n].append(speedup)

    opts = sorted(data.keys())
    all_ns = sorted(set(n for o in data.values() for n in o.keys()))

    matrix = np.zeros((len(opts), len(all_ns)))
    for i, opt in enumerate(opts):
        for j, n in enumerate(all_ns):
            if n in data[opt]:
                matrix[i, j] = np.mean(data[opt][n])
            else:
                matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(all_ns)))
    ax.set_xticklabels([str(n) for n in all_ns], fontsize=11)
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels(opts, fontsize=12)
    ax.set_xlabel("Population Size", fontsize=13)
    ax.set_title("Speedup Heatmap (Optimizer × Population)", fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(opts)):
        for j in range(len(all_ns)):
            ax.text(j, i, f"{matrix[i, j]:.1f}×", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Speedup (×)", fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_speedup_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  → 06_speedup_heatmap.png")


# ============================================================================
# Plot 7: Wall-clock time comparison (absolute)
# ============================================================================

def plot_time_comparison(pairs, outdir):
    """Per-function faceted bar chart: absolute time numpy vs torch."""

    fns = sorted(set(p["key"][1] for p in pairs))
    opts = sorted(set(p["key"][0] for p in pairs))

    fig, axes = plt.subplots(1, len(fns), figsize=(4 * len(fns), 6), sharey=True)
    if len(fns) == 1:
        axes = [axes]

    for ax, fn_name in zip(axes, fns):
        fn_pairs = [p for p in pairs if p["key"][1] == fn_name]

        opt_times = defaultdict(lambda: {"numpy": [], "torch": []})
        for p in fn_pairs:
            opt = p["key"][0]
            opt_times[opt]["numpy"].append(p["numpy"]["time_seconds"])
            opt_times[opt]["torch"].append(p["torch_cpu"]["time_seconds"])

        local_opts = sorted(opt_times.keys())
        np_means = [np.mean(opt_times[o]["numpy"]) for o in local_opts]
        tc_means = [np.mean(opt_times[o]["torch"]) for o in local_opts]

        x = np.arange(len(local_opts))
        w = 0.35
        ax.bar(x - w / 2, np_means, w, label="NumPy", color="#FF7043", alpha=0.85)
        ax.bar(x + w / 2, tc_means, w, label="PyTorch", color="#42A5F5", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(local_opts, fontsize=10, rotation=45)
        ax.set_title(fn_name.capitalize(), fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Wall-Clock Time (seconds)", fontsize=13)
    axes[0].legend(fontsize=10)
    fig.suptitle("Absolute Execution Time by Function", fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "07_time_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 07_time_comparison.png")


# ============================================================================
# Plot 8: Summary dashboard
# ============================================================================

def plot_dashboard(pairs, outdir):
    """2x2 dashboard combining key metrics."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top-left: Average speedup bars ──
    opt_speedups = defaultdict(list)
    for p in pairs:
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        speedup = np_t / tc_t if tc_t > 0 else 1.0
        opt_speedups[p["key"][0]].append(speedup)

    opts = sorted(opt_speedups.keys())
    means = [np.mean(opt_speedups[o]) for o in opts]
    ax1.barh(opts, means, color="#2196F3", edgecolor="#1565C0", alpha=0.85)
    ax1.axvline(x=1.0, color="red", linestyle="--", alpha=0.7)
    for i, v in enumerate(means):
        ax1.text(v + 0.05, i, f"{v:.1f}×", va="center", fontweight="bold")
    ax1.set_xlabel("Speedup (×)")
    ax1.set_title("Avg Speedup per Optimizer", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # ── Top-right: Quality ratio (tolerance-aware) ──
    # When both fitnesses are near-zero (< 1e-4), they are considered equal
    # regardless of float32/float64 precision differences.
    opt_quality = defaultdict(list)
    for p in pairs:
        np_f = p["numpy"]["best_fitness"]
        tc_f = p["torch_cpu"]["best_fitness"]
        if np.isfinite(np_f) and np.isfinite(tc_f):
            denom = max(abs(np_f), abs(tc_f), 1e-4)
            ratio = tc_f / denom if denom > 0 else 1.0
            np_ratio = np_f / denom if denom > 0 else 1.0
            # Compute relative quality: values near 1.0 = parity
            if abs(np_f) < 1e-4 and abs(tc_f) < 1e-4:
                opt_quality[p["key"][0]].append(1.0)
            elif np_f > 0:
                opt_quality[p["key"][0]].append(tc_f / np_f)

    opts_q = sorted(opt_quality.keys())
    medians_q = [np.median(opt_quality[o]) for o in opts_q]
    colors_q = ["#4CAF50" if m <= 1.05 else "#FF9800" if m <= 1.5 else "#F44336" for m in medians_q]
    ax2.barh(opts_q, medians_q, color=colors_q, edgecolor="gray", alpha=0.85)
    ax2.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
    for i, v in enumerate(medians_q):
        ax2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)
    ax2.set_xlabel("Fitness Ratio (torch / numpy)")
    ax2.set_title("Quality Ratio (≤1.0 = torch better)", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # ── Bottom-left: Scaling ──
    scaling_data = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        opt = p["key"][0]
        n = p["key"][2]
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        scaling_data[opt][n].append(np_t / tc_t if tc_t > 0 else 1.0)

    cm = plt.cm.Set1(np.linspace(0, 1, len(scaling_data)))
    for (opt, sizes), color in zip(sorted(scaling_data.items()), cm):
        ns = sorted(sizes.keys())
        m = [np.mean(sizes[n]) for n in ns]
        ax3.plot(ns, m, "o-", color=color, label=opt, linewidth=2, markersize=6)
    ax3.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Population Size")
    ax3.set_ylabel("Speedup (×)")
    ax3.set_title("Speedup vs Population Size", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    if len(scaling_data) > 0:
        ax3.set_xscale("log")

    # ── Bottom-right: Memory ──
    opt_mem = defaultdict(lambda: {"numpy": [], "torch": []})
    for p in pairs:
        opt = p["key"][0]
        opt_mem[opt]["numpy"].append(p["numpy"]["peak_memory_mb"])
        opt_mem[opt]["torch"].append(p["torch_cpu"]["peak_memory_mb"])

    opts_m = sorted(opt_mem.keys())
    x = np.arange(len(opts_m))
    w = 0.35
    ax4.bar(x - w / 2, [np.mean(opt_mem[o]["numpy"]) for o in opts_m], w,
            label="NumPy", color="#FF7043", alpha=0.85)
    ax4.bar(x + w / 2, [np.mean(opt_mem[o]["torch"]) for o in opts_m], w,
            label="PyTorch", color="#42A5F5", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(opts_m, fontsize=10)
    ax4.set_ylabel("Peak Memory (MB)")
    ax4.set_title("Memory Usage", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    fig.suptitle("Otorchmizer vs Opytimizer — Benchmark Dashboard", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "08_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 08_dashboard.png")


# ============================================================================
# Plot 9: GPU vs CPU Speedup Comparison (bar chart)
# ============================================================================

def plot_gpu_vs_cpu_bars(pairs, outdir):
    """Grouped bar chart: avg CPU speedup vs GPU speedup per optimizer."""

    has_gpu = any(p["torch_gpu"] is not None for p in pairs)
    if not has_gpu:
        print("  → 09_gpu_vs_cpu_bars.png (SKIPPED — no GPU data)")
        return

    opt_cpu_speedup = defaultdict(list)
    opt_gpu_speedup = defaultdict(list)
    for p in pairs:
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        opt_cpu_speedup[p["key"][0]].append(np_t / tc_t if tc_t > 0 else 1.0)
        if p["torch_gpu"] is not None:
            tg_t = p["torch_gpu"]["time_seconds"]
            opt_gpu_speedup[p["key"][0]].append(np_t / tg_t if tg_t > 0 else 1.0)

    opts = sorted(opt_cpu_speedup.keys())
    cpu_means = [np.mean(opt_cpu_speedup[o]) for o in opts]
    gpu_means = [np.mean(opt_gpu_speedup.get(o, [0])) for o in opts]
    cpu_peaks = [np.max(opt_cpu_speedup[o]) for o in opts]
    gpu_peaks = [np.max(opt_gpu_speedup.get(o, [0])) for o in opts]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(opts))
    w = 0.2

    ax.bar(x - 1.5*w, cpu_means, w, label="CPU Avg", color="#42A5F5", edgecolor="#1565C0", alpha=0.85)
    ax.bar(x - 0.5*w, cpu_peaks, w, label="CPU Peak", color="#1565C0", edgecolor="#0D47A1", alpha=0.85)
    ax.bar(x + 0.5*w, gpu_means, w, label="GPU Avg", color="#66BB6A", edgecolor="#2E7D32", alpha=0.85)
    ax.bar(x + 1.5*w, gpu_peaks, w, label="GPU Peak", color="#2E7D32", edgecolor="#1B5E20", alpha=0.85)

    for i in range(len(opts)):
        ax.text(x[i]-1.5*w, cpu_means[i]+5, f"{cpu_means[i]:.0f}×", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.text(x[i]-0.5*w, cpu_peaks[i]+5, f"{cpu_peaks[i]:.0f}×", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.text(x[i]+0.5*w, gpu_means[i]+5, f"{gpu_means[i]:.0f}×", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.text(x[i]+1.5*w, gpu_peaks[i]+5, f"{gpu_peaks[i]:.0f}×", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(opts, fontsize=13)
    ax.set_ylabel("Speedup over NumPy (×)", fontsize=13)
    ax.set_title("CPU vs GPU Speedup by Algorithm", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "09_gpu_vs_cpu_bars.png"), dpi=150)
    plt.close(fig)
    print("  → 09_gpu_vs_cpu_bars.png")


# ============================================================================
# Plot 10: GPU Scaling — speedup vs population size
# ============================================================================

def plot_gpu_scaling(pairs, outdir):
    """Line plot: GPU speedup vs population size, one line per optimizer."""

    has_gpu = any(p["torch_gpu"] is not None for p in pairs)
    if not has_gpu:
        print("  → 10_gpu_scaling.png (SKIPPED — no GPU data)")
        return

    data_cpu = defaultdict(lambda: defaultdict(list))
    data_gpu = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        opt = p["key"][0]
        n = p["key"][2]
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        data_cpu[opt][n].append(np_t / tc_t if tc_t > 0 else 1.0)
        if p["torch_gpu"] is not None:
            tg_t = p["torch_gpu"]["time_seconds"]
            data_gpu[opt][n].append(np_t / tg_t if tg_t > 0 else 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = {"PSO": "#2196F3", "WOA": "#4CAF50", "FA": "#FF9800", "GA": "#9C27B0", "HC": "#F44336"}

    for opt in sorted(data_cpu.keys()):
        ns = sorted(data_cpu[opt].keys())
        means = [np.mean(data_cpu[opt][n]) for n in ns]
        ax1.plot(ns, means, "o-", color=colors.get(opt, "gray"), label=opt, linewidth=2, markersize=8)

    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Population Size", fontsize=13)
    ax1.set_ylabel("Speedup over NumPy (×)", fontsize=13)
    ax1.set_title("PyTorch CPU Speedup", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log")

    for opt in sorted(data_gpu.keys()):
        ns = sorted(data_gpu[opt].keys())
        means = [np.mean(data_gpu[opt][n]) for n in ns]
        ax2.plot(ns, means, "s-", color=colors.get(opt, "gray"), label=opt, linewidth=2, markersize=8)

    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Population Size", fontsize=13)
    ax2.set_title("PyTorch GPU (RTX 4070) Speedup", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log")

    fig.suptitle("Speedup Scaling: CPU vs GPU", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "10_gpu_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 10_gpu_scaling.png")


# ============================================================================
# Plot 11: 3-Backend Heatmap (optimizer × pop → speedup, CPU & GPU side by side)
# ============================================================================

def plot_3backend_heatmap(pairs, outdir):
    """Side-by-side heatmaps for CPU and GPU speedup."""

    has_gpu = any(p["torch_gpu"] is not None for p in pairs)
    if not has_gpu:
        print("  → 11_3backend_heatmap.png (SKIPPED — no GPU data)")
        return

    cpu_data = defaultdict(lambda: defaultdict(list))
    gpu_data = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        opt = p["key"][0]
        n = p["key"][2]
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        cpu_data[opt][n].append(np_t / tc_t if tc_t > 0 else 1.0)
        if p["torch_gpu"] is not None:
            tg_t = p["torch_gpu"]["time_seconds"]
            gpu_data[opt][n].append(np_t / tg_t if tg_t > 0 else 1.0)

    opts = sorted(cpu_data.keys())
    all_ns = sorted(set(n for o in cpu_data.values() for n in o.keys()))

    cpu_matrix = np.zeros((len(opts), len(all_ns)))
    gpu_matrix = np.zeros((len(opts), len(all_ns)))
    for i, opt in enumerate(opts):
        for j, n in enumerate(all_ns):
            cpu_matrix[i, j] = np.mean(cpu_data[opt].get(n, [1.0]))
            gpu_matrix[i, j] = np.mean(gpu_data[opt].get(n, [1.0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    vmax = max(cpu_matrix.max(), gpu_matrix.max())

    im1 = ax1.imshow(cpu_matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
    ax1.set_xticks(range(len(all_ns)))
    ax1.set_xticklabels([str(n) for n in all_ns], fontsize=11)
    ax1.set_yticks(range(len(opts)))
    ax1.set_yticklabels(opts, fontsize=12)
    ax1.set_xlabel("Population Size", fontsize=13)
    ax1.set_title("CPU Speedup", fontsize=14, fontweight="bold")
    for i in range(len(opts)):
        for j in range(len(all_ns)):
            ax1.text(j, i, f"{cpu_matrix[i,j]:.0f}×", ha="center", va="center",
                     fontsize=10, fontweight="bold",
                     color="white" if cpu_matrix[i,j] > vmax*0.5 else "black")

    im2 = ax2.imshow(gpu_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax2.set_xticks(range(len(all_ns)))
    ax2.set_xticklabels([str(n) for n in all_ns], fontsize=11)
    ax2.set_yticks(range(len(opts)))
    ax2.set_yticklabels(opts, fontsize=12)
    ax2.set_xlabel("Population Size", fontsize=13)
    ax2.set_title("GPU Speedup (RTX 4070)", fontsize=14, fontweight="bold")
    for i in range(len(opts)):
        for j in range(len(all_ns)):
            ax2.text(j, i, f"{gpu_matrix[i,j]:.0f}×", ha="center", va="center",
                     fontsize=10, fontweight="bold",
                     color="white" if gpu_matrix[i,j] > vmax*0.5 else "black")

    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Speedup (×)")
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Speedup (×)")

    fig.suptitle("CPU vs GPU Speedup Heatmap", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "11_3backend_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 11_3backend_heatmap.png")


# ============================================================================
# Plot 12: GPU Time Invariance — showing GPU time stays flat vs population
# ============================================================================

def plot_gpu_time_invariance(pairs, outdir):
    """Show that GPU execution time stays nearly constant regardless of problem size."""

    has_gpu = any(p["torch_gpu"] is not None for p in pairs)
    if not has_gpu:
        print("  → 12_gpu_time_invariance.png (SKIPPED — no GPU data)")
        return

    np_times = defaultdict(lambda: defaultdict(list))
    cpu_times = defaultdict(lambda: defaultdict(list))
    gpu_times = defaultdict(lambda: defaultdict(list))

    for p in pairs:
        opt = p["key"][0]
        n = p["key"][2]
        np_times[opt][n].append(p["numpy"]["time_seconds"])
        cpu_times[opt][n].append(p["torch_cpu"]["time_seconds"])
        if p["torch_gpu"] is not None:
            gpu_times[opt][n].append(p["torch_gpu"]["time_seconds"])

    # Focus on PSO and WOA which scale best
    focus_opts = [o for o in ["PSO", "WOA", "HC", "GA"] if o in gpu_times]
    if not focus_opts:
        focus_opts = list(gpu_times.keys())[:4]

    fig, axes = plt.subplots(1, len(focus_opts), figsize=(5*len(focus_opts), 6), sharey=False)
    if len(focus_opts) == 1:
        axes = [axes]

    for ax, opt in zip(axes, focus_opts):
        ns = sorted(np_times[opt].keys())
        np_m = [np.mean(np_times[opt][n]) for n in ns]
        cpu_m = [np.mean(cpu_times[opt][n]) for n in ns]
        gpu_m = [np.mean(gpu_times[opt].get(n, [0])) for n in ns]

        ax.plot(ns, np_m, "o-", color="#F44336", label="NumPy", linewidth=2, markersize=6)
        ax.plot(ns, cpu_m, "s-", color="#2196F3", label="PyTorch CPU", linewidth=2, markersize=6)
        ax.plot(ns, gpu_m, "D-", color="#4CAF50", label="PyTorch GPU", linewidth=2, markersize=6)

        ax.set_xlabel("Population Size", fontsize=12)
        ax.set_ylabel("Execution Time (s)", fontsize=12)
        ax.set_title(opt, fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.suptitle("Execution Time vs Population Size (all backends)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "12_gpu_time_invariance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 12_gpu_time_invariance.png")


# ============================================================================
# Plot 13: Full dashboard with GPU data
# ============================================================================

def plot_gpu_dashboard(pairs, outdir):
    """2x2 dashboard with GPU-focused metrics."""

    has_gpu = any(p["torch_gpu"] is not None for p in pairs)
    if not has_gpu:
        print("  → 13_gpu_dashboard.png (SKIPPED — no GPU data)")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # ── Top-left: Peak speedup bars (CPU vs GPU) ──
    opt_cpu_peak = defaultdict(float)
    opt_gpu_peak = defaultdict(float)
    for p in pairs:
        np_t = p["numpy"]["time_seconds"]
        tc_t = p["torch_cpu"]["time_seconds"]
        cpu_s = np_t / tc_t if tc_t > 0 else 1.0
        opt = p["key"][0]
        opt_cpu_peak[opt] = max(opt_cpu_peak[opt], cpu_s)
        if p["torch_gpu"] is not None:
            tg_t = p["torch_gpu"]["time_seconds"]
            gpu_s = np_t / tg_t if tg_t > 0 else 1.0
            opt_gpu_peak[opt] = max(opt_gpu_peak[opt], gpu_s)

    opts = sorted(opt_cpu_peak.keys())
    x = np.arange(len(opts))
    w = 0.35
    ax1.bar(x - w/2, [opt_cpu_peak[o] for o in opts], w, label="CPU Peak", color="#42A5F5", edgecolor="#1565C0")
    ax1.bar(x + w/2, [opt_gpu_peak.get(o, 0) for o in opts], w, label="GPU Peak", color="#66BB6A", edgecolor="#2E7D32")
    for i, o in enumerate(opts):
        ax1.text(x[i]-w/2, opt_cpu_peak[o]+10, f"{opt_cpu_peak[o]:.0f}×", ha="center", va="bottom", fontsize=9)
        if o in opt_gpu_peak:
            ax1.text(x[i]+w/2, opt_gpu_peak[o]+10, f"{opt_gpu_peak[o]:.0f}×", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(opts, fontsize=12)
    ax1.set_ylabel("Peak Speedup (×)")
    ax1.set_title("Peak Speedup: CPU vs GPU", fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # ── Top-right: GPU speedup vs dimensions ──
    dim_data = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        if p["torch_gpu"] is not None:
            opt = p["key"][0]
            d = p["key"][3]
            np_t = p["numpy"]["time_seconds"]
            tg_t = p["torch_gpu"]["time_seconds"]
            dim_data[opt][d].append(np_t / tg_t if tg_t > 0 else 1.0)

    colors = {"PSO": "#2196F3", "WOA": "#4CAF50", "FA": "#FF9800", "GA": "#9C27B0", "HC": "#F44336"}
    for opt in sorted(dim_data.keys()):
        ds = sorted(dim_data[opt].keys())
        means = [np.mean(dim_data[opt][d]) for d in ds]
        ax2.plot(ds, means, "s-", color=colors.get(opt, "gray"), label=opt, linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Dimensions")
    ax2.set_ylabel("GPU Speedup (×)")
    ax2.set_title("GPU Speedup vs Dimensionality", fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log")

    # ── Bottom-left: GPU scaling with population ──
    pop_gpu = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        if p["torch_gpu"] is not None:
            opt = p["key"][0]
            n = p["key"][2]
            np_t = p["numpy"]["time_seconds"]
            tg_t = p["torch_gpu"]["time_seconds"]
            pop_gpu[opt][n].append(np_t / tg_t if tg_t > 0 else 1.0)

    for opt in sorted(pop_gpu.keys()):
        ns = sorted(pop_gpu[opt].keys())
        means = [np.mean(pop_gpu[opt][n]) for n in ns]
        ax3.plot(ns, means, "o-", color=colors.get(opt, "gray"), label=opt, linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Population Size")
    ax3.set_ylabel("GPU Speedup (×)")
    ax3.set_title("GPU Speedup vs Population", fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_xscale("log")

    # ── Bottom-right: Absolute time comparison at n=1000, d=100 ──
    big_configs = [p for p in pairs if p["key"][2] >= 500 and p["key"][3] >= 50
                   and p["torch_gpu"] is not None]
    if big_configs:
        opt_np = defaultdict(list)
        opt_cpu = defaultdict(list)
        opt_gpu = defaultdict(list)
        for p in big_configs:
            opt = p["key"][0]
            opt_np[opt].append(p["numpy"]["time_seconds"])
            opt_cpu[opt].append(p["torch_cpu"]["time_seconds"])
            opt_gpu[opt].append(p["torch_gpu"]["time_seconds"])

        opts_big = sorted(opt_np.keys())
        x = np.arange(len(opts_big))
        w = 0.25
        ax4.bar(x - w, [np.mean(opt_np[o]) for o in opts_big], w, label="NumPy", color="#F44336", alpha=0.85)
        ax4.bar(x, [np.mean(opt_cpu[o]) for o in opts_big], w, label="PyTorch CPU", color="#42A5F5", alpha=0.85)
        ax4.bar(x + w, [np.mean(opt_gpu[o]) for o in opts_big], w, label="PyTorch GPU", color="#66BB6A", alpha=0.85)
        ax4.set_xticks(x)
        ax4.set_xticklabels(opts_big, fontsize=12)
        ax4.set_ylabel("Avg Time (s)")
        ax4.set_title("Absolute Time (n≥500, d≥50)", fontweight="bold")
        ax4.legend(fontsize=10)
        ax4.grid(axis="y", alpha=0.3)
        ax4.set_yscale("log")

    fig.suptitle("GPU Acceleration Dashboard — RTX 4070", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "13_gpu_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 13_gpu_dashboard.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--input", type=str, default="benchmarks/results.json")
    parser.add_argument("--outdir", type=str, default="benchmarks/plots")
    args = parser.parse_args()

    results = load_results(args.input)
    pairs = pair_results(results)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nGenerating plots from {len(pairs)} paired benchmark configs...\n")

    plot_speedup_bars(pairs, args.outdir)
    plot_scaling(pairs, args.outdir)
    plot_dimension_scaling(pairs, args.outdir)
    plot_quality_scatter(pairs, args.outdir)
    plot_memory(pairs, args.outdir)
    plot_timing_heatmap(pairs, args.outdir)
    plot_time_comparison(pairs, args.outdir)
    plot_dashboard(pairs, args.outdir)
    plot_gpu_vs_cpu_bars(pairs, args.outdir)
    plot_gpu_scaling(pairs, args.outdir)
    plot_3backend_heatmap(pairs, args.outdir)
    plot_gpu_time_invariance(pairs, args.outdir)
    plot_gpu_dashboard(pairs, args.outdir)

    print(f"\nAll plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
