"""Benchmark suite: Opytimizer (NumPy) vs Otorchmizer (PyTorch).

Measures wall-clock time, convergence quality, and memory usage
across varying population sizes and dimensions for all shared optimizers.

Usage:
    python benchmarks/run_benchmarks.py          # run all benchmarks
    python benchmarks/run_benchmarks.py --quick   # quick smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

# ── Otorchmizer (PyTorch) ────────────────────────────────────────────────────
import torch

from otorchmizer.spaces.search import SearchSpace as TorchSearch
from otorchmizer.optimizers.swarm.pso import PSO as TorchPSO
from otorchmizer.optimizers.swarm.woa import WOA as TorchWOA
from otorchmizer.optimizers.swarm.fa import FA as TorchFA
from otorchmizer.optimizers.evolutionary.ga import GA as TorchGA
from otorchmizer.optimizers.misc.hc import HC as TorchHC
from otorchmizer.core.function import Function as TorchFunction
from otorchmizer.otorchmizer import Otorchmizer

# ── Opytimizer (NumPy) ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "opytimizer"))

from opytimizer.spaces.search import SearchSpace as NumpySearch
from opytimizer.optimizers.swarm.pso import PSO as NumpyPSO
from opytimizer.optimizers.swarm.woa import WOA as NumpyWOA
from opytimizer.optimizers.swarm.fa import FA as NumpyFA
from opytimizer.optimizers.evolutionary.ga import GA as NumpyGA
from opytimizer.optimizers.misc.hc import HC as NumpyHC
from opytimizer.core.function import Function as NumpyFunction
from opytimizer.opytimizer import Opytimizer

# Suppress logging noise and tqdm during benchmarks
import logging
logging.disable(logging.CRITICAL)

import tqdm
tqdm.tqdm.__init__orig = tqdm.tqdm.__init__
_orig_init = tqdm.tqdm.__init__
def _silent_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _orig_init(self, *args, **kwargs)
tqdm.tqdm.__init__ = _silent_init


# ============================================================================
# Benchmark functions (well-known test suite)
# ============================================================================

def sphere_np(x):
    return float(np.sum(x ** 2))

def rastrigin_np(x):
    A = 10
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

def ackley_np(x):
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    s1 = np.sum(x ** 2) / n
    s2 = np.sum(np.cos(c * x)) / n
    return float(-a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e)

def rosenbrock_np(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def sphere_torch(x):
    return (x ** 2).sum()

def rastrigin_torch(x):
    A = 10
    return A * x.shape[0] + (x ** 2 - A * torch.cos(2 * torch.pi * x)).sum()

def ackley_torch(x):
    n = x.shape[0]
    a, b, c = 20.0, 0.2, 2 * torch.pi
    s1 = (x ** 2).sum() / n
    s2 = (torch.cos(c * x)).sum() / n
    return -a * torch.exp(-b * torch.sqrt(s1)) - torch.exp(s2) + a + torch.e

def rosenbrock_torch(x):
    x_flat = x.squeeze()
    return (100 * (x_flat[1:] - x_flat[:-1] ** 2) ** 2 + (1 - x_flat[:-1]) ** 2).sum()


BENCHMARK_FUNCTIONS = {
    "sphere": (sphere_np, sphere_torch),
    "rastrigin": (rastrigin_np, rastrigin_torch),
    "ackley": (ackley_np, ackley_torch),
    "rosenbrock": (rosenbrock_np, rosenbrock_torch),
}

# ============================================================================
# Optimizer registry
# ============================================================================

OPTIMIZERS = {
    "PSO": (NumpyPSO, TorchPSO),
    "WOA": (NumpyWOA, TorchWOA),
    "FA":  (NumpyFA,  TorchFA),
    "GA":  (NumpyGA,  TorchGA),
    "HC":  (NumpyHC,  TorchHC),
}

# Bounds per benchmark function
BOUNDS = {
    "sphere":     (-5.12, 5.12),
    "rastrigin":  (-5.12, 5.12),
    "ackley":     (-5.0, 5.0),
    "rosenbrock": (-5.0, 10.0),
}


# ============================================================================
# Result container
# ============================================================================

@dataclass
class BenchResult:
    optimizer: str
    function: str
    n_agents: int
    n_variables: int
    n_iterations: int
    backend: str  # "numpy" or "torch"
    time_seconds: float = 0.0
    best_fitness: float = 0.0
    peak_memory_mb: float = 0.0
    converged: bool = False


# ============================================================================
# Runner helpers
# ============================================================================

def run_numpy(opt_cls, fn_np, n_agents, n_variables, n_iterations, bounds):
    """Run one opytimizer benchmark and return (time, best_fitness, peak_mem_mb)."""

    lb, ub = bounds
    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()

    space = NumpySearch(
        n_agents=n_agents,
        n_variables=n_variables,
        lower_bound=[lb] * n_variables,
        upper_bound=[ub] * n_variables,
    )
    optimizer = opt_cls()
    fn = NumpyFunction(fn_np)
    opt = Opytimizer(space=space, optimizer=optimizer, function=fn)
    opt.start(n_iterations=n_iterations)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    best = float(opt.space.best_agent.fit)
    return elapsed, best, peak / (1024 * 1024)


def run_torch(opt_cls, fn_torch, n_agents, n_variables, n_iterations, bounds, device="cpu"):
    """Run one otorchmizer benchmark and return (time, best_fitness, peak_mem_mb)."""

    lb, ub = bounds
    gc.collect()

    if device == "cpu":
        tracemalloc.start()
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    space = TorchSearch(
        n_agents=n_agents,
        n_variables=n_variables,
        lower_bound=lb,
        upper_bound=ub,
        device=device,
    )
    optimizer = opt_cls()
    fn = TorchFunction(fn_torch)
    ot = Otorchmizer(space=space, optimizer=optimizer, function=fn)
    ot.start(n_iterations=n_iterations)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0

    if device == "cpu":
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = peak / (1024 * 1024)
    elif torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        mem_mb = 0.0

    best = float(ot.space.population.best_fitness.item())
    return elapsed, best, mem_mb


# ============================================================================
# Main benchmark
# ============================================================================

def run_benchmarks(
    pop_sizes: List[int],
    dims: List[int],
    n_iterations: int,
    functions: List[str],
    optimizers: List[str],
    n_repeats: int = 3,
    use_gpu: bool = False,
) -> List[BenchResult]:
    """Run full benchmark matrix and return results."""

    results: List[BenchResult] = []
    total = len(optimizers) * len(functions) * len(pop_sizes) * len(dims)
    count = 0

    for opt_name in optimizers:
        np_cls, torch_cls = OPTIMIZERS[opt_name]

        # FA has O(n²) per iteration in numpy — limit its population sizes
        if opt_name == "FA":
            effective_pops = [n for n in pop_sizes if n <= 100]
        else:
            effective_pops = pop_sizes

        for fn_name in functions:
            fn_np, fn_torch = BENCHMARK_FUNCTIONS[fn_name]
            bounds = BOUNDS[fn_name]

            for n_agents in effective_pops:
                for n_vars in dims:
                    count += 1
                    label = f"[{count}/{total}] {opt_name}/{fn_name} n={n_agents} d={n_vars}"
                    print(f"  {label} ... ", end="", flush=True)

                    # ── NumPy runs ──
                    np_times, np_fits, np_mems = [], [], []
                    for _ in range(n_repeats):
                        t, f, m = run_numpy(np_cls, fn_np, n_agents, n_vars, n_iterations, bounds)
                        np_times.append(t)
                        np_fits.append(f)
                        np_mems.append(m)

                    results.append(BenchResult(
                        optimizer=opt_name, function=fn_name,
                        n_agents=n_agents, n_variables=n_vars,
                        n_iterations=n_iterations, backend="numpy",
                        time_seconds=float(np.median(np_times)),
                        best_fitness=float(np.median(np_fits)),
                        peak_memory_mb=float(np.median(np_mems)),
                    ))

                    # ── PyTorch CPU runs ──
                    torch_times, torch_fits, torch_mems = [], [], []
                    for _ in range(n_repeats):
                        t, f, m = run_torch(torch_cls, fn_torch, n_agents, n_vars, n_iterations, bounds, "cpu")
                        torch_times.append(t)
                        torch_fits.append(f)
                        torch_mems.append(m)

                    results.append(BenchResult(
                        optimizer=opt_name, function=fn_name,
                        n_agents=n_agents, n_variables=n_vars,
                        n_iterations=n_iterations, backend="torch_cpu",
                        time_seconds=float(np.median(torch_times)),
                        best_fitness=float(np.median(torch_fits)),
                        peak_memory_mb=float(np.median(torch_mems)),
                    ))

                    # ── PyTorch GPU runs (optional) ──
                    if use_gpu and torch.cuda.is_available():
                        gpu_times, gpu_fits, gpu_mems = [], [], []
                        for _ in range(n_repeats):
                            t, f, m = run_torch(torch_cls, fn_torch, n_agents, n_vars, n_iterations, bounds, "cuda")
                            gpu_times.append(t)
                            gpu_fits.append(f)
                            gpu_mems.append(m)

                        results.append(BenchResult(
                            optimizer=opt_name, function=fn_name,
                            n_agents=n_agents, n_variables=n_vars,
                            n_iterations=n_iterations, backend="torch_gpu",
                            time_seconds=float(np.median(gpu_times)),
                            best_fitness=float(np.median(gpu_fits)),
                            peak_memory_mb=float(np.median(gpu_mems)),
                        ))

                    # Print speedup summary
                    np_t = float(np.median(np_times))
                    tc_t = float(np.median(torch_times))
                    speedup = np_t / tc_t if tc_t > 0 else 0
                    msg = f"numpy={np_t:.3f}s torch_cpu={tc_t:.3f}s cpu_speedup={speedup:.2f}x"
                    if use_gpu and torch.cuda.is_available():
                        tg_t = float(np.median(gpu_times))
                        gpu_speedup = np_t / tg_t if tg_t > 0 else 0
                        msg += f" torch_gpu={tg_t:.3f}s gpu_speedup={gpu_speedup:.2f}x"
                    print(msg)

    return results


def save_results(results: List[BenchResult], path: str):
    """Save results to JSON."""
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark opytimizer vs otorchmizer")
    parser.add_argument("--quick", action="store_true", help="Quick mode (small configs)")
    parser.add_argument("--extended", action="store_true", help="Extended mode (larger pop & dims)")
    parser.add_argument("--gpu", action="store_true", help="Include GPU benchmarks")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per config")
    parser.add_argument("--output", type=str, default="benchmarks/results.json", help="Output JSON path")
    args = parser.parse_args()

    if args.quick:
        pop_sizes = [10, 50]
        dims = [5, 10]
        n_iterations = 20
        functions = ["sphere", "rastrigin"]
        optimizers = ["PSO", "WOA", "HC"]
    elif args.extended:
        pop_sizes = [10, 50, 100, 250, 500, 1000]
        dims = [5, 10, 50, 100]
        n_iterations = 50
        functions = ["sphere", "rastrigin", "ackley", "rosenbrock"]
        optimizers = ["PSO", "WOA", "FA", "GA", "HC"]
    else:
        pop_sizes = [10, 50, 100, 250]
        dims = [5, 10, 50]
        n_iterations = 30
        functions = ["sphere", "rastrigin", "ackley", "rosenbrock"]
        optimizers = ["PSO", "WOA", "FA", "GA", "HC"]

    gpu_active = args.gpu and torch.cuda.is_available()
    print("=" * 70)
    print("BENCHMARK: Opytimizer (NumPy) vs Otorchmizer (PyTorch)")
    print("=" * 70)
    print(f"  Population sizes : {pop_sizes}")
    print(f"  Dimensions       : {dims}")
    print(f"  Iterations       : {n_iterations}")
    print(f"  Functions        : {functions}")
    print(f"  Optimizers       : {optimizers}")
    print(f"  Repeats          : {args.repeats}")
    print(f"  GPU              : {gpu_active}", end="")
    if gpu_active:
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print("=" * 70)

    results = run_benchmarks(
        pop_sizes=pop_sizes,
        dims=dims,
        n_iterations=n_iterations,
        functions=functions,
        optimizers=optimizers,
        n_repeats=args.repeats,
        use_gpu=args.gpu,
    )

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(results, args.output)
