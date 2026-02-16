# Otorchmizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Otorchmizer** is a PyTorch-based library of nature-inspired meta-heuristic optimization algorithms. It is a GPU-accelerated reimplementation of [Opytimizer](https://github.com/gugarosa/opytimizer), replacing NumPy with PyTorch tensors for **orders-of-magnitude speedups** on both CPU and GPU.

---

## Highlights

| Metric | Value |
|--------|-------|
| Algorithms | 10 (PSO, AIWPSO, RPSO, SAVPSO, VPSO, WOA, FA, GA, GS, HC) |
| Peak CPU speedup | **1,055×** over NumPy |
| Peak GPU speedup | **2,311×** on RTX 4070 |
| Convergence quality | Parity with original |
| Tests | 77 passing |

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/gugarosa/otorchmizer.git
cd otorchmizer
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- NumPy, matplotlib, tqdm, dill

GPU acceleration requires a CUDA-capable PyTorch installation (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).

## Quick Start

```python
from otorchmizer.otorchmizer import Otorchmizer
from otorchmizer.spaces.search import SearchSpace
from otorchmizer.optimizers.swarm.pso import PSO

# Define objective function
def sphere(x):
    return (x ** 2).sum()

# Configure and run
space = SearchSpace(n_agents=50, n_variables=10, lower_bound=-10.0, upper_bound=10.0)
optimizer = PSO()
o = Otorchmizer(space=space, optimizer=optimizer, function=sphere)
history = o.start(n_iterations=100)

print(f"Best fitness: {history.best_fitness[-1]:.6f}")
```

### GPU Usage

```python
# Automatically uses GPU if available
space = SearchSpace(
    n_agents=1000, n_variables=100,
    lower_bound=-10.0, upper_bound=10.0,
    device="auto"  # "cpu", "cuda", or "auto"
)
```

## Architecture

The core design replaces `List[Agent]` objects with a single **Population** tensor:

```
# NumPy (opytimizer)              # PyTorch (otorchmizer)
for agent in space.agents:         positions: Tensor (n_agents, n_vars, n_dims)
    agent.position  # (n_vars,)    fitness:   Tensor (n_agents,)
    agent.fitness   # float        # All agents updated in one tensor op
```

This enables full vectorization — all agents are updated simultaneously via batched tensor operations, eliminating Python-level loops entirely.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design document.

## Algorithms

| Family | Algorithms | GPU Speedup Range |
|--------|-----------|-------------------|
| **Swarm** | PSO, AIWPSO, RPSO, SAVPSO, VPSO | 2–995× |
| **Swarm** | WOA (Whale Optimization) | 2–1,087× |
| **Swarm** | FA (Firefly Algorithm) | 1–6× |
| **Evolutionary** | GA (Genetic Algorithm) | 1–1,651× |
| **Misc** | HC (Hill Climbing), GS (Grid Search) | 3–2,311× |

FA shows lower speedup due to its inherently sequential cascade (each agent updated by all brighter agents in order).

## Benchmarks

Run the benchmarking suite:

```bash
# Quick CPU-only benchmarks
python benchmarks/run_benchmarks.py --quick

# Extended benchmarks with GPU
python benchmarks/run_benchmarks.py --extended --gpu

# Generate plots
python benchmarks/plot_results.py --input benchmarks/results_extended.json --outdir benchmarks/plots_extended
```

### Sample Results (RTX 4070)

| Optimizer | Config | NumPy | PyTorch GPU | Speedup |
|-----------|--------|-------|-------------|---------|
| HC | 1000 agents, 100d | 66.2s | 0.029s | **2,311×** |
| GA | 1000 agents, 100d | 134.1s | 0.081s | **1,651×** |
| WOA | 1000 agents, 100d | 71.4s | 0.070s | **1,087×** |
| PSO | 1000 agents, 100d | 67.5s | 0.068s | **995×** |

GPU execution time stays **nearly constant** (~0.03–0.08s) regardless of problem size.

See the full [Migration Report](report/REPORT.md) for detailed analysis and all 13 benchmark plots.

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
otorchmizer/
├── otorchmizer/           # Main package
│   ├── core/              # Population, Space, Optimizer, Function, Device
│   ├── math/              # random, distribution, general, hyper
│   ├── optimizers/        # swarm/, evolutionary/, misc/
│   ├── spaces/            # search, boolean, grid, tree, ...
│   ├── utils/             # constant, exception, logging, history, callback
│   ├── visualization/     # convergence, surface
│   └── otorchmizer.py     # Orchestrator pipeline
├── tests/                 # 77 tests (unit + integration + regression)
├── benchmarks/            # Benchmark harness, results, plots
├── report/                # Migration report with embedded figures
└── ARCHITECTURE.md        # Design document
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
