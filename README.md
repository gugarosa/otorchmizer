# Otorchmizer: A PyTorch-Powered Nature-Inspired Optimizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## Welcome to Otorchmizer.

Did you ever reach a bottleneck in your computational experiments? Are you tired of waiting hours for meta-heuristic optimization runs? If yes, Otorchmizer is the real deal! This package provides an easy-to-go implementation of **91 meta-heuristic optimization algorithms** — all powered by PyTorch tensors for GPU-accelerated performance. From populations to search spaces, from internal functions to external communication, we will foster all research related to optimizing stuff.

Otorchmizer is the modernized successor to [Opytimizer](https://github.com/gugarosa/opytimizer), delivering **up to 2,311× speedup** by replacing NumPy with PyTorch.

Use Otorchmizer if you need a library or wish to:
* Create your optimization algorithm with automatic GPU support;
* Design or use pre-loaded optimization tasks at scale;
* Run the same code on CPU, single-GPU, or multi-GPU seamlessly;
* Leverage `torch.compile` and CUDA Graphs for maximum throughput;
* Mix-and-match different strategies to solve your problem;
* Because it is fun to optimize things — even faster.

Otorchmizer is compatible with: **Python 3.10+** and **PyTorch 2.0+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.
5. Finally, we focus on **minimization**. Take that in mind when designing your problem.

---

## Citation

If you use Otorchmizer to fulfill any of your needs, please cite the original Opytimizer paper:

```BibTex
@misc{rosa2019opytimizer,
    title={Opytimizer: A Nature-Inspired Python Optimizer},
    author={Gustavo H. de Rosa, Douglas Rodrigues and João P. Papa},
    year={2019},
    eprint={1912.13002},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

---

## Getting started: 60 seconds with Otorchmizer

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think of, including GPU acceleration, `torch.compile`, and multi-GPU population splitting.

Alternatively, if you wish to learn even more, please take a minute:

Otorchmizer is based on the following structure, and you should pay attention to its tree:

```yaml
- otorchmizer
    - core
        - agent_view
        - block
        - device
        - function
        - node
        - optimizer
        - population
        - space
    - functions
        - constrained
        - multi_objective
    - math
        - distribution
        - general
        - hyper
        - random
    - optimizers
        - boolean
        - evolutionary
        - misc
        - population
        - science
        - social
        - swarm
    - spaces
        - boolean
        - graph
        - grid
        - hyper_complex
        - pareto
        - search
        - tree
    - utils
        - callback
        - constant
        - exception
        - history
        - logging
    - visualization
        - convergence
        - surface
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules. The key innovation is the **Population** class, which stores all agent data as contiguous PyTorch tensors `(n_agents, n_variables, n_dimensions)`, enabling vectorized operations and GPU parallelism. Also featured here is the **DeviceManager**, which handles CPU/GPU/multi-GPU resolution, mixed-precision, and CUDA Graph capture.

### Functions

Instead of using raw and straightforward functions, why not try this module? Compose high-level abstract functions or even new function-based ideas in order to solve your problems. Functions are auto-vectorized across the entire population via `torch.vmap` — you write a single-agent function, and we handle the batching.

### Math

Just because we are computing stuff does not mean that we do not need math. Math is the mathematical package containing low-level math implementations. From random numbers to distribution generation, you can find your needs on this module — all backed by PyTorch tensors for device-agnostic computation.

### Optimizers

This is why we are called Otorchmizer. This is the heart of heuristics, where you can find **91 meta-heuristic optimization techniques** across 7 families — swarm intelligence, evolutionary algorithms, science-inspired methods, and more. Every algorithm uses batched tensor operations, meaning the same code runs on CPU and GPU without modification.

### Spaces

One can see the space as the place that agents will update their positions and evaluate a fitness function. However, the newest approaches may consider a different type of space. Thinking about that, we are glad to support diverse space implementations.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing repeatedly.

### Visualization

Everyone needs images and plots to help visualize what is happening, correct? This package will provide every visual-related method for you. Check a specific variable convergence, your fitness function convergence, plot benchmark function surfaces, and much more!

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, Otorchmizer will be the one-to-go package that you will need, from the first installation to the daily tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```bash
pip install -e .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands are needed.

### Windows

No specific additional commands are needed.

### MacOS

No specific additional commands are needed.

### GPU Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## How-To-Use: Minimal Example

Take a look at a quick working example of Otorchmizer. Note that we are not passing many extra arguments nor additional information to the procedure. For more complex examples, please check our `examples/` folder.

```python
import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.swarm import PSO

def sphere(x):
    return (x ** 2).sum(dim=(-1, -2))

n_agents = 20
n_variables = 2
lower_bound = [-10, -10]
upper_bound = [10, 10]

space = Space(n_agents=n_agents, n_variables=n_variables,
              lower_bound=lower_bound, upper_bound=upper_bound)
space.build()

optimizer = PSO()
function = Function(sphere)

opt = Otorchmizer(space, optimizer, function)
opt.start(n_iterations=1000)
```

---

## GPU Usage

Running on GPU requires only a single parameter change — all algorithms, spaces, and functions work identically:

```python
# Automatically uses GPU if available, otherwise falls back to CPU
space = Space(n_agents=1000, n_variables=100,
              lower_bound=-10.0, upper_bound=10.0,
              device="auto")
space.build()
```

For even more performance, enable `torch.compile` JIT acceleration:

```python
optimizer = PSO()
optimizer.compile(space.population)
optimizer.torch_compile(mode="reduce-overhead")
```

---

## Why Otorchmizer over Opytimizer?

Otorchmizer is a drop-in modernization of Opytimizer. The same algorithms, the same API style, but with a fundamentally different computational engine:

| | Opytimizer | Otorchmizer |
|---|---|---|
| **Backend** | NumPy | PyTorch |
| **Agent storage** | `List[Agent]` (Python objects) | `Population` tensor `(n, v, d)` |
| **Update loop** | `for agent in agents:` (Python) | Batched tensor ops (vectorized) |
| **GPU support** | ❌ None | ✅ CUDA, multi-GPU, CUDA Graphs |
| **Mixed precision** | ❌ float64 only | ✅ float16, bfloat16, float32, float64 |
| **JIT compilation** | ❌ None | ✅ `torch.compile` |
| **Algorithms** | 92 | 91 (3 specialized deferred) |
| **CPU speedup** | 1× (baseline) | **50–1,055×** |
| **GPU speedup** | — | **up to 2,311×** |

For a detailed migration guide, see [`docs/MIGRATION_GUIDE.md`](docs/MIGRATION_GUIDE.md).

---

## Algorithms (91 total)

| Family | Count | Algorithms |
|--------|-------|-----------|
| **Swarm** | 33 | ABC, ABO, AF, AIWPSO, BA, BOA, BWO, CS, CSA, EHO, FA, FFOA, FPA, FSO, GOA, JS, KH, MFO, MRFO, PIO, PSO, RPSO, SAVPSO, SBO, SCA, SFO, SOS, SSA, SSO, STOA, VPSO, WAOA, WOA |
| **Evolutionary** | 14 | BSA, DE, EP, ES, FOA, GA, GHS, GOGHS, HS, IHS, IWO, NGHS, RRA, SGHS |
| **Misc** | 5 | AOA, CEM, DOA, GS, HC |
| **Population** | 11 | AEO, AO, COA, EPO, GCO, GWO, HHO, OSA, PPA, PVS, RFO |
| **Science** | 19 | AIG, ASO, BH, CDO, EFO, EO, ESA, GSA, HGSO, LSA, MOA, MVO, SA, SMA, TEO, TWO, WCA, WDO, WEO |
| **Social** | 6 | BSO, CI, ISA, MVPA, QSA, SSD |
| **Boolean** | 3 | BMRFO, BPSO, UMDA |

---

## Benchmarks

Results from 432 paired configurations across 3 backends (NumPy, PyTorch CPU, PyTorch GPU on an NVIDIA RTX 4070):

| Metric | Value |
|--------|-------|
| Average CPU speedup | **173×** |
| Peak CPU speedup | **1,055×** (GA, 1000 agents, 100 dims) |
| Average GPU speedup | **169×** |
| Peak GPU speedup | **2,311×** (HC, 1000 agents, 100 dims) |
| Convergence quality | Parity with original |

GPU execution time stays **nearly constant** (~0.03–0.08s) regardless of problem size, while NumPy grows linearly.

```bash
# Quick CPU-only benchmarks
python report/benchmarks/run_benchmarks.py --quick

# Full benchmark suite with GPU
python report/benchmarks/run_benchmarks.py --extended --gpu

# Generate all 13 visualization plots
python report/benchmarks/plot_results.py --input report/benchmarks/results_extended.json \
    --outdir report/benchmarks/plots_extended
```

See the full [Migration Report](report/REPORT.md) for detailed analysis, tables, and all 13 benchmark plots.

---

## Testing

```bash
python -m pytest tests/ -v
# 197 passed
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [Migration Guide](docs/MIGRATION_GUIDE.md) | For existing Opytimizer users — API mapping, code examples, FAQ |
| [Architecture Guide](ARCHITECTURE.md) | Full design document covering Population, UpdateContext, DeviceManager |
| [Migration Report](report/REPORT.md) | Detailed performance analysis with 13 benchmark plots |
| [API Reference](docs/) | Sphinx auto-generated docs (`cd docs && make html`) |
| [Examples](examples/) | Commented examples for core, optimizers, applications, GPU, and math |

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
