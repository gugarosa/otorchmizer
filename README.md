# Otorchmizer: A PyTorch-Powered Nature-Inspired Optimizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## Welcome to Otorchmizer.

Did you ever reach a bottleneck in your computational experiments? Are you tired of waiting hours for meta-heuristic optimization runs? If yes, Otorchmizer is the real deal! This package provides an easy-to-go implementation of **91 meta-heuristic optimization algorithms** — all powered by PyTorch tensors for GPU-accelerated performance. From populations to search spaces, from internal functions to external communication, we will foster all research related to optimizing stuff.

Otorchmizer builds on [Opytimizer](https://github.com/gugarosa/opytimizer) with tensor-based
population storage and GPU-capable operations. Performance depends on the algorithm,
objective, population shape, dtype, and hardware; the archived benchmarks are not a
guarantee for every optimizer or workload.

**Algorithm fidelity is not uniform.** Several exports have identity, equation,
or parameter differences from their Opytimizer migration targets. Read
[Algorithm migration limits](docs/algorithm_limits.rst) before using class names
alone as evidence in a scientific comparison.

Use Otorchmizer if you need a library or wish to:
* Create your optimization algorithm with automatic GPU support;
* Design or use pre-loaded optimization tasks at scale;
* Run compatible tensor workloads on CPU or CUDA and split populations across devices;
* Use `torch.compile` or CUDA Graphs when the update operations support them;
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

The optimizer families expose **91 meta-heuristic classes**. Implementations use tensor
populations, but some updates retain per-agent loops or backend-specific limitations.
Objective compatibility and operator support must be considered before selecting a
device or reduced-precision dtype.

### Spaces

One can see the space as the place that agents will update their positions and evaluate a fitness function. However, the newest approaches may consider a different type of space. Thinking about that, we are glad to support diverse space implementations.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing repeatedly.

### Visualization

Everyone needs images and plots to help visualize what is happening, correct? This package will provide every visual-related method for you. Check a specific variable convergence, your fitness function convergence, plot benchmark function surfaces, and much more!

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, Otorchmizer will be the one-to-go package that you will need, from the first installation to the daily tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

For a project managed by uv:

```bash
uv add otorchmizer
```

For an existing Python environment:

```bash
pip install otorchmizer
```

For development from this checkout:

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -e ".[dev,docs]"
uv run --no-sync pytest
uv run --no-sync pre-commit run --all-files
uv run --no-sync sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
```

The CPU backend above is an environment choice for local checks and CI, not a
restriction in package metadata. GPU environments should select a PyTorch build
matching their hardware. See [CONVENTIONS.md](CONVENTIONS.md) for contributor rules.

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
    return (x**2).sum(dim=(-1, -2))


n_agents = 20
n_variables = 2
lower_bound = [-10, -10]
upper_bound = [10, 10]

space = Space(n_agents=n_agents, n_variables=n_variables, lower_bound=lower_bound, upper_bound=upper_bound)
space.build()

optimizer = PSO()
function = Function(sphere)

opt = Otorchmizer(space, optimizer, function)
opt.start(n_iterations=1000)
```

---

## GPU Usage

Select a CUDA-capable PyTorch environment and a compatible objective before using a GPU:

```python
# Automatically uses GPU if available, otherwise falls back to CPU
space = Space(n_agents=1000, n_variables=100, lower_bound=-10.0, upper_bound=10.0, device="auto")
space.build()
```

Compilation is optional and does not guarantee a speedup:

```python
optimizer = PSO()
optimizer.compile(space.population)
optimizer.torch_compile(mode="reduce-overhead")
```

The engine dispatches through `optimizer(ctx)` to use the compiled callable.
Calling `optimizer.update(ctx)` directly intentionally uses the eager implementation.

---

## Why Otorchmizer over Opytimizer?

Otorchmizer retains the overall workflow, but is not a drop-in replacement:
population storage, objective batching, update dispatch, dtype, and some algorithm
variants differ. Validate migration results against the requirements of the workload.

| | Opytimizer | Otorchmizer |
|---|---|---|
| **Backend** | NumPy | PyTorch |
| **Agent storage** | `List[Agent]` (Python objects) | `Population` tensor `(n, v, d)` |
| **Update loop** | Primarily per-agent Python operations | Tensor operations, with per-agent loops where still required |
| **GPU support** | ❌ None | ✅ CUDA, multi-GPU, CUDA Graphs |
| **Mixed precision** | ❌ float64 only | ✅ float16, bfloat16, float32, float64 |
| **JIT compilation** | ❌ None | ✅ `torch.compile` |
| **Algorithms** | Depends on the referenced release | 91 exported optimizer classes |
| **Performance** | Workload-dependent reference | Measure equivalent behavior on the target environment |

For a detailed migration guide, see [`docs/MIGRATION_GUIDE.md`](docs/MIGRATION_GUIDE.md).

---

## Algorithms (91 total)

The table lists exported classes, not a certification that all 91 implementations
faithfully reproduce their migration targets. The known exceptions and unresolved
compatibility decisions are documented in [Algorithm migration limits](docs/algorithm_limits.rst).

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

The February 2026 [migration report](report/REPORT.md) records 432 paired configurations
across NumPy, PyTorch CPU, and an NVIDIA RTX 4070. The figures below summarize that
archive, not a new measurement of the current revision or every optimizer family.

| Metric | Value |
|--------|-------|
| Average CPU speedup | **173×** |
| Peak CPU speedup | **1,055×** (GA, 1000 agents, 100 dims) |
| Average GPU speedup | **169×** |
| Peak GPU speedup | **2,311×** (HC, 1000 agents, 100 dims) |
| Recorded convergence summary | Reported parity on the archived benchmark cases |

Do not extrapolate those timings to larger problems, different algorithms, compilation,
or reduced precision. The checked-in benchmark harness compares five optimizer classes,
and stochastic or parallelized variants need not follow identical trajectories.

```bash
# Install the reference implementation explicitly
uv pip install -e ".[benchmarks]"

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
```

The CPU suite is not a substitute for CUDA or multi-GPU execution on suitable hardware.

## Releasing

Review and merge changes before creating a GitHub release tagged `v<version>`.
Keep `pyproject.toml` and `otorchmizer.__version__` aligned. The publish workflow
reruns the supported interpreter matrix, style checks, documentation build, and
installed-wheel tests, then checks the tag against the package version.

PyPI publishing requires a configured Trusted Publisher for this repository and
`build-publish-to-pypi.yml`, or the repository's `PYPI_API_TOKEN` secret. Authentication
must be configured by a package owner before publication; it is not created by the PR.

---

## Documentation

| Resource | Description |
|----------|-------------|
| [Migration Guide](docs/MIGRATION_GUIDE.md) | For existing Opytimizer users — API mapping, code examples, FAQ |
| [Architecture Guide](ARCHITECTURE.md) | Full design document covering Population, UpdateContext, DeviceManager |
| [Algorithm migration limits](docs/algorithm_limits.rst) | Known identity, equation, parameter, and device limitations |
| [Migration Report](report/REPORT.md) | Detailed performance analysis with 13 benchmark plots |
| [API Reference](docs/) | Sphinx auto-generated docs (`cd docs && make html`) |
| [Examples](examples/) | Commented examples for core, optimizers, applications, GPU, and math |

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
