# Migration Guide: Opytimizer → Otorchmizer

This guide describes the tensor-based workflow for Opytimizer users. Otorchmizer is
not a drop-in replacement, and matching class names do not guarantee identical
trajectories. Validate objective batching, dtype, update semantics, and device support
for the actual workload rather than extrapolating the historical benchmark results.

Known algorithm identity, equation, and parameter differences are listed in
[Algorithm migration limits](algorithm_limits.rst). In particular, ``ABO``, ``FSO``,
and ``SSO`` currently implement different algorithm identities from their intended
migration targets.

---

## Quick Start

### Before (Opytimizer)

```python
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace
import numpy as np


def sphere(x):
    return np.sum(x**2)


space = SearchSpace(n_agents=20, n_variables=5, lower_bound=[-5] * 5, upper_bound=[5] * 5)
optimizer = PSO()
function = Function(sphere)

opt = Opytimizer(space, optimizer, function)
opt.start(n_iterations=100)
print(space.best_agent.fit)
```

### After (Otorchmizer)

```python
from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.swarm import PSO
import torch


def sphere(x):
    return (x**2).sum(dim=(-1, -2))


space = Space(
    n_agents=20, n_variables=5, lower_bound=[-5] * 5, upper_bound=[5] * 5, device="auto"
)  # ← "auto" picks GPU if available
space.build()

optimizer = PSO()
function = Function(sphere)

opt = Otorchmizer(space, optimizer, function)
opt.start(n_iterations=100)
print(space.population.best_fitness.item())
```

**Key differences:**
1. `numpy` → `torch` in the fitness function
2. `SearchSpace` → `Space` with a `device` parameter
3. `space.best_agent.fit` → `space.population.best_fitness.item()`
4. Add `device="auto"` to run on GPU automatically

---

## Concept Mapping

| Opytimizer | Otorchmizer | Notes |
|------------|-------------|-------|
| `Agent` | `Population` (batched) | All agents stored as a single tensor `(n, v, d)` |
| `Agent.position` | `population.positions[i]` | Or use `AgentView(pop, i)` for compatibility |
| `Agent.fit` | `population.fitness[i]` | |
| `space.best_agent` | `population.best_position`, `population.best_fitness` | |
| `SearchSpace` | `Space` | With `device` parameter |
| `Function(fn)` | `Function(fn)` | Same API; uses `torch.vmap` internally |
| `inspect.signature` dispatch | `UpdateContext` dataclass | All optimizers get the same context |
| `np.array` | `torch.Tensor` | GPU-ready by default |
| `deepcopy(agent)` | `tensor.clone()` | Explicit tensor snapshot |
| Unevaluated fitness | Positive-infinity tensor | Representable across supported floating dtypes |

---

## Fitness Functions

### Translating NumPy to PyTorch

Most NumPy operations have direct PyTorch equivalents:

```python
# NumPy (Opytimizer)
def rastrigin(x):
    n = x.shape[0]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# PyTorch (Otorchmizer) — single-agent version (auto-vmapped)
def rastrigin(x):
    n = x.shape[0]
    return 10 * n + (x**2 - 10 * torch.cos(2 * torch.pi * x)).sum()


# PyTorch (Otorchmizer) — batch version (explicit)
def rastrigin_batch(positions):
    # positions: (n_agents, n_variables, n_dimensions)
    x = positions.squeeze(-1)  # (n_agents, n_variables)
    n = x.shape[1]
    return 10 * n + (x**2 - 10 * torch.cos(2 * torch.pi * x)).sum(dim=1)


fn = Function(rastrigin)  # auto-vmapped
fn = Function(rastrigin_batch, batch=True)  # manual batch
```

Every objective must produce one scalar per agent. Known `vmap` incompatibilities
fall back to per-agent evaluation with a diagnostic; other objective errors are
not retried. Objectives should be free of side effects because a failed transformed
call can execute part of the objective before the fallback. Native batch functions
must return a tensor shaped `(n_agents,)`, not a scalar broadcast across the population.

### Common NumPy → PyTorch translations

| NumPy | PyTorch |
|-------|---------|
| `np.sum(x)` | `x.sum()` or `torch.sum(x)` |
| `np.exp(x)` | `torch.exp(x)` |
| `np.sqrt(x)` | `torch.sqrt(x)` |
| `np.random.uniform(0, 1, size)` | `torch.rand(size)` |
| `np.random.normal(0, 1, size)` | `torch.randn(size)` |
| `np.linalg.norm(x)` | `torch.norm(x)` or `x.norm()` |
| `np.argmin(x)` | `torch.argmin(x)` |
| `np.clip(x, a, b)` | `torch.clamp(x, a, b)` |

---

## Device Selection (CPU / GPU / Multi-GPU)

```python
# Explicit CPU
space = Space(..., device="cpu")

# Auto-detect GPU
space = Space(..., device="auto")

# Specific GPU
space = Space(..., device="cuda:0")

# Check available GPUs
from otorchmizer.core import DeviceManager

print(DeviceManager.available_gpus())  # [device(cuda:0), device(cuda:1), ...]
```

### Multi-GPU Population Splitting

```python
from otorchmizer.core import DeviceManager

# Create a large population on GPU 0
space = Space(n_agents=10000, n_variables=100, ..., device="cuda:0")
space.build()

# Split across multiple GPUs
gpus = DeviceManager.available_gpus()
sub_populations = space.population.scatter(gpus)

# Merge back
from otorchmizer.core import Population
merged = Population.gather(sub_populations, target_device=gpus[0])
```

Splitting and gathering are data-management operations, not a parallel execution
scheduler. Create independent optimizer state for each shard before updating it.
The package has no `update_on` method. See
`examples/applications/gpu/multi_gpu_optimization.py` for an explicit update loop.
Shards on the source device may share storage with the source population.

---

## Mixed Precision

For memory-bound problems, use float16 or bfloat16:

```python
from otorchmizer.core import DeviceManager, Population

# Set dtype at population level
pop = Population(
    n_agents=1000,
    n_variables=100,
    n_dimensions=1,
    lower_bound=lb,
    upper_bound=ub,
    device=torch.device("cuda:0"),
    dtype=torch.float16,
)

# Or use DeviceManager autocast for automatic mixed-precision
dm = DeviceManager("cuda:0", dtype=torch.float16)
with dm.autocast():
    fitness = function(pop.positions)  # matmuls in float16, reductions in float32
```

---

## torch.compile (JIT Acceleration)

Compilation may help compatible workloads but can also introduce overhead or graph
breaks. Measure it with the intended algorithm and backend:

```python
opt = PSO()
opt.compile(space.population)
opt.torch_compile(mode="reduce-overhead")  # JIT-compile the update method

for i in range(n_iterations):
    opt(ctx)  # uses compiled graph via __call__
    space.population.clip()
    opt.evaluate(space.population, fn)
```

---

## CUDA Graphs

For compatible fixed-shape, in-place workloads where kernel launch overhead matters:

```python
from otorchmizer.core import DeviceManager


def update_step(positions, velocities):
    velocities.mul_(0.7).add_(torch.randn_like(positions) * 0.1)
    positions.add_(velocities)


# Pre-allocate static tensors on GPU
pos = torch.randn(100, 10, device="cuda")
vel = torch.zeros(100, 10, device="cuda")

# Capture the update loop as a CUDA Graph
runner = DeviceManager.capture_graph(update_step, pos, vel, warmup=3)

# Replay with near-zero Python overhead
for _ in range(1000):
    runner.replay()
```

Capture and warmup execute the callable and can mutate its inputs. Keep the captured
tensor storage alive, and do not assume every optimizer's Python control flow or
tensor replacement is CUDA-Graph compatible.

---

## Algorithm Reference

The following 91 optimizer classes are exported. Their source implementations and
documented variants, not their names alone, define behavior:

| Family | Algorithms |
|--------|-----------|
| **Swarm** (33) | ABC, ABO, AF, AIWPSO, BA, BOA, BWO, CS, CSA, EHO, FA, FFOA, FPA, FSO, GOA, JS, KH, MFO, MRFO, PIO, PSO, RPSO, SAVPSO, SBO, SCA, SFO, SOS, SSA, SSO, STOA, VPSO, WAOA, WOA |
| **Evolutionary** (14) | BSA, DE, EP, ES, FOA, GA, GHS, GOGHS, HS, IHS, IWO, NGHS, RRA, SGHS |
| **Misc** (5) | AOA, CEM, DOA, GS, HC |
| **Population** (11) | AEO, AO, COA, EPO, GCO, GWO, HHO, OSA, PPA, PVS, RFO |
| **Science** (19) | AIG, ASO, BH, CDO, EFO, EO, ESA, GSA, HGSO, LSA, MOA, MVO, SA, SMA, TEO, TWO, WCA, WDO, WEO |
| **Social** (6) | BSO, CI, ISA, MVPA, QSA, SSD |
| **Boolean** (3) | BMRFO, BPSO, UMDA |

```python
# Import any algorithm the same way:
from otorchmizer.optimizers.swarm import PSO, WOA, FA, CS, BA
from otorchmizer.optimizers.evolutionary import GA, DE, HS
from otorchmizer.optimizers.science import SA, GSA, EO
from otorchmizer.optimizers.population import GWO, HHO
from otorchmizer.optimizers.social import BSO, QSA
from otorchmizer.optimizers.boolean import BPSO, UMDA
from otorchmizer.optimizers.misc import HC, GS, CEM
```

---

## Parameter Passing

Parameters work identically — pass a dictionary to the constructor:

```python
# Opytimizer
pso = PSO(params={"w": 0.5, "c1": 2.0, "c2": 2.0})

# Otorchmizer — same API
pso = PSO(params={"w": 0.5, "c1": 2.0, "c2": 2.0})
```

---

## Accessing Results

```python
# Opytimizer
best_position = space.best_agent.position  # np.ndarray
best_fitness = space.best_agent.fit  # float

# Otorchmizer
best_position = space.population.best_position  # torch.Tensor
best_fitness = space.population.best_fitness  # torch.Tensor (scalar)

# Convert to numpy if needed
best_np = best_position.cpu().numpy()
best_val = best_fitness.item()
```

---

## FAQ

**Q: Do I need a GPU?**
No. Use `device="cpu"` explicitly, or `device="auto"` to select CUDA when available.
CPU performance depends on the optimizer, objective, population size, and thread configuration.

**Q: Can I use my existing NumPy fitness function?**
You'll need to rewrite it using PyTorch ops. Most translations are 1:1 (see table above). The key difference is using `torch.sum()` instead of `np.sum()`, etc.

**Q: Are the results identical?**
No identical-result or convergence guarantee is made. Dtype, random-number streams,
parallel update order, and algorithm variants can change results. Compare meaningful
quality and evaluation-budget invariants rather than expecting identical trajectories.

**Q: Which algorithms are not yet migrated?**
GP, LOA, and NDS are not exported here. `GraphSpace` is an experimental descriptor,
not a complete population-backed space for the main optimization loop.
