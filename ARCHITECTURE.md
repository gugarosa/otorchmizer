# Otorchmizer — PyTorch Migration Architecture Guide

> **Target:** Migrate Opytimizer (v3.1.4) to a PyTorch-native meta-heuristic optimization framework.
> **Goal:** Replace NumPy with PyTorch tensors, enable GPU/Multi-GPU acceleration, and modernize the architecture while preserving algorithmic correctness.

---

## Table of Contents

1. [Motivation & Goals](#1-motivation--goals)
2. [Current vs Target Architecture](#2-current-vs-target-architecture)
3. [Device Management Strategy](#3-device-management-strategy)
4. [Core Module Redesign](#4-core-module-redesign)
5. [Spaces Module Redesign](#5-spaces-module-redesign)
6. [Functions Module Redesign](#6-functions-module-redesign)
7. [Optimizers Module Redesign](#7-optimizers-module-redesign)
8. [Math Module Redesign](#8-math-module-redesign)
9. [Utils Module Redesign](#9-utils-module-redesign)
10. [Visualization Module Redesign](#10-visualization-module-redesign)
11. [Performance Optimization Strategy](#11-performance-optimization-strategy)
12. [Migration Roadmap](#12-migration-roadmap)
13. [Package Structure](#13-package-structure)
14. [Testing Strategy](#14-testing-strategy)

---

## 1. Motivation & Goals

### Why PyTorch?

The original Opytimizer uses NumPy arrays and Python-level `for` loops for all agent
operations. Profiling reveals:

| Bottleneck | Current (NumPy) | Target (PyTorch) |
|---|---|---|
| Position updates | Per-agent Python loop | Batched tensor ops on GPU |
| Fitness evaluation | Sequential per-agent | Vectorized batch evaluation |
| Random number generation | `np.random.*` per call | `torch.rand` (GPU-native, batched) |
| Pairwise distances (FA, GOA) | O(n²) Python loops | `torch.cdist` (single call) |
| Deep copies (~251 uses) | `copy.deepcopy` | `tensor.clone()` (10-100× faster) |
| Sorting (GA, HS, IWO) | Python `list.sort()` | `torch.sort` (parallel merge sort) |
| Agent state arrays | Separate `np.ndarray` per state | Single contiguous tensor per state |

### Design Principles

1. **Tensor-First:** All numerical data lives in PyTorch tensors — no NumPy at the computation boundary
2. **Batch-Vectorized:** Eliminate per-agent Python loops; operate on the full population tensor
3. **Device-Agnostic:** Code works identically on CPU, single GPU, or multi-GPU
4. **Backward-Compatible API:** Preserve the Opytimizer user-facing API shape (Space + Optimizer + Function → start)
5. **No Agent Objects in Hot Loops:** The `Agent` class becomes a view/accessor, not the storage unit
6. **Standardized Signatures:** Eliminate the fragile `inspect.signature` wiring; use a single `UpdateContext` object

---

## 2. Current vs Target Architecture

### Current (Opytimizer)

```
Agent (object per candidate)          → Python loop per agent
  └─ position: np.ndarray (n_vars, n_dims)
  └─ fit: float

Space (list of Agent objects)         → Sequential evaluation
  └─ agents: List[Agent]
  └─ best_agent: Agent

Optimizer (per-agent update logic)    → for i, agent in enumerate(space.agents)
  └─ compile() → np.zeros(...)
  └─ update() → loop + np.ops
```

### Target (Otorchmizer)

```
Population (batched tensor storage)   → Single tensor operation
  └─ positions: Tensor (n_agents, n_vars, n_dims)   ← ALL agents in one tensor
  └─ fitness:   Tensor (n_agents,)                   ← ALL fitness in one tensor
  └─ best_position: Tensor (n_vars, n_dims)
  └─ best_fitness:  Tensor (1,)

Space (population manager)            → Vectorized evaluation
  └─ population: Population
  └─ device: torch.device

Optimizer (batched update logic)      → Fully vectorized tensor ops
  └─ compile() → torch.zeros(..., device=device)
  └─ update(ctx: UpdateContext) → tensor ops (no Python loops)
```

---

## 3. Device Management Strategy

### 3.1 Device Resolution

```python
class DeviceManager:
    """Centralized device management for the entire optimization pipeline."""

    def __init__(self, device: str | torch.device = "auto"):
        self.device = self._resolve(device)

    @staticmethod
    def _resolve(device: str | torch.device) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        return torch.device(device)

    def tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create a tensor on the managed device."""
        return torch.tensor(*args, **kwargs, device=self.device)

    def zeros(self, *shape, **kwargs) -> torch.Tensor:
        return torch.zeros(*shape, **kwargs, device=self.device)

    def rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.device)
```

### 3.2 Multi-GPU Strategy

For multi-GPU, the primary parallelism axis is **population splitting** — each GPU
handles a subset of agents:

```python
class MultiGPUPopulation:
    """Splits population across available GPUs for large-scale optimization."""

    def __init__(self, n_agents: int, n_variables: int, n_dimensions: int,
                 devices: list[torch.device]):
        self.devices = devices
        n_per_gpu = n_agents // len(devices)

        # Each GPU holds a shard of the population
        self.shards = [
            Population(n_per_gpu, n_variables, n_dimensions, device=dev)
            for dev in devices
        ]

    def gather_best(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Finds global best across all GPU shards."""
        bests = [(s.best_fitness.to("cpu"), s.best_position.to("cpu"))
                 for s in self.shards]
        idx = torch.stack([b[0] for b in bests]).argmin()
        return bests[idx]
```

**When multi-GPU helps:**
- Large populations (n_agents > 10,000)
- Expensive fitness functions (neural network evaluation)
- Pairwise algorithms (FA, GOA) where O(n²) benefits from splitting

**When it does NOT help:**
- Small populations (n < 100) — transfer overhead dominates
- Algorithms with heavy agent-to-agent dependencies (serial by nature)

### 3.3 Device Propagation

The device flows top-down: `Otorchmizer(device=...)` → `Space` → `Population` → `Optimizer.compile()`. All tensors created within the pipeline inherit the device from the `Population` they operate on.

---

## 4. Core Module Redesign

### 4.1 Population (replaces Agent + Space internals)

The fundamental shift: **agents are no longer individual objects — they are rows in a tensor.**

```python
class Population:
    """Stores the entire population as contiguous tensors.

    Instead of List[Agent] with per-object positions/fitness,
    everything is a single batched tensor for GPU-friendly access.
    """

    def __init__(self, n_agents: int, n_variables: int, n_dimensions: int,
                 lower_bound: torch.Tensor, upper_bound: torch.Tensor,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.n_agents = n_agents
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions

        # Core population tensors — ALL agents in one contiguous block
        self.positions = torch.zeros(n_agents, n_variables, n_dimensions, device=device)
        self.fitness   = torch.full((n_agents,), float("inf"), device=device)

        # Bounds as tensors (broadcastable)
        self.lb = lower_bound.to(device)  # (n_variables,) or (n_variables, 1)
        self.ub = upper_bound.to(device)

        # Global best (single candidate)
        self.best_position = torch.zeros(n_variables, n_dimensions, device=device)
        self.best_fitness  = torch.tensor(float("inf"), device=device)

    def clip(self) -> None:
        """Clamp all positions to bounds — fully vectorized."""
        lb = self.lb.unsqueeze(0)  # (1, n_vars, 1)
        ub = self.ub.unsqueeze(0)
        self.positions.clamp_(min=lb, max=ub)

    def initialize_uniform(self) -> None:
        """Fill all positions uniformly within bounds — single tensor op."""
        lb = self.lb.unsqueeze(0)
        ub = self.ub.unsqueeze(0)
        self.positions = torch.rand_like(self.positions) * (ub - lb) + lb

    def initialize_binary(self) -> None:
        """Fill positions with binary values."""
        self.positions = torch.round(torch.rand_like(self.positions))

    def update_best(self) -> None:
        """Find population-wide best — vectorized argmin."""
        best_idx = self.fitness.argmin()
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx].clone()
            self.best_position = self.positions[best_idx].clone()

    def clone_positions(self) -> torch.Tensor:
        """Replace deep copy with tensor clone (10-100× faster)."""
        return self.positions.clone()
```

**Performance impact:** This single change eliminates ~80% of the per-agent loop overhead. Operations like `clip_by_bound()` go from O(n × v) Python iterations to a single `clamp_` call.

### 4.2 AgentView (optional backward-compatible accessor)

For code that still needs to reference individual agents (e.g., during migration):

```python
class AgentView:
    """A lightweight, non-owning view into a Population row.

    Does NOT store data — it references the Population tensors.
    Used for backward compatibility and debugging, never in hot loops.
    """

    def __init__(self, population: Population, index: int):
        self._pop = population
        self._idx = index

    @property
    def position(self) -> torch.Tensor:
        return self._pop.positions[self._idx]

    @position.setter
    def position(self, value: torch.Tensor) -> None:
        self._pop.positions[self._idx] = value

    @property
    def fit(self) -> float:
        return self._pop.fitness[self._idx].item()

    @fit.setter
    def fit(self, value: float) -> None:
        self._pop.fitness[self._idx] = value
```

### 4.3 UpdateContext (replaces inspect.signature wiring)

The original uses `inspect.signature` to dynamically resolve `update()` arguments,
which is fragile (7+ different signature patterns exist). Replace with a single context object:

```python
@dataclass
class UpdateContext:
    """All information an optimizer might need during update().

    Every optimizer receives the same context — use what you need, ignore the rest.
    Eliminates the fragile inspect.signature() dynamic wiring.
    """
    space: "Space"
    function: "Function"
    iteration: int
    n_iterations: int
    device: torch.device
```

### 4.4 Optimizer Base Class

```python
class Optimizer:
    """Base class for all optimization algorithms.

    Subclasses MUST implement update(ctx: UpdateContext).
    Subclasses MAY override evaluate() and compile().
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.algorithm = self.__class__.__name__
        self.params = params or {}
        self.built = False

    def build(self, params: dict[str, Any] | None = None) -> None:
        if params:
            self.params.update(params)
            for k, v in params.items():
                setattr(self, k, v)
        self.built = True

    def compile(self, population: Population) -> None:
        """Pre-allocate algorithm-specific state tensors on the correct device."""
        pass

    def evaluate(self, population: Population, function: "Function") -> None:
        """Batch-evaluate all agents and update global best.

        Default: vectorized batch evaluation (no per-agent loop).
        Override only when custom evaluation logic is needed.
        """
        population.fitness = function(population.positions)
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Apply the algorithm's position-update rule.

        MUST be implemented by every optimizer subclass.
        Should use ONLY tensor operations — no Python loops over agents.
        """
        raise NotImplementedError
```

### 4.5 Function

```python
class Function:
    """Wraps a user objective function with optional batch support.

    The function receives a tensor of shape (n_agents, n_variables, n_dimensions)
    and MUST return a tensor of shape (n_agents,).

    If the user provides a single-agent function, we auto-vectorize it
    using torch.vmap for transparent batching.
    """

    def __init__(self, pointer: callable, batch: bool = False):
        self.name = getattr(pointer, "__name__", pointer.__class__.__name__)

        if batch:
            # User guarantees pointer handles (n_agents, n_vars, n_dims) → (n_agents,)
            self._fn = pointer
        else:
            # User provides f(x) for a single agent: (n_vars, n_dims) → scalar
            # We auto-vectorize with vmap
            self._fn = torch.vmap(pointer)

        self.built = True

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        return self._fn(positions)
```

**Key design:** `torch.vmap` auto-vectorizes single-agent functions into batched ones, so users can write simple `f(x) → scalar` functions and get GPU-parallel evaluation for free.

### 4.6 Space

```python
class Space:
    """Base search space managing a Population of candidates."""

    def __init__(self, n_agents: int, n_variables: int, n_dimensions: int,
                 lower_bound: list[float], upper_bound: list[float],
                 mapping: list[str] | None = None,
                 device: str | torch.device = "auto"):
        self.device = DeviceManager(device).device

        lb = torch.tensor(lower_bound, dtype=torch.float32)
        ub = torch.tensor(upper_bound, dtype=torch.float32)

        self.population = Population(
            n_agents, n_variables, n_dimensions, lb, ub, self.device
        )
        self.mapping = mapping or [f"x{i}" for i in range(n_variables)]
        self.built = False

    def build(self) -> None:
        self._initialize()
        self.built = True

    def _initialize(self) -> None:
        """Override in subclasses for custom initialization."""
        self.population.initialize_uniform()

    def clip(self) -> None:
        self.population.clip()

    # Convenience accessors (backward compat)
    @property
    def n_agents(self) -> int:
        return self.population.n_agents

    @property
    def best_agent_position(self) -> torch.Tensor:
        return self.population.best_position

    @property
    def best_agent_fitness(self) -> float:
        return self.population.best_fitness.item()
```

### 4.7 Node, Block, Cell

These structures (used by GP and graph-based optimization) are **not tensor-heavy** —
they are graph/tree structures. The migration strategy here is:

- **Node:** Keep the tree structure in Python, but store terminal values as `torch.Tensor`
  instead of `np.ndarray`. The `_evaluate()` function uses `torch` math ops.
- **Block/Cell:** Keep NetworkX DAG structure. Block pointers can wrap PyTorch callables.
- **No GPU benefit:** Tree manipulation is inherently sequential; GPU won't help here.
  The benefit comes when evaluating tree outputs across the population (batched).

---

## 5. Spaces Module Redesign

All spaces become thin wrappers around `Space` with different `_initialize()`:

| Space | Changes |
|---|---|
| **SearchSpace** | `_initialize()` → `population.initialize_uniform()` (default) |
| **BooleanSpace** | `_initialize()` → `population.initialize_binary()`, bounds fixed [0,1] |
| **GridSpace** | Build grid with `torch.meshgrid`, assign to `population.positions` |
| **HyperComplexSpace** | `n_dimensions > 1`, same uniform init |
| **TreeSpace** | Trees stay as Python `Node` objects; terminal values become tensors |
| **ParetoSpace** | Load `data_points` as tensor, no clipping override |
| **GraphSpace** | Minimal — stays similar to current placeholder |

**GridSpace vectorization example:**

```python
class GridSpace(Space):
    def _create_grid(self) -> None:
        ranges = [torch.arange(lb, ub + s, s, device=self.device)
                  for lb, ub, s in zip(self.lb, self.ub, self.step)]
        mesh = torch.meshgrid(*ranges, indexing="ij")
        self.population.positions = torch.stack([m.ravel() for m in mesh], dim=1).unsqueeze(-1)
        self.population.n_agents = self.population.positions.shape[0]
```

---

## 6. Functions Module Redesign

### 6.1 Function (single-objective)

See Section 4.5. The key change is `torch.vmap` auto-batching.

### 6.2 ConstrainedFunction

```python
class ConstrainedFunction(Function):
    def __init__(self, pointer: callable, constraints: list[callable],
                 penalty: float = 0.0, batch: bool = False):
        super().__init__(pointer, batch)
        self.constraints = constraints
        self.penalty = penalty

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        fitness = self._fn(positions)  # (n_agents,)
        for constraint in self.constraints:
            # constraint returns bool tensor (n_agents,)
            mask = ~constraint(positions)
            fitness = fitness + mask.float() * self.penalty * fitness.abs()
        return fitness
```

**Performance note:** Constraints are evaluated as **boolean masks** on the full population tensor — no per-agent branching.

### 6.3 MultiObjectiveFunction

```python
class MultiObjectiveFunction:
    def __init__(self, functions: list[callable], batch: bool = False):
        self.functions = [Function(f, batch=batch) for f in functions]
        self.built = True

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        # Returns (n_agents, n_objectives)
        return torch.stack([f(positions) for f in self.functions], dim=-1)
```

### 6.4 MultiObjectiveWeightedFunction

```python
class MultiObjectiveWeightedFunction(MultiObjectiveFunction):
    def __init__(self, functions: list[callable], weights: list[float], batch: bool = False):
        super().__init__(functions, batch)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        objectives = super().__call__(positions)  # (n_agents, n_objectives)
        return (objectives * self.weights.to(objectives.device)).sum(dim=-1)
```

---

## 7. Optimizers Module Redesign

### 7.1 Vectorization Strategy by Pattern

The 85+ optimizers fall into distinct computational patterns. Each pattern has a
specific vectorization strategy:

#### Pattern A: Velocity-Based (PSO, BA, KH, BPSO) — ~15 optimizers

**Before (per-agent loop):**
```python
for i, agent in enumerate(space.agents):
    r1 = r.generate_uniform_random_number()
    self.velocity[i] = (self.w * self.velocity[i]
                       + self.c1 * r1 * (self.local_position[i] - agent.position)
                       + self.c2 * r2 * (space.best_agent.position - agent.position))
    agent.position += self.velocity[i]
```

**After (batched tensor):**
```python
def update(self, ctx: UpdateContext) -> None:
    pop = ctx.space.population
    r1 = torch.rand(pop.n_agents, 1, 1, device=pop.device)
    r2 = torch.rand(pop.n_agents, 1, 1, device=pop.device)

    self.velocity = (self.w * self.velocity
                    + self.c1 * r1 * (self.local_position - pop.positions)
                    + self.c2 * r2 * (pop.best_position.unsqueeze(0) - pop.positions))
    pop.positions += self.velocity
```

**Speedup:** Eliminates n_agents iterations → single tensor op. On GPU with 1000 agents, expect ~50-200× speedup.

#### Pattern B: Leader-Follower (WOA, SSA, MFO) — ~20 optimizers

These update each agent toward the best or a leader. Already near-vectorizable:

```python
def update(self, ctx: UpdateContext) -> None:
    pop = ctx.space.population
    t = ctx.iteration / ctx.n_iterations

    a = 2.0 - 2.0 * t  # linearly decreasing
    r = torch.rand(pop.n_agents, pop.n_variables, pop.n_dimensions, device=pop.device)
    A = 2 * a * r - a
    C = 2 * torch.rand_like(r)

    D = torch.abs(C * pop.best_position.unsqueeze(0) - pop.positions)
    pop.positions = pop.best_position.unsqueeze(0) - A * D
```

#### Pattern C: All-Pairs Interaction (FA, GOA, KH sensing) — ~5 optimizers

**This is where PyTorch shines most.** The original has O(n²) nested Python loops.

**Before (Firefly — 4 nested loops):**
```python
for i, agent in enumerate(space.agents):
    for j, other in enumerate(space.agents):
        if other.fit < agent.fit:
            dist = euclidean_distance(agent.position, other.position)
            beta = self.beta * np.exp(-self.gamma * dist**2)
            agent.position += beta * (other.position - agent.position) + ...
```

**After (vectorized pairwise):**
```python
def update(self, ctx: UpdateContext) -> None:
    pop = ctx.space.population
    # Pairwise distances: (n_agents, n_agents)
    pos_flat = pop.positions.view(pop.n_agents, -1)
    dist_matrix = torch.cdist(pos_flat, pos_flat)  # Single GPU-optimized call

    # Attractiveness matrix: (n_agents, n_agents)
    beta_matrix = self.beta * torch.exp(-self.gamma * dist_matrix ** 2)

    # Mask: only move toward brighter (lower fitness) fireflies
    fit_i = pop.fitness.unsqueeze(1)  # (n, 1)
    fit_j = pop.fitness.unsqueeze(0)  # (1, n)
    mask = (fit_j < fit_i).float()    # (n, n) — 1 where j is better than i

    # Position differences: (n_agents, n_agents, n_vars * n_dims)
    diff = pos_flat.unsqueeze(1) - pos_flat.unsqueeze(0)  # (n, n, d)

    # Weighted movement (sum over all attracting agents)
    movement = (mask.unsqueeze(-1) * beta_matrix.unsqueeze(-1) * diff).sum(dim=1)
    pop.positions += movement.view_as(pop.positions) + self.alpha * (torch.rand_like(pop.positions) - 0.5)
```

**Speedup:** O(n²) Python loops → single `torch.cdist` + tensor broadcast. For n=500 on GPU: ~500-1000× speedup.

#### Pattern D: Selection + Crossover + Mutation (GA, DE, EP, HS) — ~15 optimizers

**Selection** (roulette/tournament) — vectorize with `torch.multinomial`:
```python
# Roulette selection (vectorized)
inv_fitness = fitness.max() - fitness + EPSILON
probs = inv_fitness / inv_fitness.sum()
selected = torch.multinomial(probs, n_selected, replacement=False)
```

**Crossover** (BLX or uniform):
```python
# Batch crossover for all pairs simultaneously
r = torch.rand(n_pairs, n_variables, 1, device=device)
alpha = r * parents_a + (1 - r) * parents_b
beta  = r * parents_b + (1 - r) * parents_a
```

**Mutation** (Gaussian):
```python
mask = torch.rand(n_offspring, n_variables, device=device) < self.p_mutation
noise = torch.randn_like(positions) * mask.unsqueeze(-1).float()
positions += noise
```

#### Pattern E: Sorting-Based (IWO, MFO, HS, GA final sort) — ~10 optimizers

```python
# Vectorized sort (replaces list.sort + deepcopy)
sorted_idx = torch.argsort(pop.fitness)
pop.positions = pop.positions[sorted_idx]
pop.fitness = pop.fitness[sorted_idx]
```

#### Pattern F: Clustering-Based (BSO) — ~2 optimizers

```python
# K-means with PyTorch (replaces custom numpy implementation)
from otorchmizer.math.general import kmeans_torch

labels = kmeans_torch(pop.positions, n_clusters=5, device=pop.device)
```

### 7.2 compile() Redesign

All 47 optimizers that override `compile()` will create tensors on the correct device:

```python
def compile(self, population: Population) -> None:
    dev = population.device
    shape = (population.n_agents, population.n_variables, population.n_dimensions)

    self.velocity = torch.zeros(shape, device=dev)
    self.local_position = torch.zeros(shape, device=dev)
    self.local_fitness = torch.full((population.n_agents,), float("inf"), device=dev)
```

### 7.3 Custom evaluate() Redesign

The 8 optimizers that override `evaluate()` (PSO, GP, CSA, BPSO, ISA, SSO, WAOA, GAO)
will use batch evaluation with conditional updates:

```python
# PSO custom evaluate (local best tracking) — vectorized
def evaluate(self, population: Population, function: Function) -> None:
    new_fitness = function(population.positions)  # (n_agents,)

    # Update local bests where improved (no loop)
    improved = new_fitness < self.local_fitness
    self.local_position[improved] = population.positions[improved].clone()
    self.local_fitness[improved] = new_fitness[improved].clone()

    # Update population fitness
    population.fitness = self.local_fitness.clone()
    population.update_best()
```

### 7.4 Optimizer Families Summary

| Family | Count | Primary Vectorization Strategy |
|---|---|---|
| Swarm | 29 | Patterns A + B + C (velocity, leader-follow, pairwise) |
| Evolutionary | 11 (16 variants) | Pattern D (selection, crossover, mutation) + E (sorting) |
| Science | 20 | Patterns A + B (physics-based forces → tensor ops) |
| Population | 12 | Pattern B (leader-follow, hierarchy) |
| Social | 6 | Patterns B + F (leader-follow, clustering) |
| Boolean | 3 | Pattern A (binary velocity + sigmoid) |
| Misc | 6 | Patterns B + E (various) |

---

## 8. Math Module Redesign

### 8.1 random.py → PyTorch RNG

All random generators become GPU-native and support batched generation:

```python
def generate_uniform_random_number(
    low: float = 0.0, high: float = 1.0,
    size: int | tuple = 1,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,)
    return torch.rand(size, device=device) * (high - low) + low

def generate_gaussian_random_number(
    mean: float = 0.0, variance: float = 1.0,
    size: int | tuple = 1,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,)
    return torch.randn(size, device=device) * variance + mean

def generate_integer_random_number(
    low: int = 0, high: int = 1,
    size: int | tuple | None = None,
    exclude_value: int | None = None,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if size is None:
        result = torch.randint(low, high, (1,), device=device).item()
    else:
        if isinstance(size, int):
            size = (size,)
        result = torch.randint(low, high, size, device=device)
    return result

def generate_binary_random_number(
    size: int = 1,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    return torch.round(torch.rand(size, device=device))
```

**Performance note:** PyTorch RNG on GPU generates millions of random numbers in
microseconds. The original Opytimizer makes 10-20 `np.random` calls per agent per
iteration — this becomes a single batched call for the entire population.

### 8.2 distribution.py → PyTorch Distributions

```python
import torch
import torch.distributions as dist

def generate_levy_distribution(
    beta: float = 0.1, size: int | tuple = 1,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    from math import gamma, pi, sin

    if isinstance(size, int):
        size = (size,)

    num = gamma(1 + beta) * sin(pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)

    u = torch.randn(size, device=device) * sigma
    v = torch.randn(size, device=device)

    return u / torch.abs(v) ** (1 / beta)

def generate_bernoulli_distribution(
    prob: float = 0.0, size: int | tuple = 1,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,)
    return torch.bernoulli(torch.full(size, prob, device=device))

def generate_choice_distribution(
    n: int, probs: torch.Tensor, size: int
) -> torch.Tensor:
    return torch.multinomial(probs, size, replacement=False)
```

### 8.3 general.py → Vectorized Utilities

```python
def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched euclidean distance."""
    return torch.linalg.norm(x - y, dim=-1)

def pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix — replaces nested loops in FA/GOA/KH."""
    flat = positions.view(positions.shape[0], -1)
    return torch.cdist(flat, flat)

def kmeans_torch(x: torch.Tensor, n_clusters: int,
                 max_iterations: int = 100, tol: float = 1e-4) -> torch.Tensor:
    """GPU-accelerated K-means."""
    n_samples = x.shape[0]
    flat = x.view(n_samples, -1)
    device = x.device

    # Random initialization
    idx = torch.randperm(n_samples, device=device)[:n_clusters]
    centroids = flat[idx].clone()
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)

    for _ in range(max_iterations):
        dists = torch.cdist(flat, centroids)
        new_labels = dists.argmin(dim=1)

        if (new_labels == labels).float().mean() > (1 - tol):
            break
        labels = new_labels

        for i in range(n_clusters):
            mask = labels == i
            if mask.any():
                centroids[i] = flat[mask].mean(dim=0)

    return labels

def tournament_selection(fitness: torch.Tensor, n: int,
                          size: int = 2) -> torch.Tensor:
    """Vectorized tournament selection."""
    device = fitness.device
    # Draw `size` random candidates for each of `n` tournaments
    candidates = torch.randint(0, len(fitness), (n, size), device=device)
    candidate_fitness = fitness[candidates]  # (n, size)
    winners = candidates[torch.arange(n, device=device), candidate_fitness.argmin(dim=1)]
    return winners
```

### 8.4 hyper.py → PyTorch Hypercomplex Math

```python
def norm(array: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(array, dim=1)

def span(array: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    n = norm(array) / (array.shape[1] ** 0.5)
    return (ub - lb) * n + lb
```

---

## 9. Utils Module Redesign

### 9.1 constant.py — Same constants, now as tensors where needed

```python
EPSILON = 1e-32
FLOAT_MAX = float("inf")  # Use torch-compatible infinity
LIGHT_SPEED = 3e5
FUNCTION_N_ARGS = { ... }  # Unchanged (GP-specific)
TEST_EPSILON = 100
```

### 9.2 exception.py — Unchanged

Custom exceptions remain pure Python — no tensor involvement.

### 9.3 history.py — Tensor-aware history

```python
class History:
    def dump(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key == "best_agent":
                output = (value[0].detach().cpu().tolist(), value[1].item())
            elif key == "positions":
                output = value.detach().cpu().tolist()
            else:
                output = value

            if not hasattr(self, key):
                setattr(self, key, [output])
            else:
                getattr(self, key).append(output)
```

**Key:** Always `.detach().cpu()` before storing history to avoid GPU memory leaks and autograd graph retention.

### 9.4 callback.py — Updated signatures

```python
class Callback:
    def on_evaluate_before(self, population: Population, function: Function) -> None: ...
    def on_update_before(self, ctx: UpdateContext) -> None: ...
    def on_update_after(self, ctx: UpdateContext) -> None: ...

class DiscreteSearchCallback(Callback):
    def on_evaluate_before(self, population: Population, function: Function) -> None:
        for i, allowed in enumerate(self.allowed_values):
            allowed_t = torch.tensor(allowed, device=population.device)
            diffs = torch.abs(population.positions[:, i, :] - allowed_t.unsqueeze(0))
            nearest_idx = diffs.argmin(dim=-1)
            population.positions[:, i, :] = allowed_t[nearest_idx].unsqueeze(-1)
```

### 9.5 logging.py — Unchanged

Logging is pure I/O — no migration needed.

---

## 10. Visualization Module Redesign

Minimal changes — visualization always works on CPU numpy data:

```python
def plot(*args, **kwargs) -> None:
    # Convert any tensors to numpy for matplotlib
    args = [a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in args]
    # ... rest unchanged ...
```

---

## 11. Performance Optimization Strategy

### 11.1 Optimization Priority Matrix

| Optimization | Impact | Effort | Priority |
|---|---|---|---|
| **Batch fitness evaluation** (vmap) | 🔴 Critical | Low | P0 |
| **Population as tensor** (no Agent objects) | 🔴 Critical | Medium | P0 |
| **Pairwise distance** (torch.cdist) | 🔴 Critical for FA/GOA/KH | Low | P0 |
| **Batch RNG** (single call per population) | 🟡 High | Low | P1 |
| **Vectorized selection** (multinomial) | 🟡 High | Low | P1 |
| **Eliminate deepcopy** (tensor.clone) | 🟡 High | Low | P1 |
| **Vectorized sorting** (torch.argsort) | 🟢 Medium | Low | P2 |
| **GPU-native K-means** | 🟢 Medium | Medium | P2 |
| **Multi-GPU population split** | 🟢 Medium | High | P3 |
| **torch.compile on hot paths** | 🟢 Medium | Low | P3 |

### 11.2 Memory Optimization

```
Original (Agent objects):
  Per agent: Python object overhead (~400 bytes)
           + np.ndarray position (~v×d×8 bytes + 128 bytes header)
           + np.ndarray lb, ub (same)
           + float fit (28 bytes)
  For 1000 agents, 10 vars: ~1000 × (400 + 3×(80+128) + 28) ≈ 1 MB

Otorchmizer (Population tensor):
  Positions: 1000 × 10 × 1 × 4 bytes = 40 KB  (float32, contiguous)
  Fitness:   1000 × 4 bytes = 4 KB
  Total: ~44 KB + bounds (negligible)

Memory reduction: ~20× for population storage
GPU bonus: contiguous memory = no cache misses
```

### 11.3 torch.compile Integration

For optimizers with complex update rules, `torch.compile` can JIT-optimize the tensor
operations:

```python
class PSO(Optimizer):
    def compile(self, population: Population) -> None:
        super().compile(population)
        # JIT-compile the update for maximum performance
        self._update_positions = torch.compile(self._update_positions_impl)

    def _update_positions_impl(self, positions, velocity, local_pos, best_pos, w, c1, c2):
        r1 = torch.rand_like(positions)
        r2 = torch.rand_like(positions)
        velocity = (w * velocity
                   + c1 * r1 * (local_pos - positions)
                   + c2 * r2 * (best_pos.unsqueeze(0) - positions))
        return positions + velocity, velocity
```

### 11.4 Benchmarking Targets

| Scenario | Opytimizer (CPU) | Otorchmizer (CPU) | Otorchmizer (GPU) |
|---|---|---|---|
| PSO, 100 agents, 10 vars, 1000 iter | Baseline | ~5-10× faster | ~50-100× faster |
| FA, 500 agents, 10 vars, 100 iter | Baseline | ~20-50× faster | ~500-1000× faster |
| GA, 200 agents, 20 vars, 500 iter | Baseline | ~3-5× faster | ~20-50× faster |
| BSO, 100 agents, 10 vars, 100 iter | Baseline | ~5-10× faster | ~30-50× faster |

---

## 12. Migration Roadmap

### Phase 1: Foundation (Core + Math + Utils)
- [ ] Set up package structure and `pyproject.toml`
- [ ] Implement `Population`, `DeviceManager`, `UpdateContext`
- [ ] Port `math/` module to PyTorch (random, distribution, general, hyper)
- [ ] Port `utils/` module (constant, exception, history, callback, logging)
- [ ] Implement `Function` with `torch.vmap` auto-batching
- [ ] Implement base `Space` and `Optimizer` classes
- [ ] Implement `Otorchmizer` orchestrator
- [ ] Port `SearchSpace` and `BooleanSpace`
- [ ] Unit tests for all core components (CPU + GPU)

### Phase 2: High-Impact Optimizers
- [ ] PSO family (PSO, AIWPSO, RPSO, SAVPSO, VPSO) — Pattern A
- [ ] WOA, SSA, MFO, SCA — Pattern B
- [ ] FA, GOA — Pattern C (biggest speedup opportunity)
- [ ] GA, DE — Pattern D
- [ ] GS, HC — Pattern E (simplest)
- [ ] Integration tests comparing results with Opytimizer

### Phase 3: Complete Optimizer Coverage
- [ ] Remaining swarm optimizers (ABC, BA, CS, etc.)
- [ ] Science-based optimizers (GSA, SA, BH, etc.)
- [ ] Population-based optimizers (GWO, HHO, etc.)
- [ ] Social optimizers (BSO, CI, etc.)
- [ ] Boolean optimizers (BMRFO, BPSO, UMDA)
- [ ] Misc optimizers (AOA, CEM, NDS, DOA)

### Phase 4: Advanced Spaces + GP
- [ ] TreeSpace + GP/GSGP (trees as Python objects, terminal values as tensors)
- [ ] HyperComplexSpace
- [ ] ParetoSpace + NDS
- [ ] GridSpace
- [ ] GraphSpace

### Phase 5: Performance + Multi-GPU
- [ ] `torch.compile` integration for hot-path optimizers
- [ ] Multi-GPU population splitting
- [ ] Comprehensive benchmarking suite
- [ ] Memory profiling and optimization

### Phase 6: Polish
- [ ] Visualization module (tensor → numpy bridge)
- [ ] Documentation and migration guide
- [ ] Examples and integration demos
- [ ] PyPI release

---

## 13. Package Structure

```
otorchmizer/
├── __init__.py
├── otorchmizer.py              # Entry-point orchestrator
├── core/
│   ├── __init__.py
│   ├── population.py           # Population (replaces Agent as storage)
│   ├── agent_view.py           # AgentView (backward-compat accessor)
│   ├── function.py             # Function with vmap auto-batching
│   ├── optimizer.py            # Optimizer base + UpdateContext
│   ├── space.py                # Space base class
│   ├── device.py               # DeviceManager
│   ├── node.py                 # Node (GP trees — tensor values)
│   ├── block.py                # Block/InputBlock/InnerBlock/OutputBlock
│   └── cell.py                 # Cell (DAG)
├── functions/
│   ├── __init__.py
│   ├── constrained.py          # ConstrainedFunction
│   └── multi_objective/
│       ├── __init__.py
│       ├── standard.py         # MultiObjectiveFunction
│       └── weighted.py         # MultiObjectiveWeightedFunction
├── math/
│   ├── __init__.py
│   ├── distribution.py         # Lévy, Bernoulli, choice (PyTorch)
│   ├── general.py              # distance, kmeans, selection (PyTorch)
│   ├── hyper.py                # Hypercomplex math (PyTorch)
│   └── random.py               # RNG wrappers (PyTorch)
├── optimizers/
│   ├── __init__.py
│   ├── boolean/                # BMRFO, BPSO, UMDA
│   ├── evolutionary/           # BSA, DE, EP, ES, FOA, GA, GP, GSGP, HS, IWO, RRA
│   ├── misc/                   # AOA, CEM, DOA, GS, HC, NDS
│   ├── population/             # AEO, AO, COA, EPO, GCO, GWO, HHO, LOA, OSA, PPA, PVS, RFO
│   ├── science/                # AIG, ASO, BH, CDO, EFO, EO, ESA, GSA, HGSO, LSA, MOA, MVO, SA, SMA, TEO, TWO, WCA, WDO, WEO, WWO
│   ├── social/                 # BSO, CI, ISA, MVPA, QSA, SSD
│   └── swarm/                  # ABC, ABO, AF, BA, BOA, BWO, CS, CSA, EHO, FA, FFOA, FPA, FSO, GOA, JS, KH, MFO, MRFO, PIO, PSO, SBO, SCA, SFO, SOS, SSA, SSO, STOA, WAOA, WOA
├── spaces/
│   ├── __init__.py
│   ├── boolean.py
│   ├── graph.py
│   ├── grid.py
│   ├── hyper_complex.py
│   ├── pareto.py
│   ├── search.py
│   └── tree.py
├── utils/
│   ├── __init__.py
│   ├── callback.py
│   ├── constant.py
│   ├── exception.py
│   ├── history.py
│   └── logging.py
└── visualization/
    ├── __init__.py
    ├── convergence.py
    └── surface.py
```

---

## 14. Testing Strategy

### 14.1 Correctness Tests

Every optimizer must produce **equivalent results** to the original Opytimizer when
run on CPU with the same random seed:

```python
def test_pso_correctness():
    torch.manual_seed(42)
    np.random.seed(42)

    # Run original
    orig_result = run_opytimizer_pso(seed=42)

    # Run otorchmizer
    new_result = run_otorchmizer_pso(seed=42, device="cpu")

    assert torch.allclose(
        torch.tensor(orig_result.best_position),
        new_result.best_position.cpu(),
        atol=1e-5
    )
```

### 14.2 Device Parity Tests

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pso_device_parity(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    torch.manual_seed(42)
    result_cpu = run_pso(device="cpu")

    torch.manual_seed(42)
    result_dev = run_pso(device=device)

    assert torch.allclose(result_cpu.best_fitness, result_dev.best_fitness.cpu(), atol=1e-4)
```

### 14.3 Performance Benchmarks

```python
@pytest.mark.benchmark
def test_pso_gpu_speedup(benchmark):
    result = benchmark(run_pso, n_agents=1000, n_vars=50, n_iter=500, device="cuda")
    assert result.best_fitness < TEST_EPSILON
```

### 14.4 Multi-GPU Tests

```python
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2+ GPUs")
def test_multi_gpu():
    result = run_pso(n_agents=10000, device="multi-gpu")
    assert result.best_fitness < TEST_EPSILON
```
