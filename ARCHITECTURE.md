# Otorchmizer architecture

Otorchmizer minimizes objectives using PyTorch-backed search populations.
This document describes the implemented system, not a migration plan or a performance target.

## Responsibilities

| Component | Responsibility |
|---|---|
| `Space` | Own population initialization, bounds, device selection, and space-specific behavior |
| `Population` | Store candidate tensors and the best evaluated result |
| `Function` | Adapt scalar or native-batch objectives to one fitness value per candidate |
| `Optimizer` | Own algorithm state, position updates, and any specialized evaluation |
| `UpdateContext` | Pass the current space, objective, iteration, budget, and device explicitly |
| `Otorchmizer` | Coordinate construction, callbacks, updates, evaluation, history, and checkpoints |

The driver retains its space and optimizer rather than copying them.
Use separate instances for independent experiments. Tensor storage does not provide
thread isolation for shared model instances or random-number generators.

## Population representation

| Tensor | Shape |
|---|---|
| `positions` | `(n_agents, n_variables, n_dimensions)` |
| `fitness` | `(n_agents,)` |
| `best_position` | `(n_variables, n_dimensions)` |
| `best_fitness` | Scalar |
| `lb`, `ub` | `(n_variables, 1)` or `(n_variables, n_dimensions)` |

Unscored fitness starts at positive infinity in the population dtype.
Current fitness describes current positions; algorithms retain personal memories
in separate state arrays. The result archive can contain a previously evaluated
candidate that is no longer in the live population.

Slicing can share storage. Clone tensors when an algorithm requires an independent
snapshot, such as an evaluated leader that must remain unchanged while followers move.
Do not reorder positions without applying the same permutation to fitness and
associated per-candidate state.

Population size is not universally fixed. FOA grows and limits its forest, keeping
positions, fitness, ages, and `n_agents` synchronized. Consumers must inspect the
current shape rather than assume the initial count remains valid.

## Objective evaluation

`Function(pointer)` accepts a scalar objective operating on
`(n_variables, n_dimensions)` and uses `torch.vmap` for population evaluation.
Known unsupported vmap operations select a diagnostic, cached per-candidate path.
Other user exceptions propagate without being retried.

`Function(pointer, batch=True)` accepts a population tensor directly. The result
must have shape `(n_agents,)`; a scalar is not broadcast into a plausible batch.
Objectives should not mutate their inputs or rely on invocation side effects.
Vectorization, fallback, and native batching can produce different Python call counts.

Constraint and multi-objective helpers compose the same evaluation boundary.
Scalar optimization, weighted objectives, and precomputed Pareto vectors are
different contracts; NDS ranks objective vectors without scalarizing them.

Checkpoints retain the original objective and batching configuration. Backend-specific
vmap closures are reconstructed on load rather than serialized as durable state.

## Optimizer lifecycle

Construction binds the space and compiles population-dependent optimizer state.
The shared interfaces are:

```python
optimizer.bind(space)
optimizer.compile(space.population)
optimizer.evaluate(space.population, function)
optimizer(context)
```

Calling the optimizer dispatches to its compiled update when configured.
Calling `optimizer.update(context)` directly selects the eager implementation.
`torch_compile()` is optional and does not promise a speedup.

An update may evaluate temporary proposals before deciding which candidates survive.
Those evaluations must retain a valid improvement in the result archive even when
selection or restart subsequently discards the proposal. Archival does not require
an algorithm's live population to be elitist.

Concrete methods retain meaningful algorithm phases and data dependencies.
A Python loop can be necessary for sequential acceptance, tree manipulation,
or coupled population operations. Replacing such a loop with a simultaneous
update is a behavioral change, not automatically an optimization.

## Driver, callbacks, and history

Each run performs task-begin callbacks and initial evaluation. Each iteration then
dispatches iteration-begin callbacks, updates, clips to the space bounds, evaluates,
records history, and dispatches iteration-end callbacks. Task-end callbacks and
elapsed-time recording occur after normal completion.

Per-run iteration counters restart; cumulative counters and algorithm state continue.
Repeated runs can therefore differ from one longer run when schedules depend on the
local iteration budget. Construction or explicit rebinding can reset optimizer state.

Callbacks receive live objects. Their ordering and mutation effects matter.
Exceptions propagate; task-end callbacks are not a substitute for resource-owning
context managers. The driver does not promise transactional rollback.

History stores independent population/result snapshots where documented.
Collecting every candidate at every iteration has a real memory and transfer cost;
enable it when the experiment needs those records.

## Structured search

GP and GSGP use Python expression trees with tensor-valued terminals.
Tree phenotypes are evaluated programs, not independent random population positions.
`best_tree`, `best_position`, and `best_fitness` must describe the same evaluated result.

Protected primitives handle their documented domains; residual non-finite programs
are not replaced with favorable midpoint values. Position-transforming callbacks
are rejected for tree optimizers because they would separate a saved genotype from
its scored phenotype. Observational callbacks remain usable.

`TreeSpace.to(device, dtype)` transfers terminal prototypes, current trees, generated
constants, the archived best tree, population tensors, bounds, and metadata together.
It invalidates scores and retains the historical best for reevaluation. Transfer
before creating the driver when possible. After a bound-space transfer, use
`optimizer.rebind(space)` before continuing; it clears compiled dispatch and recompiles state.

Exact GSGP expressions grow over generations. Neither a pruning-point selection
ratio nor tensor storage makes them bounded-size genomes or a tensor-only compiled kernel.

## Devices and numerical precision

`DeviceManager` resolves the requested device and supplies device-aware tensor factories.
Optimizers allocate state from the population's device, dtype, and dimensions.
Use native random sampling in the intended dtype instead of sampling elsewhere and
casting away precision afterward.

Scattering and gathering populations are data-transfer operations, not a parallel
optimization scheduler. Each shard needs independent compatible optimizer state.
A single-GPU test cannot establish real multi-GPU execution behavior.

Autocast follows PyTorch's operator-specific policy. Reduced-precision storage
does not guarantee that every operation, intermediate result, or objective fits its
range. Some minimum-version CPU kernels do not support half precision.
Scientific algorithms may also impose explicit fitness-domain requirements.

CUDA Graph capture requires compatible fixed control flow and retained storage.
Warmup and capture invoke the supplied callable. Tree topology and changing
population sizes must not be presented as universally graph-capturable workloads.

## Verification and delivery

Correctness evidence combines controlled equation tests, state invariants, lifecycle
and checkpoint integration, reference comparisons, and built-artifact execution.
Matching names, matching seeds across different random generators, or a low sphere
value do not establish algorithm equivalence.

Performance comparisons must record the actual implementation revision, environment,
objective, shape, dtype, evaluation budget, and repeated measurements. The repository
does not prescribe universal speedup targets or infer allocator memory from Python-heap tracing.

`pyproject.toml` owns package metadata and tool configuration. Existing Ruff, pytest,
and Sphinx checks run in CI, alongside wheel and source-distribution builds.
Publication requires a matching version tag and successful release validation.
Only load trusted dill checkpoints; they are not a version-independent interchange format.

For contributor rules, see [CONVENTIONS.md](CONVENTIONS.md).
Algorithm-specific numerical and state contracts are documented in
[algorithm contracts](docs/algorithms.rst), with callable details in the API reference.
