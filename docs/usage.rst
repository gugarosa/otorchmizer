Usage and state ownership
==========================

Otorchmizer minimizes scalar objectives. A scalar callable receives one position
tensor shaped ``(n_variables, n_dimensions)`` and returns one fitness value.
The driver adapts raw callables automatically; explicit objective adapters retain
their native batching behavior.

An optimization run
-------------------

.. testcode:: usage

   import torch

   from otorchmizer import Otorchmizer
   from otorchmizer.core import Space
   from otorchmizer.optimizers.swarm import PSO

   def sphere(position):
       """Return the squared norm of one candidate.

       Args:
           position: Candidate tensor containing variables and dimensions.

       Returns:
           Scalar fitness to minimize.

       """

       return position.square().sum()

   torch.manual_seed(7)
   space = Space(8, 3, lower_bound=-2, upper_bound=2, device="cpu", dtype=torch.float64)
   space.build()
   model = Otorchmizer(space, PSO(), sphere, save_agents=True)
   assert model.start(3) is None
   assert model.total_iterations == 3
   assert space.best_position.shape == (3, 1)
   assert torch.isfinite(space.best_fitness)

The driver retains the supplied objects. Construct a fresh space and optimizer
for an independent experiment. Initializing a new population resets its current
scores and archive; it does not transfer compatible state into an existing
optimizer automatically.

The default run is quiet. Use ``model.start(..., progress=True)`` when a progress
bar is appropriate. Applications own logging configuration.

Objective batching
------------------

.. testcode:: usage

   from otorchmizer.core import Function

   batch_objective = Function(
       lambda positions: positions.square().sum(dim=(1, 2)),
       batch=True,
   )
   values = batch_objective(space.population.positions)
   assert values.shape == (8,)
   assert values.dtype == torch.float64

Scalar adapters use ``torch.vmap`` where supported. Known vectorization
limitations select a cached per-candidate path; unrelated objective exceptions
propagate. A native batch must return exactly one fitness value per candidate,
not a scalar that silently broadcasts.

Avoid mutation and invocation-count side effects inside objectives. Logical
candidate evaluations and Python callable invocations are not necessarily equal
when vectorization is active.

History and repeated runs
-------------------------

.. testcode:: usage

   best_positions, best_fitness = model.history.get_convergence("best_agent")
   positions = model.history.get_convergence("positions")
   agent_positions = model.history.get_convergence("positions", index=1)
   agent_fitness = model.history.get_convergence("fitness", index=1)
   assert best_positions.shape == (3, 3, 1)
   assert best_fitness.shape == (3,)
   assert positions.shape == (3, 8, 3, 1)
   assert agent_positions.shape == (3, 3, 1)
   assert agent_fitness.shape == (3,)

History uses native record axes rather than horizontally concatenating unrelated
dimensions. An explicit agent index selects only that axis. Population snapshots
require ``save_agents=True``; requesting an absent key raises an error.

Repeated ``start`` calls continue optimizer state and append history, while
iteration-local schedules restart. They are not generally equivalent to one
longer run. Callback sequences are supplied per invocation rather than retained
as model state.

Callbacks
---------

.. testcode:: usage

   from otorchmizer.utils.callback import Callback

   class Recorder(Callback):
       """Collect completed cumulative iteration counters."""

       def __init__(self):
           """Initialize the recorded counters."""

           self.seen = []

       def on_iteration_end(self, iteration, opt_model):
           self.seen.append(iteration)

   recorder = Recorder()
   model.start(2, callbacks=[recorder])
   assert recorder.seen == [4, 5]
   assert model.total_iterations == 5
   assert model.iteration == 1
   model.start(0)
   assert model.total_iterations == 5

The lifecycle is task begin, initial evaluation, then iteration begin, update,
clipping, evaluation, history recording, and iteration end. Task end and elapsed
time follow normal completion. Hooks run in sequence order and see live state.

``UpdateContext`` fields are immutable snapshots; referenced objects remain
mutable. The driver resolves a fresh update context after before-update hooks.
Changes to an equivalent objective wrapper belong on ``model.function``, not
``ctx.function``. Use new task objects for a mathematically different objective
rather than mixing its scores with existing history and optimizer memories.

Exceptions propagate without a general rollback guarantee. Use context managers
for external resources rather than depending on a task-end callback after failure.
Discrete projection validates finite, nonempty allowed values inside every
variable's bounds before mutation.

Trusted checkpoints
-------------------

.. warning::

   Dill loading can execute code. Load only trusted checkpoints and retain a
   compatible environment; checkpoints are not a version-independent format.

.. testcode:: usage

   from pathlib import Path
   from tempfile import TemporaryDirectory

   from otorchmizer.utils.callback import CheckpointCallback

   with TemporaryDirectory() as directory:
       path = Path(directory) / "model.pkl"
       model.start(1, callbacks=[CheckpointCallback(path, frequency=1)])
       checkpoint = path.with_name("iter_6_model.pkl")
       assert checkpoint.is_file()
       restored = Otorchmizer.load(checkpoint)
       assert restored.total_iterations == 6
       restored.start(1)
       assert restored.total_iterations == 7
       assert not path.with_name("iter_7_model.pkl").exists()

Checkpoint filename prefixes preserve their parent directory. Frequency zero
disables automatic checkpoints. Restoring a checkpoint does not recompile
optimizer buffers. Transient compiled dispatch is dropped and objective batching
is reconstructed from its original callable; enable compilation explicitly again
when appropriate.

Structured spaces and transfers
--------------------------------

GP and GSGP operate on ``TreeSpace`` expression trees. Their population tensors
are decoded phenotypes, and their best tree must reproduce the recorded best
position and fitness.

.. testcode:: usage

   from otorchmizer.optimizers.evolutionary import GP
   from otorchmizer.spaces import TreeSpace

   tree_space = TreeSpace(
       8, 2, -2, 2,
       n_terminals=3,
       functions=["SUM", "SUB", "MUL"],
       device="cpu",
       dtype=torch.float64,
   )
   tree_optimizer = GP()
   tree_model = Otorchmizer(tree_space, tree_optimizer, sphere)
   tree_model.start(2)
   torch.testing.assert_close(
       tree_space.evaluate_tree(tree_space.best_tree),
       tree_space.best_position,
   )

   tree_space.to("cpu", dtype=torch.float32)
   tree_optimizer.rebind(tree_space)
   tree_model.start(0)
   assert tree_space.population.dtype == torch.float32

Transfer the complete tree space, including terminal prototypes, generated
constants, current trees, the archived tree, population tensors, bounds, and
metadata. Moving only the population leaves inconsistent ownership and is rejected.
Rebinding after transfer invalidates compiled dispatch and resets algorithm state.
Archived genotypes survive transfer and are reevaluated before reuse.

Position-changing callbacks are unsupported for tree optimizers when they would
separate genotype and phenotype. Observational callbacks remain usable.
Exact GSGP trees grow over generations; its semantic operators are not an
implicit bounded-size or tensor-only representation.

Devices and precision
---------------------

Select ``device="cpu"``, ``"cuda:0"``, or ``"auto"`` at space construction.
An explicit ``dtype`` preserves numeric configuration before allocation rather
than attempting to recover precision after a float32 conversion.

Population scatter/gather owns its returned storage and archives. It transfers
data, not optimizer buffers or task scheduling; each shard needs appropriate
independent optimizer state.

Actual operator support and numerical range constrain reduced precision.
GPU-only tests require both suitable hardware and a CUDA-enabled Torch build.
See :doc:`algorithms` for algorithm-specific domains and :doc:`api` for exact
callable contracts.
