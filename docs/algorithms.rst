Algorithm contracts
===================

Otorchmizer exposes 97 optimizer classes through canonical family modules.
The API pages contain constructor parameters and scientific references.
The name of an algorithm is not a guarantee of identical trajectories across
implementations, random generators, numerical precisions, or update ordering.

Reference policy
----------------

The current NumPy reference is
`gugarosa/opytimizer 5.0.2 <https://github.com/gugarosa/opytimizer/tree/033b068e7348e3007d0efc11940abab9968cc114>`_.
The 3.1.4 revision is also used in explicit comparison runs to distinguish
historical behavior from later changes. Neither reference's passing tests
override an independently demonstrated mathematical or state defect.

Where the Python references disagree with an author's implementation or a required
invariant, the correction is documented rather than hidden behind compatibility
aliases. Runtime code does not select old behavior from a version flag.

Identities and structure
------------------------

``ABO`` is Artificial Butterfly Optimization, ``FSO`` is Flying Squirrel
Optimization, ``SSO`` is Simplified Swarm Optimization, and ``WAOA`` is Walrus
Optimization. Their parameters and update phases belong to those algorithms.

Each algorithm has one implementation module. Import either its public class
from the family or its canonical module directly:

.. code-block:: python

   from otorchmizer.optimizers.science import HGSO
   from otorchmizer.optimizers.science.hgso import HGSO

The two imports identify the same class, not separate implementations.

Tensor state and archives
-------------------------

Current positions and fitness remain paired. Personal memories, velocities,
mutation strategies, family assignments, and other state must follow the same
individual whenever populations are sorted or resampled.

Temporary proposals can be valid improvements even when an algorithm does not
retain them in its next population. The best-result archive is updated before
selection, restart, separation, or replacement discards such proposals.
This does not impose elitist population selection on every algorithm.

Population size is algorithm-dependent. FOA limits its current forest before
global seeding and can then grow it. Position, fitness, age, and ``n_agents``
values remain synchronized. MOA requires a square population for its toroidal
grid; algorithms needing distinct peers validate their minimum cardinality.

Numerical and scientific choices
--------------------------------

ASO uses a component-wise K-best centroid and applies gravitational decay and
inverse mass to both interatomic and best-position forces. These choices follow
the author's ``Acceleration.m`` in the
`official ASO distribution <https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/67011/versions/1/download/zip/ASO.zip>`_.
The input fitness must be finite for mass calculation.

EP follows the Gaussian CEP form of Yao, Liu, and Lin's
`primary paper, page 83 <https://www.cse.unr.edu/~sushil/class/gas/papers/EPMadeFaster.pdf#page=2>`_:
displacement uses the parent strategy, while children inherit separately adapted
log-normal strategies. It is not the paper's Cauchy FEP variant.

FOA's age cutoff and global-seeding overwrite follow the
`authors' implementation <https://github.com/cominsys/FOA>`_.
Trees with ages below ``life_time`` survive the age filter, and each selected
global candidate produces an independent tree. Python ``transfer_rate`` is a
fraction, not the MATLAB percentage.

COA retains the reduced pack-based variant provided by the current NumPy reference:
alpha leadership, cultural tendency, greedy movement, and probabilistic exchange.
It does not implement pup birth or coyote ages from the
`author's complete COA model <https://github.com/jkpir/COA>`_.
The exchange probability is ``0.005 * n_agents``, not the author's squared
per-pack rule. These mechanics form one explicit variant rather than a mixture
of selected phases from different implementations.

CEM adapts every variable/dimension and measures elite spread around the elite
mean. ``alpha`` is a non-negative previous-state weight; values above one
extrapolate rather than form a convex average. Elite counts above population
size use the available population.

HGSO, TEO, and WWO require finite, non-negative fitness. Unsupported signed values
are rejected rather than transformed into superficially finite results.
TEO uses zero cooling rates for all-zero scores and unit rates for equal positive
scores. Its environmental temperatures follow current ranked pairs; an odd
population's median pairs with itself.

WWO uses variable-domain widths for propagation and random distinct breaking
coordinates. Positive refraction ratios preserve their scale; exact zero ratios
retain the wavelength. Equal-score populations receive one ``alpha^-1`` reduction.
Complete scaled ratios avoid overflowing intermediate quotients, including on
the minimum Torch backend. An unrepresentable positive wavelength raises an error.

PIO's phase thresholds are absolute iteration counts. It stops moving pigeons
after ``n_c2``. The landmark target retains the reference's active-population
divisor; exact zero-sum weights use ``active_mean / n_p``. Scaling finite positive
fitness does not change the weighted target.

SFO consumes promoted sardines and uniformly replenishes the configured prey
population. This is an explicit bounded-replenishment policy, not a prey-exhaustion
variant. Evaluated prey improvements enter the result archive independently of
the current elite sailfish used by the hunting equations.

The Levy sampler's ``beta`` is a stability exponent. At ``beta=2`` it uses the
Gaussian limit with standard deviation ``sqrt(2)`` rather than a degenerate
Mantegna numerator. Random sampling happens on the requested device and dtype;
omitting a floating dtype follows PyTorch's default.

Structured and multi-objective search
-------------------------------------

GP and GSGP use expression trees whose evaluated phenotypes populate the space.
Invalid programs are not converted into favorable midpoint-valued candidates.
Tree-space transfers move all owned tensors and retain archived genotypes for
reevaluation. Position-changing callbacks are rejected when they would separate
genotypes from scored phenotypes.

Exact GSGP expressions grow over generations. GP's pruning-point selection
parameter is not a general depth or memory limit, and GSGP does not silently
accept it as growth control. Python tree operations are not a tensor-only
compiled kernel.

LOA uses batched pride, sex, group, personal-memory, and nomad state rather than
per-agent ``Lion`` compatibility objects.

NDS ranks precomputed objective vectors without scalarization. Its default
orientation is maximization; select ``maximize=False`` for minimization.
Rank zero is nondominated, duplicates share a rank, and reevaluation resets
domination counts. Its best position is a current first-front representative,
not a historical total ordering of multi-objective results.
The explicit domination matrix requires quadratic storage.

``NBJS`` is a source-library Jellyfish Search variant with bound-independent
passive motion. It is not presented as a separately peer-reviewed algorithm.

Execution scope
---------------

Correctness checks include controlled equations, state and archive invariants,
all-export CPU/CUDA execution in float32 and float64, and explicit current/reference
comparisons. Equal seed integers do not imply equal random streams across NumPy
and PyTorch.

Reduced precision has backend and range limits. PyTorch 2.0 CPU lacks float16
kernels for operations including tensor-bounded clipping, exponentials, and
multinomial sampling. Use float32 or float64 on that backend. Unbounded random
steps can also exceed a dtype's range.
Compilation and CUDA Graph capture depend on control flow and retained storage;
dynamic forests and Python genomes are not universally graph-capturable.

No application-wide speedup is inferred from an execution test or a representative
comparison. Performance claims require equivalent workloads and repeated measurements
on the actual target environment.
