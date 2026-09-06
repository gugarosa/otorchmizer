Migration contracts and limits
==============================

Version 2.0 completes the optimizer inventory against
`Opytimizer 3.1.4 <https://github.com/gugarosa/opytimizer/tree/v3.1.4>`_,
revision ``e050d12f3a597e41d50c234b3186262578766ca7``.
All 97 optimizer classes are exported. This is an inventory and implementation
contract, not a claim of identical random trajectories or exhaustive mathematical
assurance for every workload.

The tensor-based ``Population`` and ``UpdateContext`` interfaces remain the
Otorchmizer execution model. Reference defects are not preserved merely to obtain
matching outputs, and consequential compatibility changes are versioned explicitly.

New optimizer migrations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Export
     - Execution contract
   * - ``GP``
     - Genetic programming with expression trees, reproduction, tournament selection, crossover, and mutation.
   * - ``GSGP``
     - Geometric semantic crossover and mutation over actual expression trees.
   * - ``LOA``
     - Lion prides and nomads, hunting, movement, mating, defense, migration, and population equilibrium.
   * - ``NDS``
     - Nondominated sorting of precomputed objective vectors without scalarization.
   * - ``NBJS``
     - The source library's Jellyfish Search variant with bound-independent passive motion.
   * - ``WWO``
     - Water Wave Optimization with propagation, breaking, refraction, height, and wavelength adaptation.

``NBJS`` is a source-library variant whose reference marks its publication as
pending. It is not presented as a separately peer-reviewed algorithm.
``GraphSpace`` remains an experimental descriptor: the migration source also
does not contain a working graph-search implementation.

Breaking changes from 1.x
--------------------------

The previous ``ABO``, ``FSO``, and ``SSO`` exports represented unrelated algorithm
identities. Version 2.0 restores the intended targets rather than retaining
mislabeled implementations behind aliases.

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Export
     - Previous surface
     - Version 2.0 surface
   * - ``ABO``
     - African Buffalo-style update
     - Artificial Butterfly Optimization with ``sunspot_ratio`` and ``a``
   * - ``FSO``
     - Fish School-style update
     - Flying Squirrel Optimizer with ``beta``
   * - ``SSO``
     - Social Spider-style update
     - Simplified Swarm Optimization with ``C_w``, ``C_p``, and ``C_g``
   * - ``PIO``
     - ``n_c`` and a normalized phase split
     - Absolute ``n_c1`` and ``n_c2`` iteration thresholds, with ``R``
   * - ``WDO``, ``SSD``
     - ``c_val``
     - ``c``
   * - ``UMDA``
     - ``lower_bound_prob``, ``upper_bound_prob``
     - ``lower_bound``, ``upper_bound``
   * - ``TWO``
     - ``alpha_val``, ``beta_val``
     - ``alpha``, ``beta``
   * - ``TEO``
     - Numeric switches
     - Boolean ``c1`` and ``c2`` switches
   * - ``FOA``
     - Fixed population storage
     - A dynamic forest whose position, fitness, and age counts remain synchronized
   * - ``HGSO``, ``TEO``, ``WWO``
     - Unenforced fitness-domain assumptions
     - Finite, non-negative fitness with explicit zero-score policies

The parameter dictionary still supports custom attributes. Acceptance of an
arbitrary key is not proof that the algorithm reads it. Update saved parameter
dictionaries using the table above rather than relying on old spellings.

``PIO`` does not move pigeons after ``n_c2``. Set its absolute thresholds to suit
the requested iteration budget. ``FOA`` applies its area limit before global
seeding, so the forest can temporarily exceed that limit after new seeds are
introduced. Consumers must read the current ``population.n_agents`` rather than
cache the initial count.

Restored mechanics
------------------

The migration restores the documented partial implementations, including:

* ASO's Lennard-Jones interactions, vector-valued centroid, and complete
  inverse-mass and gravitational scaling
* AO's four strategies and ``n_cycles``, ``U``, and ``w`` spiral parameters
* EP strategy inheritance, RRA stalling searches and restart, and FOA global seeding
* PPA's crow, cuckoo, and cat phases and correct cuckoo-pool selection
* QSA's reciprocal-fitness queue allocation and all three business phases
* HGSO's cluster-specific state, cluster leaders, and Henry coefficient schedule
* CDO, EFO, ESA, LSA, TEO, TWO, and WEO equations and omitted update phases

The restored science optimizers use these parameter defaults in their ``params``
dictionaries. State such as EFO's ``RI`` can subsequently change during updates.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Export
     - Defaults
   * - ``CDO``
     - No algorithm-specific parameters
   * - ``EFO``
     - ``positive_field=0.1``, ``negative_field=0.5``, ``ps_ratio=0.1``, ``r_ratio=0.4``,
       ``phi=(1+sqrt(5))/2``, ``RI=0``
   * - ``ESA``
     - ``n_electrons=5``
   * - ``HGSO``
     - ``n_clusters=2``, ``l1=0.0005``, ``l2=100``, ``l3=0.001``, ``alpha=1``, ``beta=1``, ``K=1``
   * - ``LSA``
     - ``max_time=10``, ``E=2.05``, ``p_fork=0.01``
   * - ``TEO``
     - ``c1=True``, ``c2=True``, ``pro=0.05``, ``n_TM=4``
   * - ``TWO``
     - ``mu_s=1``, ``mu_k=1``, ``delta_t=1``, ``alpha=0.9``, ``beta=0.05``
   * - ``WEO``
     - ``E_min=-3.5``, ``E_max=-0.5``, ``theta_min=-pi/3.6``, ``theta_max=-pi/9``
   * - ``WWO``
     - ``h_max=5``, ``alpha=1.001``, ``beta=0.001``, ``k_max=1``

Batching can change the sequential update order of the reference, and CPU/CUDA
random streams need not match. Compare objective-evaluation budgets, solution
quality, and algorithm invariants rather than expecting bit-for-bit trajectories.

Explicit numerical and state policies
-------------------------------------

ASO uses a coordinate-wise centroid rather than averaging unrelated coordinates
into a scalar. Both interatomic and best-position forces receive the gravitational
decay and inverse-mass factor. QSA and ASO require finite fitness for their
weighting equations rather than silently converting invalid scores into defaults.

HGSO, TEO, and WWO enforce finite, non-negative fitness instead of inventing
signed-fitness extensions from absolute values or epsilon substitutions.
TEO uses zero cooling rates for all-zero scores and unit rates for equal positive
scores. WWO preserves a wavelength when its refraction ratio has an exact zero
numerator or denominator; valid positive ratios are not floored by machine epsilon.
An equal-fitness wave population receives one ``alpha^-1`` wavelength reduction.
Arbitrary objective shifts can change algorithms that use raw fitness ratios,
even when the location of the mathematical optimum is unchanged.

EP implements the Gaussian CEP equations in Yao, Liu, and Lin's
`primary paper, page 83 <https://www.cse.unr.edu/~sushil/class/gas/papers/EPMadeFaster.pdf#page=2>`_.
Child displacement uses the parent's strategy, while the child inherits a
separately adapted log-normal strategy. This corrects the source's additive
adaptation and inheritance mismatch without substituting Cauchy FEP mutation.
The result archive retains evaluated improvements even when tournament selection
does not retain that individual.

The shared Levy sampler's ``beta`` is a stability exponent, not a skewness
parameter. Its ``beta=2`` endpoint is Gaussian with standard deviation
``sqrt(2)`` rather than the degenerate numerator of the Mantegna ratio formula.
Sampling uses the requested device and dtype without first drawing in float32
and casting the result. Omitting ``dtype`` preserves the PyTorch default.

SFO uses an explicit bounded-replenishment policy: promoted sardines are consumed
and uniformly replaced, maintaining the configured prey population size.
This differs from variants that progressively exhaust the prey population.
The result archive retains evaluated prey bests independently of the current
elite sailfish used by the hunting equations.

FOA follows the original authors' MATLAB behavior where it differs from the Python
migration: ``PopulationLimiting.m`` retains ages strictly below ``life_time``,
and ``GlobalSeeding.m`` overwrites selected coordinates and creates one independent
tree per selected candidate. This avoids the Python source's late age cutoff and
repeated references to a mutable seed. The Python ``transfer_rate`` remains a
fraction in ``[0, 1]`` rather than the MATLAB percentage.
See the `authors' implementation <https://github.com/cominsys/FOA>`_.

PIO normalizes finite active fitness before computing its landmark target, so
scaling positive scores does not change that target. Its exact zero-sum policy
uses ``active_mean / n_p``, consistent with equal positive weights.
NaN and infinite active scores are rejected rather than hidden by this policy.

NDS defaults to maximization, matching the source orientation. Use
``NDS({"maximize": False})`` for minimization. Front rank zero is nondominated,
duplicates share a rank, and repeated evaluations reset domination counts.
The stored fitness values are ranks, not scalarized objective values.
``best_position`` is a representative of the current first front, not a historical
total ordering of multi-objective solutions. Its explicit domination matrix
requires quadratic storage in the number of points.

Structured search spaces
-------------------------

GP and GSGP operate on ``TreeSpace`` genomes, and positions are their evaluated
phenotypes rather than unrelated random samples. ``Optimizer.bind(space)``
provides the space-level initialization boundary while preserving the existing
``compile(population)`` and ``evaluate(population, function)`` interfaces.

The shared defaults are ``p_reproduction=0.25``, ``p_mutation=0.1``,
``p_crossover=0.1``, and ``prunning_ratio=0.0``. The ``prunning_ratio`` spelling
is retained from the reference. In GP it restricts eligible subtree operation
points, not overall tree size or depth. GSGP uses whole-tree geometric semantic
operators, requires this setting to remain zero, and adds ``mutation_step=0.1``.

Position-transforming callbacks cannot silently change the phenotype while
leaving its saved expression tree unchanged. GP/GSGP reject this unsupported
combination; lifecycle and read-only callbacks remain usable.

Tree topology and terminal tensors have separate ownership from the population
tensor. Device or dtype changes must move the complete tree-space state rather
than only calling ``population.to(...)``. Use ``TreeSpace.to(device, dtype)``
before constructing the optimization engine. Mismatches are errors, not implicit
output casts. Transfer invalidates scores and retains the historical best tree
for reevaluation, even when it is absent from the current population.

After transferring an already-bound space, call ``optimizer.rebind(space)`` before
continuing. Rebinding clears compiled dispatch and recompiles optimizer state.
Compilation must be requested again where the chosen optimizer supports it.

Exact GSGP expressions grow over generations. This is a real memory and traversal
cost, not hidden behind a claim of constant-size genomes. Python tree mutation and
traversal are not a ``torch.compile``-compatible tensor kernel.

Device and precision scope
--------------------------

The regression matrix includes CPU and CUDA execution in float32 and float64.
GPU execution requires both suitable hardware and a CUDA-enabled PyTorch build:
a CPU-only wheel reports CUDA as unavailable even when a GPU is installed.

Backend support and numerical range also constrain reduced precision. PyTorch
2.0 CPU lacks several half-precision kernels, and unbounded Levy steps can exceed
the range of a reduced-precision dtype. No universal float16 or bfloat16 guarantee
is made for every algorithm, objective, and backend combination.

The migration work exercises an NVIDIA RTX 4070 with PyTorch 2.14 and CUDA 13.2,
including native random sampling, graph replay of a compatible in-place kernel,
and CPU/CUDA population transfers. One GPU does not establish real multi-GPU
execution coverage.

Compilation and CUDA Graph capture depend on control flow and tensor-storage
behavior. Dynamic forests and Python tree genomes do not become graph-capturable
merely because their numerical values are stored in tensors. No application-wide
speedup is inferred from these execution checks.
