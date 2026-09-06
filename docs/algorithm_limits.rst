Algorithm migration limits
==========================

The optimizer exports are not a blanket guarantee of equivalence to
Opytimizer 3.1.4. The February 2026 migration introduced substantive variants,
omitted mechanics, and parameter differences that ordinary convergence smoke
tests did not detect.

The current engineering work repairs demonstrated tensor, state-tracking,
selection, and equation defects where the intended behavior is well supported.
It does not silently replace a published algorithm with a different algorithm
or rename its public parameters.

Identity mismatches
-------------------

These exports need a separate compatibility decision: restore the intended
migration target under the existing name, or retain and explicitly name the
implemented variant. Until then, do not cite them as the corresponding
Opytimizer algorithm merely because their abbreviations match.

.. list-table::
   :header-rows: 1
   :widths: 12 44 44

   * - Export
     - Current implementation
     - Opytimizer migration target
   * - ``ABO``
     - African Buffalo-style update
     - Artificial Butterfly Optimization
   * - ``FSO``
     - Fish School-style update
     - Flying Squirrel Optimizer
   * - ``SSO``
     - Social Spider-style update
     - Simplified Swarm Optimization

Incomplete or materially different variants
--------------------------------------------

The following findings were identified by inspecting the implementations and
their migration source. They are not resolved by formatting or by reaching a
low value on a sphere objective.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Export
     - Remaining migration concern
   * - ``ASO``
     - The defining Lennard-Jones force is absent and ``alpha`` does not affect the update.
   * - ``HGSO``
     - Cluster handling, cluster-best attraction, and the Henry coefficient schedule differ.
   * - ``TWO``
     - Friction parameters and directional displacement mechanics are incomplete.
   * - ``QSA``
     - Queue allocation differs and business phases two and three are absent.
   * - ``AO``
     - Strategy equations and upstream ``n_cycles``, ``U``, and ``w`` parameters need reconciliation.
   * - ``FOA``
     - Global seeding and the meaning of ``area_limit`` under fixed population storage need reconciliation.
   * - ``PIO``
     - The current schedule does not implement the upstream ``n_c1``/``n_c2`` thresholds.
   * - ``SFO``
     - Prey consumption and replenishment policies conflict across references; current top-N selection is a partial variant.
   * - ``CDO``, ``EFO``, ``ESA``, ``LSA``, ``TEO``, ``WEO``
     - Several defining equations or update phases differ from the referenced migration implementations.

RRA and PPA variant differences, and EP strategy inheritance, require further
paper-level confirmation before a mathematical rewrite. These are unresolved
hypotheses rather than claims that every difference is a defect.

Parameter compatibility
-----------------------

The parameter dictionary supports custom optimizer attributes. Consequently,
an accepted key does not prove that a particular algorithm reads it.
These migration-name differences remain important:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Export
     - Migration-source name
     - Current implementation name
   * - ``WDO``, ``SSD``
     - ``c``
     - ``c_val``
   * - ``UMDA``
     - ``lower_bound``, ``upper_bound``
     - ``lower_bound_prob``, ``upper_bound_prob``
   * - ``TWO``
     - ``alpha``, ``beta``
     - ``alpha_val``, ``beta_val``

Device and precision scope
--------------------------

Tensor storage does not make every operation equally supported on every
device or dtype. For example, PyTorch 2.0 CPU kernels do not implement some
half-precision clamp and exponential operations, and half-precision distance
operations may also be unavailable. Float32/float64 algorithm contracts and
the core tensor fixes were exercised on the minimum dependency stack.
Broader CPU float16 update coverage was exercised on PyTorch 2.14, not claimed
for PyTorch 2.0. No tests were weakened to hide these backend limitations.
The local review did not execute on real CUDA or multi-GPU hardware.

Compilation and CUDA Graph capture also depend on the actual control flow
and tensor-storage behavior. They are optional execution tools, not
guaranteed speedups or proof of algorithmic equivalence.
