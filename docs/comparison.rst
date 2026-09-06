Reference comparisons
=====================

The comparison tool runs Otorchmizer, current Opytimizer, and the historical
reference in separate interpreter processes. Reference adapters are explicit
test inputs, not runtime compatibility layers in Otorchmizer.

Revisions
---------

The current-upstream review uses these immutable reference points:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Implementation
     - Version
     - Revision
   * - Current Opytimizer
     - 5.0.2
     - ``033b068e7348e3007d0efc11940abab9968cc114``
   * - Historical Opytimizer
     - 3.1.4
     - ``e050d12f3a597e41d50c234b3186262578766ca7``
   * - Otorchmizer
     - Installed source version
     - Recorded by the tool for each run

Use a clean checkout for a reproducible published comparison. The report records
source location, commit, dirty state, package version, interpreter, backend,
device, and dtype rather than inferring them from a label.

Run the tool
------------

Prepare an interpreter containing each reference's dependencies and use its
corresponding checkout. The current upstream checkout provides its own locked
uv workflow. Keep the current and historical NumPy implementations isolated;
do not replace imports inside one running Python process.

.. code-block:: powershell

   python tools\compare_implementations.py --help
   python tools\compare_implementations.py compare --help

The ``--runner`` and ``--source`` options explicitly pair each interpreter with
its checkout. For example, after choosing suitable paths:

.. code-block:: powershell

   python tools\compare_implementations.py compare `
     --runner "target=C:\work\otorchmizer\.venv\Scripts\python.exe" `
     --source "target=C:\work\otorchmizer" `
     --runner "current=C:\references\current\.venv\Scripts\python.exe" `
     --source "current=C:\references\current" `
     --runner "legacy=C:\references\legacy\.venv\Scripts\python.exe" `
     --source "legacy=C:\references\legacy" `
     --optimizer PSO --objective sphere `
     --agents 20 --variables 5 --iterations 20 --repeats 3 `
     --output comparison.json

Use the ``inventory`` command to inspect all exported optimizer names without
claiming that export equality proves mathematical equivalence. Select individual
optimizers and objectives with repeated CLI options.

For CUDA, the target interpreter must contain a CUDA-capable Torch build:

.. code-block:: powershell

   --device target=cuda --dtype target=float32

References remain CPU NumPy implementations. When dtypes differ, the conceptual
initial population is paired but storage rounding is part of the experiment.
The tool reports the actual dtype instead of claiming bitwise-identical inputs.

What is checked
---------------

Each case receives controlled initial positions, a seed, an iteration budget,
and explicit repeats. The tool records logical objective-evaluation counts,
since equal iteration budgets do not always imply equal computational work.

Required invariants include finite feasible positions, a best position that
reproduces its recorded fitness, and retention of the best observed evaluation.
Stored current-position fitness must correspond to current positions.
The references' PSO personal-best convention is identified explicitly rather
than being confused with Otorchmizer's current-position fitness.

Worker errors, invalid configurations, and required-invariant failures produce
explicit diagnostics and nonzero exit status. A reference failure is evidence
to investigate, not permission to weaken Otorchmizer's checks or patch the
reference invisibly.

Interpret results
-----------------

The execution selection is representative, not an exhaustive equation proof
for all 97 algorithms. Dedicated tests cover controlled equations, algorithm
state, numerical domains, device behavior, callbacks, and checkpoints.
Primary publications or author code resolve disputed formulas where current
and historical implementations retain the same defect.

NumPy and Torch seeds do not identify the same random stream. Equivalent
objectives and budgets can therefore yield different trajectories and outcomes.
Record and assess those differences instead of replacing assertions with a
single convergence threshold.

Optional wall times cover the worker's evaluation/update loop, excluding process
startup and imports. They are raw repeated observations, not automatic speedup
claims. The tool does not compare Python-heap tracing with native Torch or CUDA
allocator memory, and does not emit unmeasured convergence classifications.

Generated reports belong with experiment artifacts, not as permanent claims
about later repository revisions. Current usage and API documentation describe
the implementation; historical benchmark figures are not carried forward as
current assurances.
