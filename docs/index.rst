Welcome to Otorchmizer's documentation!
========================================

Otorchmizer is a PyTorch-based nature-inspired meta-heuristic optimization framework. It is the modernized successor to `Opytimizer <https://github.com/gugarosa/opytimizer>`_, providing GPU acceleration, multi-GPU population splitting, mixed-precision, and ``torch.compile`` JIT integration.

Use Otorchmizer if you need a library or wish to:

* Create your own optimization algorithm with GPU support;
* Design or use pre-loaded optimization tasks;
* Use batched tensor storage and measure performance on your own workload;
* Because it is fun to optimize things.

Otorchmizer is compatible with: **Python 3.10+** and **PyTorch 2.0+**.

GPU, reduced-precision, compilation, and CUDA Graph compatibility depend on the
chosen algorithm, objective, operations, and hardware. Historical benchmark numbers
are not guarantees for all optimizer classes.

.. warning::

    Some optimizer exports differ materially from the algorithms they were
    intended to migrate. Read :doc:`algorithm_limits` before treating matching
    class names or convergence smoke tests as evidence of algorithmic fidelity.

.. toctree::
    :maxdepth: 2
    :caption: Package Reference

    algorithm_limits
    api/otorchmizer
    api/otorchmizer.core
    api/otorchmizer.functions
    api/otorchmizer.math
    api/otorchmizer.optimizers
    api/otorchmizer.spaces
    api/otorchmizer.utils
    api/otorchmizer.visualization

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
