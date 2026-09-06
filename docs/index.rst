Otorchmizer documentation
==========================

Otorchmizer provides optimization and search algorithms using PyTorch population
tensors. It builds on `Opytimizer <https://github.com/gugarosa/opytimizer>`_ while
using explicit ownership, objective batching, and device-aware execution.

The package supports **Python 3.10+** and **PyTorch 2.0+**.

GPU, reduced-precision, compilation, and CUDA Graph compatibility depend on the
chosen algorithm, objective, operations, and hardware. Read :doc:`algorithms`
for scientific references, parameter semantics, numerical domains, and intentional
implementation choices. Matching class names or random seeds do not establish
equivalent stochastic trajectories.

.. toctree::
    :maxdepth: 2
    :caption: Package Reference

    usage
    algorithms
    comparison
    api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
