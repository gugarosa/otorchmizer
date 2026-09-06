# Otorchmizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Otorchmizer provides **97 optimization and search algorithms** with PyTorch-backed
populations, objective batching, and CPU/CUDA execution.
It builds on [Opytimizer](https://github.com/gugarosa/opytimizer) while using an
explicit tensor-based execution model rather than emulating NumPy agent objects.

The library includes continuous, Boolean, grid, hyper-complex, Pareto, and
expression-tree search spaces. Algorithm state, numerical domains, and ownership
are part of the API, not implied guarantees from a class name.

## Install

```bash
pip install otorchmizer
```

For CUDA, select a PyTorch build compatible with the installed driver. With uv,
this also replaces an existing CPU-only Torch wheel:

```bash
uv pip install --torch-backend auto --reinstall-package torch otorchmizer
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For pip-specific CUDA commands, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
Having a GPU does not make a CPU-only PyTorch wheel CUDA-capable.

## Optimize an objective

```python
import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Space
from otorchmizer.optimizers.swarm import PSO


def sphere(position: torch.Tensor) -> torch.Tensor:
    """Return the squared norm of one candidate.

    Args:
        position: Candidate tensor with variables and dimensions.

    Returns:
        Scalar fitness to minimize.

    """

    return position.square().sum()


space = Space(
    n_agents=20,
    n_variables=4,
    lower_bound=-5,
    upper_bound=5,
    device="auto",
)
space.build()
model = Otorchmizer(space, PSO(), sphere)
model.start(n_iterations=100)

print(space.best_position)
print(space.best_fitness.item())
```

Population positions have shape `(n_agents, n_variables, n_dimensions)`.
A scalar objective receives one `(n_variables, n_dimensions)` candidate.
Native-batch objectives receive the whole population and return `(n_agents,)`.
Known vmap limitations use a documented per-agent evaluation path; unrelated
objective errors propagate.

The supplied space and optimizer remain mutable and are not copied.
Construct separate instances for independent tasks. Repeated runs continue
algorithm state while restarting iteration-local schedules.

## Algorithms

| Family | Count | Exports |
|---|---:|---|
| Swarm | 34 | ABC, ABO, AF, AIWPSO, BA, BOA, BWO, CS, CSA, EHO, FA, FFOA, FPA, FSO, GOA, JS, KH, MFO, MRFO, NBJS, PIO, PSO, RPSO, SAVPSO, SBO, SCA, SFO, SOS, SSA, SSO, STOA, VPSO, WAOA, WOA |
| Evolutionary | 16 | BSA, DE, EP, ES, FOA, GA, GHS, GOGHS, GP, GSGP, HS, IHS, IWO, NGHS, RRA, SGHS |
| Miscellaneous | 6 | AOA, CEM, DOA, GS, HC, NDS |
| Population | 12 | AEO, AO, COA, EPO, GCO, GWO, HHO, LOA, OSA, PPA, PVS, RFO |
| Science | 20 | AIG, ASO, BH, CDO, EFO, EO, ESA, GSA, HGSO, LSA, MOA, MVO, SA, SMA, TEO, TWO, WCA, WDO, WEO, WWO |
| Social | 6 | BSO, CI, ISA, MVPA, QSA, SSD |
| Boolean | 3 | BMRFO, BPSO, UMDA |

GP/GSGP use `TreeSpace` genomes; LOA owns lion demographics and personal memories;
NDS ranks precomputed objective vectors without scalarizing them.
Read [algorithm contracts](docs/algorithms.rst) for parameter units,
supported fitness domains, numerical policies, and intentional differences from reference implementations.

## State, precision, and devices

The result archive preserves evaluated improvements even when an algorithm's live
population discards them. Current fitness and personal-best memories are separate.
Some algorithms, including FOA, change population size during execution.

Device and dtype changes must include all owned state. Use `TreeSpace.to(...)`
for expression-tree spaces and rebind an existing optimizer before continuing.
Moving only population tensors does not transfer unrelated optimizer buffers or tree terminals.

CUDA, reduced precision, `torch.compile`, and CUDA Graph capture depend on the
actual algorithm, objective, operations, and hardware. Tree topology and changing
population sizes are not universally capturable. Scatter/gather transfers data;
it does not schedule independent optimization jobs across GPUs.

Only load trusted dill checkpoints. They can execute code and are not a
version-independent interchange format. Original objectives are retained while
transient vectorization wrappers are reconstructed after loading.

## Documentation

| Guide | Content |
|---|---|
| [Architecture](ARCHITECTURE.md) | Implemented responsibilities, data shapes, lifecycle, and ownership |
| [Usage](docs/usage.rst) | Objective adaptation, device setup, results, and structured spaces |
| [Algorithm contracts](docs/algorithms.rst) | Scientific references, parameter semantics, domains, and numerical policies |
| [Reference comparisons](docs/comparison.rst) | Isolated current/historical reference runs and evidence interpretation |
| [API reference](docs/index.rst) | Public modules and callable contracts |
| [Examples](examples/) | Executable core, optimizer, and application examples |
| [Conventions](CONVENTIONS.md) | Contribution rules, documentation style, and validation standards |

## Develop and validate

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -e ".[dev,docs]"
uv run --no-sync pytest
uv run --no-sync pre-commit run --all-files
uv run --no-sync sphinx-build -W --keep-going -b html docs docs/_build/html
uv run --no-sync sphinx-build -W --keep-going -b doctest docs docs/_build/doctest
uv build
```

The CPU backend is a development-environment choice, not a library restriction.
The test suite includes deterministic equations, state and lifecycle invariants,
checkpoint behavior, artifact installation, and optional real-CUDA execution.
GPU-only tests require a CUDA-capable PyTorch build.

Compare equivalent objectives, initial configurations, evaluation budgets, and
repeated measurements. Matching integer seeds across NumPy and PyTorch does not
produce identical random streams. No universal speedup, convergence result,
or numerical equivalence is inferred from an inventory or a smoke test.

## Release

Keep package metadata and `otorchmizer.__version__` aligned. A GitHub release tagged
`v<version>` runs the interpreter, style, documentation, and installed-wheel gates
before publishing matching artifacts. Publishing requires the repository's
configured PyPI authentication.

## Citation

Please cite the original Opytimizer work when using this library in research:

```bibtex
@misc{rosa2019opytimizer,
    title={Opytimizer: A Nature-Inspired Python Optimizer},
    author={Gustavo H. de Rosa, Douglas Rodrigues and João P. Papa},
    year={2019},
    eprint={1912.13002},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
