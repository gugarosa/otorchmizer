# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""CPU and CUDA execution contracts for every public optimizer export."""

import pytest
import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers import boolean, evolutionary, misc, population, science, social, swarm
from otorchmizer.spaces import ParetoSpace, TreeSpace

_OPTIMIZERS = [
    getattr(family, name)
    for family in (boolean, evolutionary, misc, population, science, social, swarm)
    for name in family.__all__
]


def _sphere(position):
    return position.square().sum()


def test_export_inventory_matches_the_reference():
    names = [optimizer.__name__ for optimizer in _OPTIMIZERS]

    assert len(names) == len(set(names)) == 97
    assert {"GP", "GSGP", "LOA", "NDS", "NBJS", "WWO"} <= set(names)


@pytest.mark.parametrize("optimizer_class", _OPTIMIZERS, ids=lambda optimizer: optimizer.__name__)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")],
        ),
    ],
)
def test_optimizer_execution_preserves_tensor_and_fitness_contracts(optimizer_class, dtype, device):
    torch.manual_seed(12)
    name = optimizer_class.__name__
    if name in {"GP", "GSGP"}:
        space = TreeSpace(
            n_agents=16,
            n_variables=4,
            lower_bound=-2,
            upper_bound=2,
            n_terminals=4,
            functions=["SUM", "SUB", "MUL"],
            device=device,
        )
    elif name == "NDS":
        values = torch.linspace(-1, 1, 20, device=device, dtype=dtype).reshape(5, 4)
        space = ParetoSpace(values, device=device)
    else:
        binary = optimizer_class.__module__.split(".")[2] == "boolean"
        space = Space(
            n_agents=60 if name == "LOA" else 16,
            n_variables=4,
            n_dimensions=2,
            lower_bound=0 if binary else -2,
            upper_bound=1 if binary else 2,
            device=device,
        )
        space.build()
        if binary:
            space.population.initialize_binary()
    if name in {"GP", "GSGP"}:
        space.to(space.device, dtype=dtype)
    else:
        space.population.to(space.device, dtype=dtype)

    optimizer = optimizer_class()
    function = Function(_sphere)
    engine = Otorchmizer(space, optimizer, function)
    engine.start(2)
    candidates = space.population

    assert candidates.positions.shape == (candidates.n_agents, candidates.n_variables, candidates.n_dimensions)
    assert candidates.fitness.shape == (candidates.n_agents,)
    for value in (candidates.positions, candidates.fitness, candidates.best_position, candidates.best_fitness):
        assert value.dtype is dtype
        assert value.device.type == device
        assert torch.isfinite(value).all()

    if name == "NDS":
        assert optimizer.n_pareto_points == 1
        assert (optimizer.status >= 0).all()
    else:
        assert (candidates.positions >= candidates.lb).all()
        assert (candidates.positions <= candidates.ub).all()
        assert (candidates.best_position >= candidates.lb).all()
        assert (candidates.best_position <= candidates.ub).all()
        torch.testing.assert_close(candidates.fitness, function(candidates.positions))
        torch.testing.assert_close(candidates.best_fitness, function(candidates.best_position.unsqueeze(0))[0])
