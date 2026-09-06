# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Front-ranking and engine regressions for nondominated sorting."""

from types import SimpleNamespace

import pytest
import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, UpdateContext
from otorchmizer.optimizers.misc import NDS
from otorchmizer.spaces import ParetoSpace


def _unused_objective(_position):
    raise AssertionError("NDS must rank objective vectors without scalarizing them")


@pytest.mark.parametrize(
    ("maximize", "expected"),
    [(True, [0, 1, 1, 2, 0, 0]), (False, [2, 1, 1, 0, 2, 0])],
)
def test_nondominated_fronts_handle_duplicates_and_both_orientations(maximize, expected):
    values = torch.tensor([[3, 3], [3, 2], [2, 3], [2, 2], [3, 3], [1, 4]], dtype=torch.float64)
    space = ParetoSpace(values, device="cpu")
    optimizer = NDS({"maximize": maximize})
    optimizer.compile(space.population)
    function = Function(_unused_objective)
    ctx = UpdateContext(space, function, 0, 1, space.device)

    optimizer.update(ctx)
    first_counts = optimizer.count.clone()
    optimizer.update(ctx)

    assert optimizer.status.tolist() == expected
    assert optimizer.n_pareto_points == expected.count(0)
    assert torch.equal(optimizer.count, first_counts)
    assert not optimizer.set[0, 4]
    assert not optimizer.set[4, 0]
    assert not optimizer.set.diagonal().any()
    assert torch.equal(space.population.positions.squeeze(-1), values)
    assert torch.equal(space.population.fitness, optimizer.status.to(dtype=space.population.dtype))


def test_nondominated_sorting_preserves_close_float64_objectives():
    values = torch.tensor([[1.0, 1.0], [1.0 + 2**-30, 1.0]], dtype=torch.float64)
    space = ParetoSpace(values, device="cpu")
    optimizer = NDS()
    engine = Otorchmizer(space, optimizer, Function(_unused_objective))

    engine.start(1)

    assert space.population.dtype is torch.float64
    assert torch.equal(space.population.positions.squeeze(-1), values)
    assert optimizer.status.tolist() == [1, 0]
    assert optimizer.n_pareto_points == 1
    assert torch.equal(space.best_position.squeeze(-1), values[1])


def test_nondominated_sorting_refreshes_the_current_front_representative():
    space = ParetoSpace(torch.tensor([[2.0, 2.0], [1.0, 1.0]]), device="cpu")
    optimizer = NDS()
    optimizer.compile(space.population)
    function = Function(_unused_objective)
    optimizer.evaluate(space.population, function)
    space.population.positions[0].zero_()

    optimizer.evaluate(space.population, function)

    assert optimizer.status.tolist() == [1, 0]
    assert torch.equal(space.best_position, space.population.positions[1])
    assert space.best_fitness.item() == 0


def test_nondominated_sorting_rejects_nan_objectives():
    space = ParetoSpace(torch.tensor([[1.0, torch.nan], [0.0, 0.0]]), device="cpu")
    optimizer = NDS()
    optimizer.compile(space.population)
    ctx = UpdateContext(SimpleNamespace(population=space.population), Function(_unused_objective), 0, 1, space.device)

    with pytest.raises(ValueError, match="NaN objective"):
        optimizer.update(ctx)
