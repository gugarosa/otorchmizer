# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Execution checks requiring an actual CUDA device."""

import pytest
import torch

from otorchmizer.core import DeviceManager, Population
from otorchmizer.functions import ConstrainedFunction
from otorchmizer.functions.multi_objective import MultiObjectiveWeightedFunction
from otorchmizer.spaces import GridSpace, SearchSpace
from otorchmizer.utils.callback import DiscreteSearchCallback

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


def test_cuda_graph_replay_mutates_the_retained_storage():
    manager = DeviceManager("cuda:0", dtype=torch.float64)
    values = manager.zeros(8)
    pointer = values.data_ptr()
    runner = manager.capture_graph(torch.Tensor.add_, values, 1.0, warmup=2)
    before = values.clone()

    runner.replay()
    runner.replay()
    torch.cuda.synchronize()

    assert values.data_ptr() == pointer
    torch.testing.assert_close(values, before + 2)


def test_cuda_autocast_uses_the_documented_reduced_precision():
    manager = DeviceManager("cuda:0")
    matrix = manager.ones(8, 8)

    with manager.autocast():
        product = matrix @ matrix

    assert product.is_cuda
    assert product.dtype is torch.float16
    torch.testing.assert_close(product, torch.full_like(product, 8))


def test_mixed_cpu_cuda_scatter_gather_preserves_float64_population():
    population = Population(
        5,
        2,
        2,
        torch.full((2,), -1.0),
        torch.ones(2),
        device=torch.device("cuda:0"),
        dtype=torch.float64,
    )
    population.initialize_uniform()
    population.fitness = population.positions.square().sum(dim=(1, 2))
    population.update_best()

    shards = population.scatter([torch.device("cpu"), torch.device("cuda:0")])
    merged = Population.gather(shards, torch.device("cuda:0"))

    assert [shard.n_agents for shard in shards] == [3, 2]
    assert shards[0].device.type == "cpu"
    assert shards[1].device.type == "cuda"
    assert merged.dtype is torch.float64
    assert torch.equal(merged.positions, population.positions)
    assert torch.equal(merged.fitness, population.fitness)
    assert torch.equal(merged.best_position, population.best_position)
    assert torch.equal(merged.best_fitness, population.best_fitness)


def test_cuda_support_contracts_preserve_precision_bounds_and_archive_ownership():
    space = SearchSpace(2, 1, [1.00000001], [1.00000002], dtype=torch.float64, device="cuda:0")
    assert space.population.lb.item() == 1.00000001
    space.population.positions.fill_(1.000000019)
    DiscreteSearchCallback([[1.00000001, 1.00000002]]).on_evaluate_before(space.population, None)
    assert (space.population.positions == 1.00000002).all()
    weighted = MultiObjectiveWeightedFunction([lambda x: x.new_tensor(1.0)], [1.00000001])
    result = weighted(space.population.positions)
    assert result.is_cuda
    assert result[0].item() == 1.00000001
    constrained = ConstrainedFunction(lambda x: x.sum(), [lambda x: x.new_tensor(True)], penalty=1)
    assert torch.isposinf(constrained(torch.full((2, 1, 1), torch.inf, device="cuda:0"))).all()
    grid = GridSpace(1, 0.1, 0, 0.9, device="cuda:0", dtype=torch.float64)
    assert grid.n_agents == 10
    assert grid.grid[-1, 0].item() == 0.9
    shards = space.population.scatter([torch.device("cuda:0"), torch.device("cpu")])
    before = space.population.best_position.clone()
    shards[0].best_position.zero_()
    assert torch.equal(space.population.best_position, before)
