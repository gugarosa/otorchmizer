# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.device import DeviceManager
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer
from otorchmizer.core.population import Population
from otorchmizer.core.space import Space
from otorchmizer.otorchmizer import Otorchmizer
from otorchmizer.utils.callback import CallbackVessel, DiscreteSearchCallback


def _population(n_agents=2, dtype=torch.float32):
    return Population(n_agents, 1, 1, torch.zeros(1), torch.ones(1), dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_population_unevaluated_fitness_is_representable(dtype):
    population = _population(dtype=dtype)

    assert population.fitness.dtype == dtype
    assert population.best_fitness.dtype == dtype
    assert torch.isposinf(population.fitness).all()
    assert torch.isposinf(population.best_fitness)


def test_population_tracks_fitness_above_float32_maximum():
    population = _population(dtype=torch.float64)
    population.positions[:] = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.float64)
    population.fitness[:] = torch.tensor([1e100, 2e100], dtype=torch.float64)

    population.update_best()

    assert population.best_fitness.item() == 1e100
    assert population.best_position.item() == 0.25


def test_population_initialize_static_preserves_configured_dtype():
    population = _population(dtype=torch.float64)
    values = torch.tensor([[0.25], [0.75]], dtype=torch.float32)

    population.initialize_static(values)

    assert population.positions.dtype == population.dtype == torch.float64
    torch.testing.assert_close(population.positions, values.unsqueeze(-1).double())


def test_population_initialize_static_rejects_inconsistent_shape():
    population = _population()

    with pytest.raises(e.SizeError, match="values"):
        population.initialize_static(torch.zeros(3, 1))

    assert population.positions.shape == (2, 1, 1)


@pytest.mark.parametrize("method", ["rand", "randn"])
def test_device_manager_random_factories_preserve_dtype(method):
    manager = DeviceManager("cpu", dtype=torch.float64)

    result = getattr(manager, method)(3, 2)

    assert result.dtype == torch.float64
    assert result.device == torch.device("cpu")
    assert result.shape == (3, 2)


@pytest.mark.parametrize("enabled", [True, False])
def test_device_manager_cpu_autocast_controls_dtype(enabled):
    manager = DeviceManager("cpu")
    values = torch.ones(2, 2)

    with manager.autocast(enabled=enabled):
        result = values @ values

    assert result.dtype == (torch.bfloat16 if enabled else torch.float32)


def test_device_manager_scatter_uses_every_target_device():
    values = torch.arange(4)
    devices = [torch.device("cpu")] * 3

    chunks = DeviceManager.scatter(values, devices)

    assert [len(chunk) for chunk in chunks] == [2, 1, 1]
    assert torch.equal(DeviceManager.gather(chunks, torch.device("cpu")), values)


def test_population_scatter_uses_every_target_device():
    population = _population(n_agents=4)
    population.positions[:, 0, 0] = torch.arange(4)

    chunks = population.scatter([torch.device("cpu")] * 3)

    assert [chunk.n_agents for chunk in chunks] == [2, 1, 1]
    merged = Population.gather(chunks, torch.device("cpu"))
    assert torch.equal(merged.positions, population.positions)


@pytest.mark.parametrize("count", [0, 3])
def test_population_scatter_rejects_impossible_target_counts(count):
    population = _population(n_agents=2)

    with pytest.raises(e.ValueError, match="devices"):
        population.scatter([torch.device("cpu")] * count)


def test_device_manager_scatter_rejects_no_targets():
    with pytest.raises(ValueError, match="devices"):
        DeviceManager.scatter(torch.arange(4), [])


def test_population_gather_selects_best_without_float32_rounding():
    worse = _population(n_agents=1, dtype=torch.float64)
    better = _population(n_agents=1, dtype=torch.float64)
    worse.best_fitness.fill_(1.00000004)
    better.best_fitness.fill_(1.00000001)
    worse.best_position.fill_(0.4)
    better.best_position.fill_(0.1)

    merged = Population.gather([worse, better], torch.device("cpu"))

    assert merged.best_fitness.dtype == torch.float64
    assert merged.best_fitness.item() == better.best_fitness.item()
    assert torch.equal(merged.best_position, better.best_position)


def test_population_gather_rejects_empty_input():
    with pytest.raises(e.ValueError, match="populations"):
        Population.gather([], torch.device("cpu"))


def test_otorchmizer_update_uses_compiled_dispatch(monkeypatch):
    calls = []
    optimizer = Optimizer()
    monkeypatch.setattr(optimizer, "update", lambda ctx: calls.append("ordinary"))
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: lambda ctx: calls.append("compiled"))
    optimizer.torch_compile()
    space = Space(2, 1, device="cpu")
    space.build()
    model = Otorchmizer(space, optimizer, Function(lambda x: x.sum()))

    model.update(CallbackVessel())

    assert calls == ["compiled"]


def test_discrete_search_callback_preserves_float64_allowed_values():
    population = _population(n_agents=1, dtype=torch.float64)
    population.positions.fill_(0.5 + 3 * 2**-33)
    callback = DiscreteSearchCallback([[0.5, 0.5 + 2**-31]])

    callback.on_evaluate_before(population, None)

    assert population.positions.item() == 0.5 + 2**-31
