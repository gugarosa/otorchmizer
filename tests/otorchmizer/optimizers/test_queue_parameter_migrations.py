# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Regressions for queue phases and canonical migration parameters."""

from math import sin
from types import SimpleNamespace

import pytest
import torch

import otorchmizer.math.random as r
import otorchmizer.utils.exception as e
from otorchmizer.core import Function, Population, UpdateContext
from otorchmizer.optimizers.boolean import UMDA
from otorchmizer.optimizers.science import WDO
from otorchmizer.optimizers.social import QSA, SSD


def _context(values, square=True):
    positions = torch.tensor(values, dtype=torch.float64).reshape(-1, 1, 1)
    population = Population(len(values), 1, 1, torch.tensor([-100.0]), torch.tensor([100.0]), dtype=positions.dtype)
    population.initialize_static(positions)
    calls = []

    def objective(candidates):
        calls.extend(candidates[:, 0, 0].tolist())
        values = candidates[:, 0, 0]
        return values.square() if square else values

    population.fitness = positions[:, 0, 0].square() if square else positions[:, 0, 0].clone()
    population.update_best()
    return (
        UpdateContext(
            SimpleNamespace(population=population), Function(objective, batch=True), 0, 10, population.device
        ),
        calls,
    )


def test_queue_allocation_uses_reciprocal_leader_fitness(monkeypatch):
    ctx, calls = _context([1, 2, 4, 10, 12, 14], square=False)
    optimizer = QSA()
    optimizer.compile(ctx.space.population)
    monkeypatch.setattr(torch, "rand", lambda size, **kwargs: torch.full(size, 0.5, **kwargs))
    monkeypatch.setattr(
        r, "generate_gamma_random_number", lambda _shape, _scale, size, **kwargs: torch.zeros(size, **kwargs)
    )

    optimizer.update(ctx)

    assert calls[:6] == [1, 2, 1, 2, 4, 4]
    assert torch.equal(ctx.space.population.fitness, ctx.space.population.positions[:, 0, 0])


def test_queue_second_phase_uses_a_difference_of_distinct_donors(monkeypatch):
    ctx, calls = _context([-3, -2, 1])
    optimizer = QSA()
    optimizer.compile(ctx.space.population)
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))
    monkeypatch.setattr(torch, "randperm", lambda _n, **kwargs: torch.tensor([0, 2, 1], **kwargs))
    monkeypatch.setattr(
        r,
        "generate_gamma_random_number",
        lambda _shape, _scale, size, **kwargs: torch.full(
            size if isinstance(size, tuple) else (size,), float(3 <= len(calls) < 6), **kwargs
        ),
    )

    optimizer.update(ctx)

    assert calls[:3] == [1, -2, -3]
    assert calls[3:6] == [5, 2, 1]
    assert torch.equal(ctx.space.population.positions, torch.ones_like(ctx.space.population.positions))
    assert ctx.space.population.best_fitness.item() == 1


def test_queue_third_phase_updates_coordinates(monkeypatch):
    ctx, calls = _context([1, 3, 4])
    optimizer = QSA()
    optimizer.compile(ctx.space.population)
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))
    monkeypatch.setattr(torch, "randperm", lambda n, **kwargs: torch.arange(n, **kwargs))
    monkeypatch.setattr(
        r,
        "generate_gamma_random_number",
        lambda _shape, _scale, size, **kwargs: torch.full(
            size if isinstance(size, tuple) else (size,), float(len(calls) >= 6), **kwargs
        ),
    )

    optimizer.update(ctx)

    assert calls[6:9] == [3, 1, -2]
    assert torch.equal(ctx.space.population.positions[:, 0, 0], torch.tensor([1, 1, -2], dtype=torch.float64))
    assert torch.equal(ctx.space.population.fitness, ctx.space.population.positions[:, 0, 0].square())


def test_wind_canonical_c_parameter_controls_coriolis_force(monkeypatch):
    ctx, _ = _context([0, 0])
    optimizer = WDO({"alpha": 0.0, "g": 0.0, "RT": 0.0, "c": 0.0, "v_max": 3.0})
    optimizer.compile(ctx.space.population)
    optimizer.velocity.fill_(1)
    monkeypatch.setattr(
        torch, "randint", lambda _low, _high, size, **kwargs: torch.zeros(size, dtype=torch.long, **kwargs)
    )

    optimizer.update(ctx)

    assert torch.equal(optimizer.velocity, torch.ones_like(optimizer.velocity))


def test_ski_canonical_c_parameter_controls_velocity_and_decays(monkeypatch):
    ctx, _ = _context([2, 2, 2])
    optimizer = SSD({"c": 3.0, "decay": 0.5})
    optimizer.compile(ctx.space.population)
    optimizer.velocity.fill_(1)
    optimizer.local_position.fill_(4)
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.full(shape, 0.5, **kwargs))

    optimizer.update(ctx)

    assert torch.allclose(optimizer.velocity, torch.full_like(optimizer.velocity, 2 * sin(0.5)))
    assert optimizer.c == 1.5


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_umda_canonical_probability_bounds_control_samples(monkeypatch, probability):
    ctx, _ = _context([1 - probability] * 4)
    optimizer = UMDA({"lower_bound": probability, "upper_bound": probability})
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.5))

    optimizer.update(ctx)

    assert torch.equal(ctx.space.population.positions, torch.full_like(ctx.space.population.positions, probability))


@pytest.mark.parametrize("params", [{"p_selection": 0}, {"lower_bound": 0.8, "upper_bound": 0.2}])
def test_umda_rejects_invalid_selection_and_probability_intervals(params):
    with pytest.raises(e.ValueError):
        UMDA(params)
