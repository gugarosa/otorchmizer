# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Equation regressions for the atom-search migration."""

from math import exp, sqrt
from types import SimpleNamespace

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.core import Function, Population, UpdateContext
from otorchmizer.optimizers.science import ASO


def _context(positions, fitness, iteration=0):
    population = Population(
        len(positions),
        positions.shape[1],
        positions.shape[2],
        torch.full(positions.shape[1:], -100.0),
        torch.full(positions.shape[1:], 100.0),
        dtype=positions.dtype,
    )
    population.initialize_static(positions.clone())
    population.fitness = fitness.clone()
    population.update_best()
    return UpdateContext(
        SimpleNamespace(population=population), Function(lambda x: x.square().sum()), iteration, 10, population.device
    )


def test_atom_force_uses_lennard_jones_potential_and_inverse_mass(monkeypatch):
    positions = torch.tensor([[[0.0], [3.0]], [[2.0], [5.0]]], dtype=torch.float64)
    ctx = _context(positions, torch.ones(2, dtype=positions.dtype))
    optimizer = ASO({"alpha": 1.0, "beta": 0.0})
    optimizer.compile(ctx.space.population)
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.ones(*shape, **kwargs))

    optimizer.update(ctx)

    potential = 6 * 1.24**-7 - 12 * 1.24**-13
    expected = torch.full((2, 1), 2 * potential / sqrt(2), dtype=positions.dtype)
    assert torch.allclose(optimizer.velocity[0], expected)
    assert torch.allclose(optimizer.velocity[1], -expected)
    assert torch.allclose(ctx.space.population.positions, positions + optimizer.velocity)


def test_atom_centroid_is_translation_invariant(monkeypatch):
    positions = torch.tensor([[[0.0, 2.0], [3.0, 1.0]], [[2.0, 4.0], [5.0, 3.0]]], dtype=torch.float64)
    offset = positions.new_tensor([[10.0, -7.0], [30.0, 2.0]])
    contexts = [_context(values, positions.new_ones(2)) for values in (positions, positions + offset)]
    velocities = []
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.ones(*shape, **kwargs))

    for ctx in contexts:
        optimizer = ASO({"alpha": 1.0, "beta": 0.0})
        optimizer.compile(ctx.space.population)
        optimizer.update(ctx)
        velocities.append(optimizer.velocity)

    assert torch.allclose(velocities[0], velocities[1])


def test_atom_constraint_shares_decay_and_mass_scaling():
    positions = torch.tensor([[[0.0]], [[2.0]]], dtype=torch.float64)
    ctx = _context(positions, positions.new_tensor([0.0, 1.0]), iteration=1)
    optimizer = ASO({"alpha": 0.0, "beta": 0.2})
    optimizer.compile(ctx.space.population)

    optimizer.update(ctx)

    second_mass = exp(-1) / (1 + exp(-1))
    expected = exp(-2) * 0.2 * -2 / second_mass
    assert optimizer.velocity[0].item() == 0
    assert optimizer.velocity[1].item() == pytest.approx(expected)


def test_atom_handles_coincident_atoms_and_extreme_finite_fitness():
    positions = torch.zeros((3, 2, 2), dtype=torch.float64)
    ctx = _context(positions, positions.new_tensor([-1e308, 0, 1e308]))
    optimizer = ASO()
    optimizer.compile(ctx.space.population)

    optimizer.update(ctx)

    assert torch.equal(ctx.space.population.positions, positions)
    assert torch.isfinite(optimizer.velocity).all()


@pytest.mark.parametrize("beta", [-0.1, 1.1])
def test_atom_rejects_invalid_constraint_weight(beta):
    with pytest.raises(e.ValueError, match="`beta` must be between"):
        ASO({"beta": beta})
