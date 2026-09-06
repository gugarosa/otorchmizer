# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Regression tests for miscellaneous, science, social, and Boolean optimizer contracts."""

from types import SimpleNamespace

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.optimizers.boolean import BMRFO, BPSO
from otorchmizer.optimizers.misc import CEM, DOA
from otorchmizer.optimizers.science import ASO, CDO, EO, ESA, GSA, HGSO, LSA, SMA, WCA, WDO
from otorchmizer.optimizers.social import CI, ISA, MVPA, QSA, SSD


def _population(n_agents=4, n_variables=2, n_dimensions=1, dtype=torch.float32):
    return Population(
        n_agents=n_agents,
        n_variables=n_variables,
        n_dimensions=n_dimensions,
        lower_bound=torch.full((n_variables,), -10.0, dtype=dtype),
        upper_bound=torch.full((n_variables,), 10.0, dtype=dtype),
        dtype=dtype,
    )


def _context(population, function, iteration=0, n_iterations=10):
    return UpdateContext(
        space=SimpleNamespace(population=population),
        function=function,
        iteration=iteration,
        n_iterations=n_iterations,
        device=population.device,
    )


def _sum(position):
    return position.sum()


def _sphere(position):
    return position.square().sum()


def test_bmrfo_updates_global_best_between_foraging_phases(monkeypatch):
    population = _population(n_variables=4)
    population.positions.fill_(1)
    population.fitness.fill_(4)
    population.best_position.fill_(1)
    population.best_fitness.fill_(4)
    function = Function(_sum)

    monkeypatch.setattr(torch, "rand", lambda *size, **kwargs: torch.zeros(*size, **kwargs))
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)

    BMRFO().update(_context(population, function))

    assert population.best_fitness.item() == 0
    assert torch.count_nonzero(population.best_position) == 0
    assert torch.count_nonzero(population.positions) == 0


@pytest.mark.parametrize("optimizer_class", [BPSO, ISA, SSD])
def test_personal_best_updates_without_beating_global_best(optimizer_class):
    population = _population(n_agents=2, n_variables=4)
    population.positions[0].zero_()
    population.positions[1].fill_(1)
    function = Function(_sphere)
    optimizer = optimizer_class()
    optimizer.compile(population)
    optimizer.evaluate(population, function)

    improved_position = population.positions[1].clone()
    improved_position[2:].zero_()
    population.positions[1] = improved_position
    optimizer.evaluate(population, function)

    assert torch.equal(
        optimizer.local_position[1], improved_position.bool() if optimizer_class is BPSO else improved_position
    )
    assert optimizer.local_fitness[1].item() == 2
    assert population.best_fitness.item() == 0


def test_sma_weights_are_finite_and_canonically_bounded():
    population = _population(n_agents=4, n_variables=1)
    population.positions[:, 0, 0] = torch.arange(4)
    population.fitness = torch.arange(4, dtype=population.dtype)
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = SMA({"z": 1.0})
    optimizer.compile(population)

    torch.manual_seed(2)
    optimizer.update(_context(population, Function(_sphere)))

    bound = torch.log10(population.fitness.new_tensor(2.0))
    assert torch.isfinite(optimizer.weight).all()
    assert optimizer.weight.min() >= 1 - bound
    assert optimizer.weight.max() <= 1 + bound


def test_eo_generation_rate_includes_exponential_factor(monkeypatch):
    population = _population(n_agents=1, n_variables=1)
    population.positions.fill_(4)
    population.fitness.fill_(10)
    population.best_position.fill_(4)
    population.best_fitness.fill_(10)
    optimizer = EO()
    optimizer.compile(population)
    optimizer.C = [population.positions.new_full((1, 1), 2) for _ in range(4)]
    optimizer.C_fit = [population.fitness.new_tensor(float(i)) for i in range(4)]

    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([0], device=population.device))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.75, device=kwargs.get("device"), dtype=population.dtype),
    )

    optimizer.update(_context(population, Function(_sphere)))

    factor = 2 * (torch.exp(population.positions.new_tensor(-0.75)) - 1)
    initial_generation = 0.5 * 0.75 * (2 - 0.75 * 4)
    expected = 2 + (4 - 2) * factor + (initial_generation * factor / 0.75) * (1 - factor)
    assert torch.allclose(population.positions.squeeze(), expected)


def test_eo_requires_positive_volume():
    with pytest.raises(e.ValueError, match="`V` must be positive"):
        EO({"V": 0})


def test_ci_supports_full_dimensions_and_signed_fitness(monkeypatch):
    population = _population(n_agents=3, n_variables=2, n_dimensions=3)
    population.positions.copy_(torch.arange(18, dtype=population.dtype).reshape_as(population.positions) / 10)
    population.fitness = population.fitness.new_tensor([-1.0, 0.0, 1.0])
    optimizer = CI({"t": 0})
    optimizer.compile(population)
    sampled_weights = []

    def select(weights, *_args, **_kwargs):
        sampled_weights.append(weights.clone())
        return torch.zeros(1, dtype=torch.long, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", select)
    optimizer.update(_context(population, Function(_sphere)))

    assert optimizer.lower.shape == population.positions.shape
    assert optimizer.upper.shape == population.positions.shape
    assert sampled_weights
    assert all(torch.isfinite(weights).all() and (weights >= 0).all() for weights in sampled_weights)
    assert all(torch.allclose(weights.sum(), weights.new_tensor(1.0)) for weights in sampled_weights)


def test_ci_preserves_reciprocal_probabilities_for_positive_fitness(monkeypatch):
    population = _population(n_agents=3)
    population.fitness = population.fitness.new_tensor([1.0, 2.0, 4.0])
    optimizer = CI({"t": 0})
    optimizer.compile(population)
    sampled_weights = []

    def select(weights, *_args, **_kwargs):
        sampled_weights.append(weights.clone())
        return torch.zeros(1, dtype=torch.long, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", select)
    optimizer.update(_context(population, Function(_sphere)))

    expected = population.fitness.reciprocal()
    expected /= expected.sum()
    assert torch.allclose(sampled_weights[0], expected)


def test_ci_preserves_positive_reciprocal_probabilities_with_unscored_candidates(monkeypatch):
    population = _population(n_agents=3)
    population.fitness = population.fitness.new_tensor([1.0, 2.0, torch.inf])
    optimizer = CI({"t": 0})
    optimizer.compile(population)
    sampled_weights = []

    def select(weights, *_args, **_kwargs):
        sampled_weights.append(weights.clone())
        return torch.zeros(1, dtype=torch.long, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", select)
    optimizer.update(_context(population, Function(_sphere)))

    assert torch.allclose(sampled_weights[0], population.fitness.new_tensor([2 / 3, 1 / 3, 0]))


@pytest.mark.parametrize("invalid", [torch.nan, -torch.inf])
def test_ci_rejects_invalid_fitness_values(invalid):
    population = _population(n_agents=3)
    population.fitness = population.fitness.new_tensor([1.0, invalid, torch.inf])
    optimizer = CI()
    optimizer.compile(population)

    with pytest.raises(e.ValueError, match="must not contain NaN or negative infinity"):
        optimizer.update(_context(population, Function(_sphere)))


def test_ci_rejects_an_entirely_unscored_population():
    population = _population(n_agents=3)
    optimizer = CI()
    optimizer.compile(population)

    with pytest.raises(e.ValueError, match="`population.fitness` must contain at least one finite value"):
        optimizer.update(_context(population, Function(_sphere)))


def test_ci_shrinks_intervals_symmetrically_around_selected_position(monkeypatch):
    population = _population(n_agents=2, n_variables=1)
    population.positions[:, 0, 0] = population.positions.new_tensor([2.0, -3.0])
    population.fitness = population.fitness.new_tensor([1.0, 2.0])
    optimizer = CI({"r": 0.8, "t": 0})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda weights, *_args, **_kwargs: torch.zeros(1, dtype=torch.long, device=weights.device),
    )

    optimizer.update(_context(population, Function(_sphere)))

    assert torch.allclose(optimizer.lower[:, 0, 0], population.positions.new_tensor([-6.0, -6.0]))
    assert torch.allclose(optimizer.upper[:, 0, 0], population.positions.new_tensor([10.0, 10.0]))


def test_cem_adapts_every_dimension(monkeypatch):
    population = _population(n_agents=3, n_variables=2, n_dimensions=2)
    optimizer = CEM({"alpha": 0.0, "n_updates": 3})
    optimizer.compile(population)
    optimizer.mean.zero_()
    optimizer.std.fill_(1)
    samples = population.positions.new_tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[3.0, 6.0], [9.0, 10.0]],
        ]
    )
    monkeypatch.setattr(torch, "randn_like", lambda _tensor: samples.clone())

    optimizer.update(_context(population, Function(_sphere)))

    assert optimizer.mean.shape == (2, 2)
    assert optimizer.std.shape == (2, 2)
    expected_mean = samples.mean(dim=0)
    expected_std = ((samples - expected_mean.unsqueeze(0)) ** 2).mean(dim=0).sqrt()
    assert torch.allclose(optimizer.mean, expected_mean)
    assert torch.allclose(optimizer.std, expected_std)


def test_cem_constant_elites_have_zero_spread(monkeypatch):
    population = _population(n_agents=3, n_variables=2, n_dimensions=2)
    optimizer = CEM({"alpha": 0.0, "n_updates": 3})
    optimizer.compile(population)
    optimizer.mean.zero_()
    optimizer.std.fill_(1)
    samples = population.positions.new_full(population.positions.shape, 3.5)
    monkeypatch.setattr(torch, "randn_like", lambda _tensor: samples.clone())

    optimizer.update(_context(population, Function(_sphere)))

    assert torch.equal(optimizer.mean, population.positions.new_full((2, 2), 3.5))
    assert torch.count_nonzero(optimizer.std) == 0


@pytest.mark.parametrize(
    ("optimizer", "population", "message"),
    [
        (ISA(), _population(n_agents=1), "at least 2 for ISA"),
        (MVPA(), _population(n_agents=3), "at least `n_teams`"),
        (MVPA({"n_teams": 1}), _population(), "`n_teams` must be at least 2"),
        (QSA(), _population(n_agents=2), "at least 3 for QSA"),
        (WCA(), _population(n_agents=1), "at least `nsr`"),
    ],
)
def test_population_constraints_are_explicit(optimizer, population, message):
    with pytest.raises((e.SizeError, e.ValueError), match=message):
        optimizer.compile(population)


def test_wca_uses_reference_flow_raining_and_schedule(monkeypatch):
    population = _population(n_agents=4, n_variables=1)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.0, 1.0, 2.0, 3.0])
    population.fitness = population.fitness.new_tensor([1.0, 1.0, 4.0, 9.0])
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = WCA({"d_max": 10.0})
    optimizer.compile(population)

    monkeypatch.setattr(torch, "rand", lambda *size, **kwargs: torch.zeros(*size, **kwargs))
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    optimizer.update(_context(population, Function(_sphere), n_iterations=10))

    assert optimizer.flows.tolist() == [1, 1]
    assert population.positions[:, 0, 0].tolist() == [0.0, 1.0, 0.0, -10.0]
    assert optimizer.d_max == pytest.approx(9.0)


def test_wca_flow_displacement_uses_reference_factor(monkeypatch):
    population = _population(n_agents=3, n_variables=1)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.0, 4.0, 2.0])
    population.fitness = population.fitness.new_tensor([1.0, 1.0, 4.0])
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = WCA({"d_max": 0.0})
    optimizer.compile(population)

    def quarter(*size, **kwargs):
        return torch.full(size, 0.25, device=kwargs.get("device"), dtype=population.dtype)

    monkeypatch.setattr(torch, "rand", quarter)
    optimizer.update(_context(population, Function(_sphere)))

    assert torch.allclose(population.positions[:, 0, 0], population.positions.new_tensor([0.0, 1.0, 2.0]))


def test_wca_refreshes_best_before_raining(monkeypatch):
    population = _population(n_agents=3, n_variables=1)
    population.positions[:, 0, 0] = population.positions.new_tensor([4.0, 6.0, 1.0])
    population.fitness = population.fitness.new_tensor([16.0, 36.0, 1.0])
    population.best_position.fill_(4)
    population.best_fitness.fill_(16)
    optimizer = WCA({"d_max": 10.0})
    optimizer.compile(population)

    monkeypatch.setattr(torch, "rand", lambda *size, **kwargs: torch.zeros(*size, **kwargs))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    optimizer.update(_context(population, Function(_sphere)))

    assert population.best_fitness.item() == 1
    assert population.best_position.item() == 1
    assert population.positions[2].item() == 1


def test_compiled_floating_state_uses_population_dtype_and_infinite_sentinels():
    population = _population(n_agents=8, n_dimensions=2, dtype=torch.float64)
    cdo = CDO()
    eo = EO()
    bpso = BPSO()
    optimizers_and_state = [
        (CEM(), ("mean", "std")),
        (DOA(), ("chaotic_map",)),
        (ASO(), ("velocity",)),
        (cdo, ("alpha_pos", "beta_pos", "gamma_pos")),
        (eo, ("C",)),
        (ESA(), ("D",)),
        (GSA(), ("velocity",)),
        (HGSO(), ("coeff", "pressure", "constant")),
        (ISA(), ("local_position", "velocity", "local_fitness")),
        (LSA(), ("direction",)),
        (SMA(), ("weight",)),
        (SSD(), ("local_position", "velocity", "local_fitness")),
        (WDO(), ("velocity",)),
    ]

    bpso.compile(population)
    for optimizer, attributes in optimizers_and_state:
        optimizer.compile(population)
        for attribute in attributes:
            state = getattr(optimizer, attribute)
            tensors = state if isinstance(state, list) else [state]
            assert all(tensor.dtype == population.dtype for tensor in tensors)

    assert bpso.local_fitness.dtype == population.dtype
    assert torch.isinf(bpso.local_fitness).all()
    assert torch.isinf(torch.stack(eo.C_fit)).all()
    assert torch.isinf(torch.stack([cdo.alpha_fit, cdo.beta_fit, cdo.gamma_fit])).all()
