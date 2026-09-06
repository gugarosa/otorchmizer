# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Deterministic contracts for canonical science optimizer migrations."""

import math
from types import SimpleNamespace

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.optimizers.science import CDO, EFO, ESA, HGSO, LSA, TEO, TWO, WEO, WWO


def _population(n_agents=4, n_variables=1, n_dimensions=1, lower=-100.0, upper=100.0, dtype=torch.float64):
    return Population(
        n_agents=n_agents,
        n_variables=n_variables,
        n_dimensions=n_dimensions,
        lower_bound=torch.full((n_variables,), lower, dtype=dtype),
        upper_bound=torch.full((n_variables,), upper, dtype=dtype),
        dtype=dtype,
    )


def _context(population, function, iteration=1, n_iterations=10):
    return UpdateContext(
        space=SimpleNamespace(population=population),
        function=function,
        iteration=iteration,
        n_iterations=n_iterations,
        device=population.device,
    )


def _sphere(position):
    return position.square().sum()


def _negative_sum(position):
    return -position.sum()


def _positive_sphere(position):
    return position.square().sum() + 1


def _constant_rand(value, dtype):
    def generate(*size, **kwargs):
        shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
        return torch.full(shape, value, device=kwargs.get("device"), dtype=kwargs.get("dtype", dtype))

    return generate


def test_cdo_applies_three_canonical_radiation_components(monkeypatch):
    population = _population(n_agents=3)
    population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 2.0, 3.0])
    population.fitness = population.fitness.new_tensor([1.0, 2.0, 3.0])
    optimizer = CDO()
    optimizer.compile(population)
    initial = population.positions.clone()
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))

    optimizer.update(_context(population, Function(_sphere), iteration=2, n_iterations=10))

    ws = 2.4
    s_gamma = math.log10(1 + 0.5 * 299999)
    s_beta = math.log10(1 + 0.5 * 269999)
    s_alpha = math.log10(1 + 0.5 * 15999)

    def component(target, scale, denominator):
        rho = torch.pi * 0.25 / denominator - ws * 0.5
        gradient = (torch.pi * 0.25 * target - initial).abs()
        return scale * (initial - rho * gradient)

    expected = (
        component(population.positions.new_tensor([[3.0]]), 1.0, s_gamma)
        + component(population.positions.new_tensor([[2.0]]), 0.5, 0.5 * s_beta)
        + component(population.positions.new_tensor([[1.0]]), 0.25, 0.25 * s_alpha)
    ) / 3
    assert torch.allclose(population.positions, expected)
    assert optimizer.alpha_fit.item() == 1
    assert optimizer.beta_fit.item() == 2
    assert optimizer.gamma_fit.item() == 3


def test_cdo_half_precision_log_sampling_matches_float32(monkeypatch):
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, torch.float32))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))
    results = []

    for dtype in (torch.float16, torch.float32):
        population = _population(n_agents=3, dtype=dtype)
        population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 2.0, 3.0])
        population.fitness = population.fitness.new_tensor([1.0, 2.0, 3.0])
        optimizer = CDO()
        optimizer.compile(population)
        optimizer.update(_context(population, Function(_sphere), iteration=2, n_iterations=10))
        results.append(population.positions.float())

    assert torch.isfinite(results[0]).all()
    assert torch.allclose(results[0], results[1], atol=2e-2, rtol=2e-2)


def test_efo_builds_one_canonical_field_candidate(monkeypatch):
    population = _population(n_agents=4)
    population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 3.0, 5.0, 10.0])
    population.fitness = population.fitness.new_tensor([1.0, 9.0, 25.0, 100.0])
    population.best_position.fill_(1)
    population.best_fitness.fill_(1)
    optimizer = EFO()
    optimizer.compile(population)
    random_values = iter([0.25, 0.9, 0.9])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: population.positions.new_tensor(next(random_values)))
    ranks = iter([0, 3, 1])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(next(ranks), device=population.device))

    optimizer.update(_context(population, Function(_sphere)))

    expected = 10 + optimizer.phi * 0.25 * (1 - 3) - 0.25 * (10 - 3)
    assert population.positions[3, 0, 0].item() == pytest.approx(expected)
    assert population.fitness[3].item() == pytest.approx(expected**2)


def test_efo_rotates_random_reset_index(monkeypatch):
    population = _population(n_agents=4, n_variables=2, lower=-5, upper=5)
    population.positions[:, :, 0] = population.positions.new_tensor([[0, 0], [1, 1], [2, 2], [4, 4]])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = EFO({"ps_ratio": 1.0, "r_ratio": 1.0})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.0, population.dtype))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.zeros_like(tensor))
    ranks = iter([0, 3, 1, 0, 3, 1])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(next(ranks), device=population.device))

    optimizer.update(_context(population, Function(_sphere)))

    assert optimizer.RI == 1
    assert population.positions[-1, 0, 0].item() == -5


def test_esa_updates_orbital_radius_before_acceleration(monkeypatch):
    population = _population(n_agents=1)
    population.positions.fill_(2)
    population.fitness.fill_(100)
    population.best_position.fill_(1)
    population.best_fitness.fill_(1)
    optimizer = ESA({"n_electrons": 3})
    optimizer.compile(population)
    optimizer.D.fill_(1)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.full(size, 2, device=population.device),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))

    optimizer.update(_context(population, Function(_sphere)))

    expected_radius = 2.375 - 1 + 0.5 * (1 - 0.25)
    assert optimizer.D.item() == pytest.approx(expected_radius)
    assert population.positions.item() == pytest.approx(2 + 0.5 * expected_radius)


def test_esa_archives_best_electron_even_when_atom_does_not_move(monkeypatch):
    population = _population(n_agents=1)
    population.positions.fill_(1)
    population.fitness.fill_(2)
    population.best_position.fill_(1)
    population.best_fitness.fill_(2)
    optimizer = ESA({"n_electrons": 1})
    optimizer.compile(population)
    optimizer.D.fill_(0.75)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.full(size, 2, device=population.device),
    )
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.0, population.dtype))

    optimizer.update(_context(population, Function(_positive_sphere)))

    assert population.positions.item() == 1
    assert population.best_position.item() == 0
    assert population.best_fitness.item() == 1


def test_hgso_uses_cluster_best_and_henry_schedule(monkeypatch):
    population = _population(n_agents=4)
    population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 2.0, 10.0, 20.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.fill_(1)
    population.best_fitness.fill_(1)
    optimizer = HGSO({"n_clusters": 2, "alpha": 0.0})
    optimizer.compile(population)
    optimizer.coeff.fill_(1)
    optimizer.pressure.fill_(1)
    optimizer.constant[:] = population.fitness.new_tensor([0.1, 0.2])
    monkeypatch.setattr(torch, "rand", _constant_rand(0.75, population.dtype))

    optimizer.update(_context(population, Function(_sphere), iteration=1, n_iterations=2))

    temperature = math.exp(-0.5)
    expected_coefficient = math.exp(-0.2 * (1 / temperature - 1 / 298.15))
    gamma = math.exp(-(1.0 + 0.05) / (400.0 + 0.05))
    assert optimizer.coeff[1].item() == pytest.approx(expected_coefficient)
    assert population.positions[3].item() == pytest.approx(20 + 0.75 * gamma * (10 - 20))


def test_hgso_state_matches_cluster_and_population_cardinality():
    population = _population(n_agents=5, dtype=torch.float32)
    optimizer = HGSO({"n_clusters": 2})
    optimizer.compile(population)

    assert optimizer.coeff.shape == (2,)
    assert optimizer.constant.shape == (2,)
    assert optimizer.pressure.shape == (5,)
    assert optimizer.coeff.dtype == population.dtype
    with pytest.raises(e.SizeError, match="must not exceed"):
        HGSO({"n_clusters": 6}).compile(population)


def test_hgso_snapshots_cluster_best_before_member_updates(monkeypatch):
    population = _population(n_agents=2, lower=0, upper=10)
    population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 2.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.fill_(1)
    population.best_fitness.fill_(1)
    optimizer = HGSO({"n_clusters": 1})
    optimizer.compile(population)
    optimizer.coeff.fill_(10)
    optimizer.pressure.fill_(1)
    optimizer.constant.zero_()
    monkeypatch.setattr(torch, "rand", _constant_rand(0.25, population.dtype))

    optimizer.update(_context(population, Function(_sphere)))

    gamma = math.exp(-(1.0 + 0.05) / (4.0 + 0.05))
    expected_follower = 2 - 0.25 * gamma * (1 - 2) - 0.25 * (10 * 1 - 2)
    assert population.positions[1].item() == pytest.approx(expected_follower)


def test_hgso_rejects_signed_fitness_before_state_mutation():
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([-1.0, 0.0])
    population.fitness = population.positions[:, 0, 0].clone()
    population.best_position.fill_(-1)
    population.best_fitness.fill_(-1)
    optimizer = HGSO({"n_clusters": 2})
    optimizer.compile(population)
    coefficient = optimizer.coeff.clone()
    positions = population.positions.clone()

    with pytest.raises(e.ValueError, match="finite non-negative values for HGSO"):
        optimizer.update(_context(population, Function(_sphere)))

    assert torch.equal(optimizer.coeff, coefficient)
    assert torch.equal(population.positions, positions)


def test_hgso_half_precision_positive_domain_matches_float32(monkeypatch):
    monkeypatch.setattr(torch, "rand", _constant_rand(0.25, torch.float32))
    results = []

    for dtype in (torch.float16, torch.float32):
        population = _population(n_agents=2, lower=0, upper=10, dtype=dtype)
        population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 2.0])
        population.fitness = population.positions.square().sum(dim=(-1, -2))
        population.best_position.fill_(1)
        population.best_fitness.fill_(1)
        optimizer = HGSO({"n_clusters": 2})
        optimizer.compile(population)
        optimizer.coeff.fill_(0.5)
        optimizer.pressure.fill_(1)
        optimizer.constant.zero_()
        optimizer.update(_context(population, Function(_sphere)))
        results.append(population.positions.float())

    assert torch.isfinite(results[0]).all()
    assert torch.allclose(results[0], results[1], atol=2e-2, rtol=2e-2)


def test_hgso_rejects_invalid_worst_gas_replacement_before_commit(monkeypatch):
    population = _population(n_agents=10, lower=-2, upper=2)
    population.positions.fill_(1)
    population.fitness.fill_(1)
    population.best_position.fill_(1)
    population.best_fitness.fill_(1)
    optimizer = HGSO({"alpha": 0.0, "beta": 0.0})
    optimizer.compile(population)
    optimizer.constant.zero_()
    positions = population.positions.clone()
    fitness = population.fitness.clone()
    best_position = population.best_position.clone()
    best_fitness = population.best_fitness.clone()
    monkeypatch.setattr(torch, "rand", _constant_rand(0.25, population.dtype))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))
    function = Function(lambda position: position.sum())

    with pytest.raises(e.ValueError, match="finite non-negative values for HGSO"):
        optimizer.update(_context(population, function))

    assert torch.equal(population.positions, positions)
    assert torch.equal(population.fitness, fitness)
    assert torch.equal(population.best_position, best_position)
    assert torch.equal(population.best_fitness, best_fitness)


def test_lsa_probes_direction_and_uses_signed_exponential_steps(monkeypatch):
    population = _population(n_agents=2, n_variables=2)
    population.positions[0].zero_()
    population.positions[1, :, 0] = population.positions.new_tensor([-1.0, 1.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = LSA({"p_fork": 0.0})
    optimizer.compile(population)
    optimizer.direction.fill_(1)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, math.exp(-1)))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)

    optimizer.update(_context(population, Function(_sphere)))

    assert torch.equal(optimizer.direction, -torch.ones_like(optimizer.direction))
    assert torch.count_nonzero(population.positions[1]) == 0
    assert population.fitness[1].item() == 0


def test_lsa_archives_direction_probe_improvement(monkeypatch):
    population = _population(n_agents=2, lower=-10, upper=10)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.1, 2.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2)) + 1
    population.best_position.fill_(0.1)
    population.best_fitness.fill_(1.01)
    optimizer = LSA({"p_fork": 0.0})
    optimizer.compile(population)
    optimizer.direction.fill_(-1)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.ones_like(tensor))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)

    optimizer.update(_context(population, Function(_positive_sphere)))

    assert population.best_position.item() == pytest.approx(0.0)
    assert population.best_fitness.item() == pytest.approx(1.0)


def test_teo_applies_memory_environment_and_exchange_equations(monkeypatch):
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 2.0])
    population.fitness = population.fitness.new_tensor([1.0, 4.0])
    population.best_position.fill_(1)
    population.best_fitness.fill_(1)
    optimizer = TEO({"pro": 0.0, "n_TM": 1})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))

    optimizer.update(_context(population, Function(_sphere), iteration=5, n_iterations=10))

    hot = 0.5 + 0.5 * math.exp(-0.125)
    assert optimizer.environment[:, 0, 0].tolist() == pytest.approx([0.5, 0.25])
    assert population.positions[0].item() == pytest.approx(hot)
    assert population.positions[1].item() == pytest.approx(1.0)
    assert len(optimizer.TM) == 1
    optimizer.update(_context(population, Function(_sphere), iteration=6, n_iterations=10))
    assert len(optimizer.TM) == 1


def test_teo_zero_fitness_and_disabled_randomness_leave_positions_unchanged(monkeypatch):
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([2.0, 4.0])
    population.fitness.zero_()
    population.best_position.fill_(2)
    population.best_fitness.zero_()
    optimizer = TEO({"c1": False, "c2": False, "pro": 0.0})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(2 / 3, population.dtype))
    function = Function(lambda position: position.new_zeros(()))

    optimizer.update(_context(population, function, iteration=5, n_iterations=10))

    assert population.positions[:, 0, 0].tolist() == [2.0, 4.0]
    assert optimizer.environment[:, 0, 0].tolist() == [4.0, 2.0]


def test_teo_odd_population_keeps_median_with_self_environment(monkeypatch):
    population = _population(n_agents=3)
    population.positions[:, 0, 0] = population.positions.new_tensor([1.0, 2.0, 3.0])
    population.fitness.zero_()
    population.best_position.fill_(1)
    population.best_fitness.zero_()
    optimizer = TEO({"c1": False, "c2": False, "pro": 0.0})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))
    function = Function(lambda position: position.new_zeros(()))

    optimizer.update(_context(population, function))

    assert optimizer.environment[:, 0, 0].tolist() == [3.0, 2.0, 1.0]
    assert population.positions[:, 0, 0].tolist() == [1.0, 2.0, 3.0]


def test_teo_rejects_signed_fitness_before_mutation(monkeypatch):
    population = _population(n_agents=3)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.5, 1.0, 2.0])
    population.fitness = population.positions[:, 0, 0] - 1
    population.best_position.fill_(0.5)
    population.best_fitness.fill_(-0.5)
    optimizer = TEO({"n_TM": 1, "pro": 0.0})
    optimizer.compile(population)
    positions = population.positions.clone()
    environment = optimizer.environment.clone()
    monkeypatch.setattr(torch, "rand", _constant_rand(2 / 3, population.dtype))

    with pytest.raises(e.ValueError, match="finite non-negative values for TEO"):
        optimizer.update(_context(population, Function(lambda position: position.sum() - 1), iteration=5))

    assert torch.equal(population.positions, positions)
    assert torch.equal(optimizer.environment, environment)
    assert optimizer.TM == []


def test_two_restores_friction_and_directional_acceleration(monkeypatch):
    population = _population(n_agents=2, n_variables=2)
    population.positions[0, :, 0] = population.positions.new_tensor([2.0, 2.0])
    population.positions[1].zero_()
    population.fitness = population.fitness.new_tensor([8.0, 0.0])
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = TWO({"mu_s": 1.0, "mu_k": 1.0, "delta_t": 1.0, "alpha": 0.9, "beta": 0.05})
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    monkeypatch.setattr(torch, "rand", _constant_rand(1.0, population.dtype))

    optimizer.update(_context(population, Function(_sphere), iteration=0))

    assert torch.allclose(population.positions[0, :, 0], population.positions.new_tensor([1.0, 1.0]))
    assert optimizer.alpha == 0.9
    assert optimizer.beta == 0.05
    assert not hasattr(optimizer, "alpha_val")
    assert not hasattr(optimizer, "beta_val")


def test_weo_monolayer_probability_depends_on_fitness(monkeypatch):
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([2.0, 0.0])
    population.fitness = population.fitness.new_tensor([-2.0, 0.0])
    population.best_position.fill_(2)
    population.best_fitness.fill_(-2)
    optimizer = WEO()
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.1))
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: torch.tensor([0, 1], device=population.device))
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))

    optimizer.update(_context(population, Function(_negative_sum), iteration=1, n_iterations=10))

    assert population.positions[0].item() == 2
    assert population.positions[1].item() == 1


def test_weo_droplet_flux_matches_documented_equation(monkeypatch):
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([2.0, 0.0])
    population.fitness = population.fitness.new_tensor([-2.0, 0.0])
    population.best_position.fill_(2)
    population.best_fitness.fill_(-2)
    optimizer = WEO()
    optimizer.compile(population)
    theta = optimizer.theta_max
    cosine = math.cos(theta)
    expected_flux = (1 / 2.6) * (2 / 3 + cosine**3 / 3 - cosine) ** (-2 / 3) * (1 - cosine)
    best_cosine = math.cos(optimizer.theta_min)
    best_flux = (1 / 2.6) * (2 / 3 + best_cosine**3 / 3 - best_cosine) ** (-2 / 3) * (1 - best_cosine)
    threshold = (best_flux + expected_flux) / 2
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, threshold))
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: torch.tensor([0, 1], device=population.device))
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))

    optimizer.update(_context(population, Function(_negative_sum), iteration=10, n_iterations=10))

    assert population.positions[1].item() == 1


def test_wwo_propagation_scales_by_search_width(monkeypatch):
    population = _population(n_agents=2, lower=-100, upper=100)
    population.positions[:, 0, 0] = population.positions.new_tensor([50.0, 80.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.fill_(50)
    population.best_fitness.fill_(2500)
    optimizer = WWO({"beta": 0.0})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.25, population.dtype))
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: torch.tensor([0], device=population.device))

    optimizer.update(_context(population, Function(_sphere)))

    assert population.positions[0].item() == pytest.approx(0.0)
    assert population.best_position.item() == pytest.approx(0.0)


def test_wwo_breaking_uses_distinct_random_dimensions(monkeypatch):
    population = _population(n_agents=2, n_variables=3, lower=0, upper=10)
    population.positions[0, :, 0] = population.positions.new_tensor([5.0, 5.0, 5.0])
    population.positions[1, :, 0] = population.positions.new_tensor([8.0, 8.0, 8.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.copy_(population.positions[0])
    population.best_fitness.copy_(population.fitness[0])
    optimizer = WWO({"beta": 0.1, "k_max": 1})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.25, population.dtype))
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: torch.tensor([2, 0, 1], device=population.device))
    monkeypatch.setattr(torch, "randn_like", lambda tensor: -torch.ones_like(tensor))

    optimizer.update(_context(population, Function(_sphere)))

    assert population.positions[0, :, 0].tolist() == pytest.approx([2.5, 2.5, 1.5])
    assert population.best_position[:, 0].tolist() == pytest.approx([2.5, 2.5, 1.5])


def test_wwo_propagates_breaks_and_tracks_global_best(monkeypatch):
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([2.0, 3.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.fill_(2)
    population.best_fitness.fill_(4)
    optimizer = WWO({"beta": 0.0025})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.4975, population.dtype))
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: torch.tensor([0], device=population.device))
    monkeypatch.setattr(torch, "randn_like", lambda tensor: -torch.ones_like(tensor))

    optimizer.update(_context(population, Function(_sphere)))

    assert population.positions[0].item() == pytest.approx(1.0)
    assert population.best_position.item() == pytest.approx(1.0)
    assert population.best_fitness.item() == pytest.approx(1.0)
    assert optimizer.height[0].item() == optimizer.h_max


def test_wwo_refracts_exhausted_wave_and_updates_length(monkeypatch):
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.0, 4.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = WWO({"h_max": 1})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)

    optimizer.update(_context(population, Function(_sphere)))

    assert population.positions[1].item() == pytest.approx(2.0)
    assert population.fitness[1].item() == pytest.approx(4.0)
    assert optimizer.height[1].item() == 1
    assert optimizer.length[1].item() == pytest.approx(2.0, rel=1e-3)


def test_wwo_wavelength_ratios_are_scale_invariant(monkeypatch):
    results = []
    for scale in (1.0, 1e-12):
        population = _population(n_agents=2, lower=-10, upper=10)
        population.positions[:, 0, 0] = population.positions.new_tensor([0.0, 2.0])
        population.fitness = scale * (population.positions.square().sum(dim=(-1, -2)) + 1)
        population.best_position.zero_()
        population.best_fitness.fill_(scale)
        optimizer = WWO({"h_max": 1})
        optimizer.compile(population)
        monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))
        monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
        function = Function(lambda position, scale=scale: scale * (position.square().sum() + 1))

        optimizer.update(_context(population, function))
        results.append((population.positions.clone(), optimizer.length.clone()))

    assert torch.equal(results[0][0], results[1][0])
    assert torch.allclose(results[0][1], results[1][1])
    assert results[0][1].tolist() == pytest.approx([0.5 / 1.001, 1.25])


def test_wwo_zero_policy_preserves_nonzero_wavelength(monkeypatch):
    population = _population(n_agents=2, lower=-10, upper=10)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.0, 1.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = WWO({"h_max": 1})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_rand(0.5, population.dtype))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)

    optimizer.update(_context(population, Function(_sphere)))

    assert (optimizer.length > 0).all()
    assert optimizer.length[0].item() == pytest.approx(0.5 / 1.001)


@pytest.mark.parametrize(
    ("dtype", "large"),
    [
        (torch.float16, 50000.0),
        (torch.float32, 3e38),
        (torch.float64, 1e308),
    ],
)
def test_wwo_scaled_ratio_avoids_intermediate_overflow_and_underflow(dtype, large):
    large_value = torch.tensor(large, dtype=dtype)
    half = torch.tensor(0.5, dtype=dtype)

    overflow_prone = WWO._scaled_ratio(half, large_value, half)
    underflow_prone = WWO._scaled_ratio(large_value, half, large_value)

    assert overflow_prone.dtype == dtype
    assert underflow_prone.dtype == dtype
    assert torch.equal(overflow_prone, large_value)
    assert torch.equal(underflow_prone, half)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_wwo_scaled_ratio_rejects_unrepresentable_wavelength(dtype):
    maximum = torch.tensor(torch.finfo(dtype).max, dtype=dtype)
    minimum = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype)
    half = torch.tensor(0.5, dtype=dtype)

    with pytest.raises(e.ValueError, match="must be representable"):
        WWO._scaled_ratio(maximum, maximum, half)
    with pytest.raises(e.ValueError, match="must be representable"):
        WWO._scaled_ratio(minimum, minimum, maximum)


def test_wwo_rejects_signed_fitness_before_mutation():
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.0, 1.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2)) - 1
    population.best_position.zero_()
    population.best_fitness.fill_(-1)
    optimizer = WWO()
    optimizer.compile(population)
    positions = population.positions.clone()
    lengths = optimizer.length.clone()

    with pytest.raises(e.ValueError, match="finite non-negative values for WWO"):
        optimizer.update(_context(population, Function(lambda position: position.square().sum() - 1)))

    assert torch.equal(population.positions, positions)
    assert torch.equal(optimizer.length, lengths)


def test_wwo_state_is_device_local_and_dtype_preserving():
    population = _population(n_agents=3, n_variables=2, n_dimensions=2)
    optimizer = WWO()
    optimizer.compile(population)

    assert optimizer.height.shape == (3,)
    assert optimizer.height.device == population.device
    assert optimizer.length.shape == (3,)
    assert optimizer.length.dtype == population.dtype


def test_science_migration_population_and_parameter_edges():
    with pytest.raises(e.SizeError, match="at least 3 for CDO"):
        CDO().compile(_population(n_agents=2))
    with pytest.raises(e.ValueError, match="non-empty neutral field"):
        EFO({"positive_field": 0.75, "negative_field": 0.2}).compile(_population(n_agents=4))
    with pytest.raises(e.SizeError, match="at least 2 for WEO"):
        WEO().compile(_population(n_agents=1))
    with pytest.raises(e.ValueError, match="`mu_k` must be positive"):
        TWO({"mu_k": 0})
    with pytest.raises(e.TypeError, match="`c1` must be a bool"):
        TEO({"c1": 1})
    with pytest.raises(e.ValueError, match="`alpha` must be positive"):
        WWO({"alpha": 0})
