# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Current-upstream parity regressions for evolutionary and population optimizers."""

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest
import torch

from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.space import Space
from otorchmizer.optimizers.evolutionary import (
    BSA,
    DE,
    EP,
    ES,
    FOA,
    GA,
    GP,
    GSGP,
    HS,
    IWO,
    NGHS,
    RRA,
    SGHS,
)
from otorchmizer.optimizers.population import AEO, AO, COA, EPO, GCO, LOA, OSA, PVS, RFO


def _sphere(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum()


def _space(
    n_agents: int,
    n_variables: int = 1,
    lower: float = -10,
    upper: float = 10,
) -> Space:
    space = Space(
        n_agents=n_agents,
        n_variables=n_variables,
        lower_bound=[lower] * n_variables,
        upper_bound=[upper] * n_variables,
    )
    space.population.initialize_uniform()
    return space


def _context(
    space: Space,
    function: Callable,
    iteration: int = 0,
    n_iterations: int = 10,
) -> UpdateContext:
    return UpdateContext(
        space=space,
        function=function,
        iteration=iteration,
        n_iterations=n_iterations,
        device=space.device,
    )


def _queued_factory(values: list[float]) -> Callable:
    iterator = iter(values)

    def factory(*shape, device=None, dtype=None, **_kwargs):
        return torch.full(shape, next(iterator), device=device, dtype=dtype)

    return factory


def test_aeo_producer_scales_the_complete_uniform_sample(monkeypatch):
    space = _space(1, lower=-10, upper=10)
    space.population.positions.fill_(2)
    space.population.best_position.fill_(2)
    space.population.fitness.fill_(torch.inf)
    candidates = []

    def capture(positions):
        candidates.append(positions.clone())
        return positions.new_zeros(positions.shape[0])

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.full(shape, 0.5, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )

    AEO().update(_context(space, capture))

    assert candidates[0].item() == pytest.approx(1.0)


def test_aeo_consumers_only_sample_preceding_consumers(monkeypatch):
    space = _space(4)
    space.population.positions[:, 0, 0] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    space.population.fitness[:] = torch.tensor([16.0, 9.0, 4.0, 1.0])
    space.population.best_position.fill_(1)
    sampled_highs = []

    def randint(low, high, shape, device=None):
        if shape == (1,):
            sampled_highs.append(high)
        return torch.full(shape, low, dtype=torch.long, device=device)

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.full(shape, 0.9, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, device=None, dtype=None: torch.ones(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randint", randint)

    AEO().update(_context(space, lambda positions: positions.new_zeros(positions.shape[0])))

    assert sampled_highs == [2, 3]


def test_epo_temperature_flag_and_c_are_per_source_equations(monkeypatch):
    space = _space(1, n_variables=2)
    space.population.positions.fill_(1)
    space.population.best_position.fill_(2)
    sampled_shapes = []
    values = iter([0.75, 0.25, 0.5])

    def random(*shape, device=None, dtype=None):
        sampled_shapes.append(shape)
        return torch.full(shape, next(values), device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", random)

    EPO().update(_context(space, Function(_sphere), iteration=1))

    assert sampled_shapes == [(1, 1, 1), (1, 1, 1), (1, 2, 1)]
    temperature = 10 / 9
    avoidance = 2 * (temperature + 1) * 0.25 - temperature
    social = abs(2 * torch.exp(torch.tensor(-1 / 1.5)) - torch.exp(torch.tensor(-1))) ** 2
    expected = 2 - avoidance * abs(social * 2 - 0.5)
    assert torch.allclose(space.population.positions, torch.full_like(space.population.positions, expected))


def test_sghs_uses_its_signed_best_based_generator(monkeypatch):
    space = _space(2)
    space.population.positions[:, 0, 0] = torch.tensor([0.0, 10.0])
    space.population.fitness[:] = torch.tensor([0.0, 100.0])
    space.population.best_position.zero_()
    optimizer = SGHS({"HMCR": 1.0, "PAR": 0.0, "bw": 1.0})

    monkeypatch.setattr(torch, "rand", _queued_factory([0.1, 0.75, 0.9]))
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *_args, **_kwargs: pytest.fail("SGHS must not sample a memory index."),
    )

    harmony = optimizer._generate_new_harmony(space.population, space.device)

    assert harmony.item() == pytest.approx(0.5)


def test_nghs_unconditionally_replaces_worst_but_preserves_archive(monkeypatch):
    space = _space(2)
    space.population.positions[:, 0, 0] = torch.tensor([0.0, 1.0])
    space.population.fitness[:] = torch.tensor([0.0, 1.0])
    space.population.best_position.zero_()
    space.population.best_fitness.zero_()
    optimizer = NGHS({"pm": 1.0})

    monkeypatch.setattr(torch, "rand", _queued_factory([0.5, 0.0, 0.9]))
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.zeros(shape, dtype=torch.long, device=device),
    )

    optimizer.update(_context(space, lambda positions: (positions**2).sum(dim=(-1, -2))))

    assert space.population.positions[1].item() == pytest.approx(8.0)
    assert space.population.fitness[1].item() == pytest.approx(64.0)
    assert space.population.best_position.item() == 0
    assert space.population.best_fitness.item() == 0


def test_rfo_defaults_are_seeded_random_source_parameters():
    torch.manual_seed(31)
    expected_phi = torch.rand(1).item() * math.tau
    expected_theta = torch.rand(1).item()

    torch.manual_seed(31)
    optimizer = RFO()

    assert optimizer.phi == pytest.approx(expected_phi)
    assert optimizer.theta == pytest.approx(expected_theta)


def test_rfo_noticing_uses_angular_equation(monkeypatch):
    space = _space(1, n_variables=3, lower=-5, upper=5)
    space.population.positions.zero_()
    space.population.fitness.zero_()
    optimizer = RFO({"phi": 1.0, "theta": 0.5, "p_replacement": 0.0})
    optimizer.compile(space.population)

    monkeypatch.setattr(torch, "rand", _queued_factory([0.5, 0.0, 1.0, 0.0]))

    optimizer.update(_context(space, lambda positions: -positions.sum(dim=(-1, -2))))

    expected = 0.01 * math.sin(1.0)
    assert torch.allclose(space.population.positions, torch.full_like(space.population.positions, expected))


def test_rfo_habitat_randomization_respects_population_bounds(monkeypatch):
    space = _space(2, lower=-5, upper=-3)
    space.population.initialize_static(torch.tensor([[-5.0], [-4.0]]))
    function = Function(_sphere)
    optimizer = RFO({"p_replacement": 0.5})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(torch, "rand", _queued_factory([0, 0, 0, 0, 0, 1]))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))

    optimizer.update(_context(space, function))

    assert space.population.positions[0].item() == -5
    assert space.population.fitness[0].item() == 25


def test_gco_requires_three_distinct_weighted_donors(monkeypatch):
    space = _space(3)
    space.population.fitness[:] = torch.tensor([1.0, 4.0, 9.0])
    optimizer = GCO({"CR": 1.0})
    optimizer.compile(space.population)
    calls = []

    def multinomial(probabilities, n, replacement):
        calls.append((n, replacement))
        return torch.arange(3)

    monkeypatch.setattr(torch, "multinomial", multinomial)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )

    optimizer.update(_context(space, lambda positions: (positions**2).sum(dim=(-1, -2))))

    assert calls == [(3, False)] * 3


def test_gco_rejects_invalid_population_and_negative_scale():
    with pytest.raises(ValueError, match="at least 3"):
        GCO().compile(_space(2).population)

    with pytest.raises(ValueError, match="non-negative"):
        GCO({"F": -1})


def test_loa_selects_lowest_fitness_hunting_group_and_ignores_empty_groups(monkeypatch):
    space = _space(8)
    space.population.positions[:, 0, 0] = torch.arange(8)
    optimizer = LOA({"P": 1, "N": 0.25, "S": 0.8})
    optimizer.compile(space.population)
    pride_members = (optimizer.pride == 0).nonzero(as_tuple=True)[0]
    females = pride_members[optimizer.female[pride_members]]
    assignments = torch.tensor([1, 1, 2, 2, 3])
    group_fitness = space.population.fitness.new_tensor([0.4, 0.6, 4.0, 6.0, 5.0])
    space.population.fitness.fill_(100)
    space.population.fitness[females] = group_fitness
    lions = optimizer._make_lions(space.population)
    center_positions = []

    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: assignments.to(device),
    )

    def candidate(position, _prey, center):
        if center:
            center_positions.append(position.item())
        return position.clone()

    monkeypatch.setattr(optimizer, "_hunting_candidate", candidate)
    optimizer._hunting(lions, space.population, Function(_sphere))

    expected = set(space.population.positions[females[:2]].flatten().tolist())
    assert set(center_positions) == expected

    center_positions.clear()
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.ones(shape, dtype=torch.long, device=device),
    )
    optimizer._hunting(lions, space.population, Function(_sphere))

    assert len(center_positions) == females.numel()


def test_pvs_rejects_insufficient_peers_before_mutation():
    space = _space(2)
    original = space.population.positions.clone()

    with pytest.raises(ValueError, match="at least 3"):
        PVS().update(_context(space, Function(_sphere)))

    assert torch.equal(space.population.positions, original)


def test_iwo_exposes_sigma_and_validates_coupled_parameters():
    optimizer = IWO()
    assert optimizer.sigma == 0

    with pytest.raises(ValueError, match="max_seeds"):
        IWO({"min_seeds": 5, "max_seeds": 4})
    with pytest.raises(ValueError, match="init_sigma"):
        IWO({"final_sigma": 2.0, "init_sigma": 1.0})

    space = _space(3)
    function = Function(_sphere)
    optimizer.evaluate(space.population, function)
    optimizer.update(_context(space, function, iteration=1))

    expected = ((10 - 1) ** optimizer.e / 10**optimizer.e) * (
        optimizer.init_sigma - optimizer.final_sigma
    ) + optimizer.final_sigma
    assert optimizer.sigma == pytest.approx(expected)


@pytest.mark.parametrize(
    "params",
    [
        {"init_sigma": 1e-4, "final_sigma": 1e-5},
        {"final_sigma": 1e-5, "init_sigma": 1e-4},
        {"min_seeds": 3, "max_seeds": 4},
        {"max_seeds": 4, "min_seeds": 3},
    ],
)
def test_iwo_coupled_parameters_are_build_order_independent(params):
    optimizer = IWO(params)

    for name, value in params.items():
        assert getattr(optimizer, name) == value


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"min_seeds": 5, "max_seeds": 4}, "max_seeds"),
        ({"init_sigma": 1.0, "final_sigma": 2.0}, "init_sigma"),
    ],
)
def test_iwo_build_rejects_invalid_pairs_without_committing(params, message):
    optimizer = IWO()
    previous = (
        optimizer.min_seeds,
        optimizer.max_seeds,
        optimizer.init_sigma,
        optimizer.final_sigma,
    )

    with pytest.raises(ValueError, match=message):
        optimizer.build(params)

    assert (
        optimizer.min_seeds,
        optimizer.max_seeds,
        optimizer.init_sigma,
        optimizer.final_sigma,
    ) == previous


def test_iwo_direct_reassignment_preserves_coupled_invariants():
    optimizer = IWO()

    with pytest.raises(ValueError, match="min_seeds"):
        optimizer.min_seeds = optimizer.max_seeds + 1
    with pytest.raises(ValueError, match="final_sigma"):
        optimizer.final_sigma = optimizer.init_sigma + 1
    with pytest.raises(ValueError, match="init_sigma"):
        optimizer.init_sigma = optimizer.final_sigma / 2


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("_max_seeds", -1, "max_seeds"),
        ("_init_sigma", 0.0, "init_sigma"),
    ],
)
def test_iwo_update_revalidates_coupled_state(attribute, value, message):
    optimizer = IWO()
    setattr(optimizer, attribute, value)
    space = _space(3)
    function = Function(_sphere)
    optimizer.evaluate(space.population, function)

    with pytest.raises(ValueError, match=message):
        optimizer.update(_context(space, function))


def test_iwo_valid_small_sigma_pair_updates():
    optimizer = IWO({"init_sigma": 1e-4, "final_sigma": 1e-5})
    space = _space(3)
    function = Function(_sphere)
    optimizer.evaluate(space.population, function)

    optimizer.update(_context(space, function, iteration=1))

    assert optimizer.final_sigma <= optimizer.sigma <= optimizer.init_sigma


def test_fraction_parameter_is_normalized_before_tensor_math():
    space = _space(4)
    function = Function(_sphere)
    optimizer = BSA({"F": Fraction(1, 2)})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    optimizer.update(_context(space, function))

    assert type(optimizer.F) is float
    assert optimizer.F == 0.5


def test_loa_all_hunters_in_one_group_cannot_select_an_empty_group(monkeypatch):
    space = _space(8)
    optimizer = LOA({"P": 1, "N": 0.25, "S": 0.8})
    optimizer.compile(space.population)
    pride_members = (optimizer.pride == 0).nonzero(as_tuple=True)[0]
    females = pride_members[optimizer.female[pride_members]]
    space.population.fitness.fill_(0)
    space.population.fitness[females] = 1e38
    lions = optimizer._make_lions(space.population)
    center_positions = []

    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.full(
            shape,
            2,
            dtype=torch.long,
            device=device,
        ),
    )

    def candidate(position, _prey, center):
        if center:
            center_positions.append(position.clone())
        return position.clone()

    monkeypatch.setattr(optimizer, "_hunting_candidate", candidate)
    optimizer._hunting(lions, space.population, Function(lambda x: x.new_tensor(1e38)))

    assert len(center_positions) == females.numel()


def test_loa_group_ranking_is_scale_safe_for_large_finite_fitness(monkeypatch):
    space = _space(8)
    optimizer = LOA({"P": 1, "N": 0.25, "S": 0.8})
    optimizer.compile(space.population)
    pride_members = (optimizer.pride == 0).nonzero(as_tuple=True)[0]
    females = pride_members[optimizer.female[pride_members]]
    assignments = torch.tensor([1, 1, 1, 2, 2])
    space.population.fitness.fill_(0)
    space.population.fitness[females] = space.population.fitness.new_tensor([1.2e38, 1.2e38, 1.2e38, 2e38, 2e38])
    lions = optimizer._make_lions(space.population)
    center_positions = []

    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, _shape, device=None: assignments.to(device),
    )

    def candidate(position, _prey, center):
        if center:
            center_positions.append(position.item())
        return position.clone()

    monkeypatch.setattr(optimizer, "_hunting_candidate", candidate)
    optimizer._hunting(lions, space.population, Function(lambda x: x.new_tensor(1e38)))

    expected = set(space.population.positions[females[:3]].flatten().tolist())
    assert set(center_positions) == expected


@pytest.mark.parametrize(
    ("optimizer_type", "params"),
    [
        (BSA, {"F": np.float32(1.0), "mix_rate": np.int64(1)}),
        (DE, {"CR": np.float32(0.5), "F": np.float32(1.0)}),
        (EP, {"bout_size": np.float32(0.5), "clip_ratio": np.float32(0.1)}),
        (ES, {"child_ratio": np.float32(0.5)}),
        (FOA, {"area_limit": np.int64(3), "transfer_rate": np.float32(0.5)}),
        (GA, {"p_selection": np.float32(0.5)}),
        (GP, {"p_mutation": np.float32(0.5)}),
        (GSGP, {"mutation_step": np.float32(0.5)}),
        (HS, {"HMCR": np.float32(0.5)}),
        (IWO, {"min_seeds": np.int64(1), "max_seeds": np.int64(2)}),
        (RRA, {"max_stall": np.int64(2)}),
        (AO, {"n_cycles": np.int64(2), "alpha": np.float32(0.5)}),
        (COA, {"n_p": np.int64(1)}),
        (EPO, {"f": np.float32(2.0)}),
        (GCO, {"F": np.float32(1.0)}),
        (LOA, {"P": np.int64(1), "N": np.float32(0.5)}),
        (OSA, {"beta": np.float32(1.0)}),
        (RFO, {"phi": np.float32(1.0), "theta": np.float32(0.5)}),
    ],
)
def test_optimizer_parameters_accept_valid_numpy_scalars(optimizer_type, params):
    optimizer = optimizer_type(params)

    for name, value in params.items():
        assert getattr(optimizer, name) == value
        expected_type = int if isinstance(value, np.integer) else float
        assert type(getattr(optimizer, name)) is expected_type
