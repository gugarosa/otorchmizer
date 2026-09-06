# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Regression tests for evolutionary and population optimizer contracts."""

from collections.abc import Callable

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
    GHS,
    GOGHS,
    HS,
    IHS,
    IWO,
    NGHS,
    RRA,
    SGHS,
)
from otorchmizer.optimizers.evolutionary.de import _sample_excluding
from otorchmizer.optimizers.population import AEO, AO, COA, EPO, GCO, GWO, HHO, OSA, PPA, PVS, RFO


def _sphere(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum()


def _space(
    n_agents: int = 6,
    n_variables: int = 3,
    lower: float = -5,
    upper: float = 5,
    dtype: torch.dtype = torch.float32,
) -> Space:
    space = Space(
        n_agents=n_agents,
        n_variables=n_variables,
        lower_bound=[lower] * n_variables,
        upper_bound=[upper] * n_variables,
    )
    space.population.to(torch.device("cpu"), dtype)
    space.population.initialize_uniform()
    return space


def _context(
    space: Space,
    function: Function,
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


def _constant_rand(values: list[float]) -> Callable:
    iterator = iter(values)

    def sample(*size, device=None, dtype=None, **_kwargs):
        return torch.full(size, next(iterator), device=device, dtype=dtype)

    return sample


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize(
    "optimizer_type",
    [
        BSA,
        DE,
        EP,
        ES,
        FOA,
        GA,
        HS,
        IHS,
        GHS,
        SGHS,
        NGHS,
        GOGHS,
        IWO,
        RRA,
        AEO,
        AO,
        COA,
        EPO,
        GCO,
        GWO,
        HHO,
        OSA,
        PPA,
        PVS,
        RFO,
    ],
)
def test_updates_preserve_population_dtype(dtype, optimizer_type):
    torch.manual_seed(7)
    function = Function(_sphere)
    space = _space(dtype=dtype)
    optimizer = optimizer_type()
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    optimizer.update(_context(space, function))

    assert space.population.positions.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize(
    ("optimizer_type", "state_names"),
    [
        (BSA, ("old_positions",)),
        (EP, ("strategy",)),
        (ES, ("strategy",)),
        (GCO, ("life", "counter")),
        (PPA, ("velocity",)),
    ],
)
def test_compiled_floating_state_matches_population_dtype(dtype, optimizer_type, state_names):
    space = _space(dtype=dtype)
    optimizer = optimizer_type()

    optimizer.compile(space.population)

    for state_name in state_names:
        assert getattr(optimizer, state_name).dtype == dtype


def test_de_samples_three_distinct_non_target_donors():
    torch.manual_seed(4)
    n_agents = 128
    targets = torch.arange(n_agents)

    first = _sample_excluding(targets.unsqueeze(1), n_agents)
    second = _sample_excluding(torch.stack((targets, first), dim=1), n_agents)
    third = _sample_excluding(torch.stack((targets, first, second), dim=1), n_agents)

    assert torch.all(first != targets)
    assert torch.all(second != targets)
    assert torch.all(third != targets)
    assert torch.all(first != second)
    assert torch.all(first != third)
    assert torch.all(second != third)


def test_de_requires_four_agents():
    optimizer = DE()
    space = _space(n_agents=3)

    with pytest.raises(ValueError, match="at least 4"):
        optimizer.compile(space.population)


@pytest.mark.parametrize(
    ("random_values", "expected_changed"),
    [
        ([1, 1, 1, 0], 1),
        ([1, 1, 0, 1, 0.5, 0.5], 2),
    ],
)
def test_bsa_crossover_preserves_nonselected_variables(monkeypatch, random_values, expected_changed):
    space = _space(n_agents=2, n_variables=4, lower=-10, upper=10)
    space.population.positions.zero_()
    space.population.fitness.fill_(100)
    optimizer = BSA({"F": 1.0, "mix_rate": 1})
    optimizer.compile(space.population)
    optimizer.old_positions.fill_(1)
    function = Function(_sphere)

    monkeypatch.setattr(torch, "rand", _constant_rand(random_values))
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, size, device=None: torch.zeros(size, dtype=torch.long, device=device),
    )

    optimizer.update(_context(space, function))

    changed = (space.population.positions.squeeze(-1) != 0).sum(dim=1)
    assert torch.equal(changed, torch.full_like(changed, expected_changed))


def test_es_initializes_strategies_for_every_selectable_parent():
    torch.manual_seed(3)
    space = _space(n_agents=4, n_variables=1, lower=-10, upper=10)
    optimizer = ES({"child_ratio": 0.5})

    optimizer.compile(space.population)

    assert torch.all(optimizer.strategy > 0)


def test_ga_caps_odd_population_selection_at_an_even_cardinality():
    torch.manual_seed(2)
    function = Function(_sphere)
    space = _space(n_agents=3)
    optimizer = GA({"p_selection": 1.0})
    optimizer.evaluate(space.population, function)

    selected = optimizer._roulette_selection(space.population)

    assert selected.shape == (2,)
    assert selected.unique().numel() == 2


def test_hs_pitch_adjustment_is_signed():
    torch.manual_seed(5)
    space = _space(n_agents=5, n_variables=2, lower=-10, upper=10)
    space.population.positions.zero_()
    optimizer = HS({"HMCR": 1.0, "PAR": 1.0, "bw": 1.0})

    harmonies = torch.stack(
        [optimizer._generate_new_harmony(space.population, space.population.device) for _ in range(64)]
    )

    assert harmonies.min() < 0
    assert harmonies.max() > 0


def test_nghs_reflection_interval_midpoint_is_the_best(monkeypatch):
    space = _space(n_agents=2, n_variables=1, lower=-100, upper=100)
    space.population.positions[:, 0, 0] = torch.tensor([2.0, 10.0])
    space.population.fitness[:] = torch.tensor([4.0, 100.0])
    space.population.best_position[:] = torch.tensor([[2.0]])
    optimizer = NGHS({"pm": 0.0})

    monkeypatch.setattr(torch, "rand", _constant_rand([0.5, 0.5]))

    harmony = optimizer._generate_new_harmony(space.population, space.population.device)

    # This midpoint invariant corrects the reflected-endpoint defect inherited from Opytimizer 3.1.4
    assert harmony.item() == pytest.approx(space.population.best_position.item())


def test_epo_final_zero_based_iteration_is_nonsingular(monkeypatch):
    space = _space(n_agents=1, n_variables=1)
    space.population.positions.fill_(1)
    space.population.best_position.fill_(2)
    optimizer = EPO()

    monkeypatch.setattr(torch, "rand", _constant_rand([0.25, 0.25, 0.25]))

    optimizer.update(_context(space, Function(_sphere), iteration=9, n_iterations=10))

    t_profile = 11.0
    avoidance = 2 * (t_profile + 1) * 0.25 - t_profile
    social_force = abs(optimizer.f * torch.exp(torch.tensor(-9 / optimizer.l)) - torch.exp(torch.tensor(-9))) ** 2
    distance = abs(social_force * 2 - 0.25)
    expected = 2 - avoidance * distance
    assert torch.isfinite(space.population.positions).all()
    assert space.population.positions.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("optimizer_type", [AO, PVS])
def test_greedy_population_updates_do_not_worsen_agents(optimizer_type):
    torch.manual_seed(4)
    function = Function(_sphere)
    space = _space(n_agents=12)
    optimizer = optimizer_type()
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)
    old_fitness = (
        torch.sort(space.population.fitness).values if optimizer_type is PVS else space.population.fitness.clone()
    )

    optimizer.update(_context(space, function, iteration=3))

    assert torch.all(space.population.fitness <= old_fitness)
    assert torch.allclose(space.population.fitness, function(space.population.positions))


def test_rfo_rejects_worse_relocation_and_noticing_candidates():
    torch.manual_seed(0)

    def double_well(x):
        return ((x**2 - 1) ** 2).sum()

    function = Function(double_well)
    space = _space(n_agents=4, n_variables=1, lower=-2, upper=2)
    initial = torch.tensor([[-1.0], [1.0], [1.0], [1.0]])
    space.population.initialize_static(initial)
    optimizer = RFO({"p_replacement": 0.0})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    optimizer.update(_context(space, function))

    assert optimizer.n_replacement == 0
    assert torch.equal(space.population.positions.squeeze(-1), initial)
    assert torch.equal(space.population.fitness, torch.zeros(4))


def test_rfo_refreshes_fitness_before_habitat_ranking(monkeypatch):
    function = Function(_sphere)
    space = _space(n_agents=3, n_variables=1, lower=-5, upper=5)
    space.population.initialize_static(torch.tensor([[0.0], [3.0], [2.0]]))
    optimizer = RFO({"p_replacement": 1 / 3})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(torch, "rand", _constant_rand([0, 0, 0, 1, 0, 0, 0, 0]))

    optimizer.update(_context(space, function))

    assert space.population.positions[1].item() == pytest.approx(3 - 3**0.5)
    assert space.population.positions[2].item() == 0
    assert torch.allclose(space.population.fitness, function(space.population.positions))


def test_rfo_full_replacement_retains_best_and_uses_frozen_habitat(monkeypatch):
    function = Function(_sphere)
    space = _space(n_agents=3, n_variables=1, lower=-5, upper=5)
    space.population.initialize_static(torch.tensor([[1.0], [1.25], [1.5]]))
    optimizer = RFO({"p_replacement": 1.0})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(torch, "rand", _constant_rand([0, 0, 0, 1, 0, 0, 0, 1, 1, 1]))
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)

    optimizer.update(_context(space, function))

    assert optimizer.n_replacement == space.population.n_agents
    torch.testing.assert_close(space.population.positions, torch.full_like(space.population.positions, -3.875))
    assert space.population.best_position.item() == 0.75
    assert space.population.best_fitness.item() == 0.75**2
    assert torch.allclose(space.population.fitness, function(space.population.positions))


def test_rfo_does_not_restore_an_elite_worse_than_all_replacements(monkeypatch):
    function = Function(_sphere)
    space = _space(n_agents=3, n_variables=1, lower=-5, upper=5)
    space.population.initialize_static(torch.tensor([[1.0], [2.0], [3.0]]))
    optimizer = RFO({"p_replacement": 1.0})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(torch, "rand", _constant_rand([0] * 10))
    optimizer.update(_context(space, function))

    assert torch.count_nonzero(space.population.positions) == 0
    assert torch.count_nonzero(space.population.fitness) == 0
    assert space.population.best_fitness.item() == 0


def test_gwo_requires_three_agents():
    optimizer = GWO()
    space = _space(n_agents=2)

    with pytest.raises(ValueError, match="at least 3"):
        optimizer.compile(space.population)


def test_coa_rejects_more_packs_than_agents():
    optimizer = COA({"n_p": 3})
    space = _space(n_agents=2)

    with pytest.raises(ValueError, match="must not exceed"):
        optimizer.compile(space.population)


def test_osa_uses_uniform_noise_and_half_range_alpha(monkeypatch):
    function = Function(_sphere)
    space = _space(n_agents=2, n_variables=1, lower=-5, upper=5)
    space.population.initialize_static(torch.tensor([[0.0], [2.0]]))
    optimizer = OSA({"beta": 1.0})
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(torch, "rand", _constant_rand([0, 0, 0, 0.25, 1, 0.25]))
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *_args, **_kwargs: pytest.fail("OSA must use uniform noise."),
    )

    optimizer.update(_context(space, function, n_iterations=2))

    assert space.population.positions[1].item() == pytest.approx(2.5)
