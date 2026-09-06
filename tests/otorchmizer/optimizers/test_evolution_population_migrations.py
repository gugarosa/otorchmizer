# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Equation and state regressions for completed evolutionary and population migrations."""

from collections.abc import Callable

import pytest
import torch

import otorchmizer.math.distribution as distribution
import otorchmizer.math.general as general
import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.space import Space
from otorchmizer.optimizers.evolutionary import EP, FOA, RRA
from otorchmizer.optimizers.population import AO, PPA


def _sphere(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum()


def _space(
    n_agents: int,
    n_variables: int = 1,
    n_dimensions: int = 1,
    lower: float = -10,
    upper: float = 10,
    dtype: torch.dtype = torch.float32,
) -> Space:
    space = Space(
        n_agents=n_agents,
        n_variables=n_variables,
        n_dimensions=n_dimensions,
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


def _queued_factory(values: list[float]) -> Callable:
    iterator = iter(values)

    def factory(*shape, device=None, dtype=None, **_kwargs):
        return torch.full(shape, next(iterator), device=device, dtype=dtype)

    return factory


def _ones_distribution(**kwargs) -> torch.Tensor:
    return torch.ones(
        kwargs["size"],
        device=kwargs.get("device"),
        dtype=kwargs.get("dtype"),
    )


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("alpha", -1.0, e.ValueError),
        ("delta", -1.0, e.ValueError),
        ("n_cycles", 0, e.ValueError),
        ("U", -1.0, e.ValueError),
        ("w", -1.0, e.ValueError),
    ],
)
def test_ao_canonical_parameter_validation(name, value, error):
    optimizer = AO()

    with pytest.raises(error):
        setattr(optimizer, name, value)


def test_ao_restores_canonical_source_parameters():
    optimizer = AO()

    assert optimizer.n_cycles == 10
    assert optimizer.U == pytest.approx(0.00565)
    assert optimizer.w == pytest.approx(0.005)


def test_ao_strategy_one_equation(monkeypatch):
    space = _space(1, n_variables=2, lower=-100, upper=100)
    space.population.initialize_static(torch.tensor([[2.0, 4.0]]))
    space.population.best_position = torch.tensor([[1.0], [3.0]])
    space.population.fitness.fill_(torch.inf)
    optimizer = AO()

    monkeypatch.setattr(torch, "rand", _queued_factory([0.25, 0.4]))
    monkeypatch.setattr(distribution, "generate_levy_distribution", _ones_distribution)

    optimizer.update(_context(space, Function(_sphere), iteration=2))

    expected = space.population.best_position * 0.8 + (
        torch.tensor([[2.0], [4.0]]) - space.population.best_position * 0.4
    )
    assert torch.allclose(space.population.positions[0], expected)


def test_ao_strategy_two_uses_canonical_spiral(monkeypatch):
    space = _space(1, n_variables=2, lower=-100, upper=100)
    initial = torch.tensor([[2.0], [4.0]])
    space.population.initialize_static(initial.squeeze(-1).unsqueeze(0).clone())
    space.population.best_position = torch.tensor([[1.0], [3.0]])
    space.population.fitness.fill_(torch.inf)
    optimizer = AO({"n_cycles": 4, "U": 0.5, "w": 0.25})

    monkeypatch.setattr(torch, "rand", _queued_factory([0.75, 0.5]))
    monkeypatch.setattr(torch, "randint", lambda *_args, **kwargs: torch.zeros((1,), dtype=torch.long))
    monkeypatch.setattr(distribution, "generate_levy_distribution", _ones_distribution)

    optimizer.update(_context(space, Function(_sphere), iteration=2))

    variable = torch.tensor([[1.0], [2.0]])
    cycle = optimizer.n_cycles + optimizer.U * variable
    theta = -optimizer.w * variable + 3 * torch.pi / 2
    expected = space.population.best_position + initial + cycle * (torch.cos(theta) - torch.sin(theta)) * 0.5
    assert torch.allclose(space.population.positions[0], expected)


def test_ao_strategy_three_equation(monkeypatch):
    space = _space(1, n_variables=2, lower=-100, upper=100)
    initial = torch.tensor([[2.0], [4.0]])
    space.population.initialize_static(initial.squeeze(-1).unsqueeze(0).clone())
    space.population.best_position = torch.tensor([[1.0], [3.0]])
    space.population.fitness.fill_(torch.inf)
    optimizer = AO({"alpha": 0.2, "delta": 0.3})

    monkeypatch.setattr(torch, "rand", _queued_factory([0.9, 0.25]))

    optimizer.update(_context(space, Function(_sphere), iteration=8))

    expected = (
        (space.population.best_position - initial) * optimizer.alpha - 0.25 + ((200 * 0.25 - 100) * optimizer.delta)
    )
    assert torch.allclose(space.population.positions[0], expected)


def test_ao_strategy_four_equation(monkeypatch):
    space = _space(1, n_variables=2, lower=-100, upper=100)
    initial = torch.tensor([[2.0], [4.0]])
    space.population.initialize_static(initial.squeeze(-1).unsqueeze(0).clone())
    space.population.best_position = torch.tensor([[1.0], [3.0]])
    space.population.fitness.fill_(torch.inf)
    optimizer = AO()

    monkeypatch.setattr(torch, "rand", _queued_factory([0.1, 0.75]))
    monkeypatch.setattr(distribution, "generate_levy_distribution", _ones_distribution)

    optimizer.update(_context(space, Function(_sphere), iteration=8))

    g1 = 0.5
    g2 = 0.4
    quality = 8 ** (g1 / 81)
    expected = quality * space.population.best_position - g1 * initial * 0.75 - g2 + 0.75 * g1
    assert torch.allclose(space.population.positions[0], expected)


def test_ep_children_use_parent_strategy_and_inherit_adapted_strategy(monkeypatch):
    space = _space(2, lower=-10, upper=10)
    space.population.initialize_static(torch.tensor([[5.0], [5.0]]))
    space.population.fitness[:] = 25
    optimizer = EP({"bout_size": 1.0, "clip_ratio": 1.0})
    optimizer.compile(space.population)
    optimizer.strategy.fill_(1)
    tau_local = 1 / 2**0.5
    normal_values = iter(
        [
            torch.full_like(optimizer.strategy, torch.log(torch.tensor(2.0)) / tau_local),
            -torch.ones_like(optimizer.strategy),
        ]
    )

    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda _tensor: next(normal_values))
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.zeros(shape, dtype=torch.long, device=device),
    )

    optimizer.update(_context(space, Function(_sphere)))

    assert torch.allclose(space.population.positions, torch.full_like(space.population.positions, 4))
    assert torch.allclose(optimizer.strategy, torch.full_like(optimizer.strategy, 2))


def test_ep_archives_best_evaluated_child_dropped_by_tournament(monkeypatch):
    space = _space(2, lower=-10, upper=10)
    space.population.initialize_static(torch.tensor([[3.0], [4.0]]))
    function = Function(_sphere)
    optimizer = EP({"bout_size": 0.5, "clip_ratio": 1.0})
    optimizer.compile(space.population)
    optimizer.strategy.fill_(1)
    optimizer.evaluate(space.population, function)
    normal_values = iter(
        [
            torch.zeros_like(optimizer.strategy),
            torch.tensor([[[-3.0]], [[1.0]]]),
        ]
    )

    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda _tensor: next(normal_values))
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, _shape, device=None: torch.tensor([3, 3, 2, 0], device=device),
    )

    optimizer.update(_context(space, function))

    assert torch.allclose(space.population.positions.flatten(), torch.tensor([3.0, 4.0]))
    assert space.population.best_position.item() == 0
    assert space.population.best_fitness.item() == 0


def test_foa_runs_local_limit_and_global_phases_with_dynamic_population(monkeypatch):
    space = _space(2, lower=-5, upper=5)
    space.population.initialize_static(torch.tensor([[0.0], [4.0]]))
    function = Function(_sphere)
    optimizer = FOA(
        {
            "life_time": 2,
            "area_limit": 2,
            "LSC": 1,
            "GSC": 1,
            "transfer_rate": 1.0,
        }
    )
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)
    optimizer.age[:] = torch.tensor([0, 1])
    random_values = iter([0.6, 0.7])

    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.zeros(shape, dtype=torch.long, device=device),
    )
    monkeypatch.setattr(
        torch,
        "rand_like",
        lambda tensor: torch.full_like(tensor, next(random_values)),
    )

    optimizer.update(_context(space, function))

    assert space.population.n_agents == 3
    assert space.n_agents == 3
    assert optimizer.age.shape == (3,)
    assert optimizer.age[0] == 0
    assert torch.allclose(space.population.positions.flatten(), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(space.population.fitness, torch.tensor([0.0, 1.0, 4.0]))
    assert space.population.best_fitness == 0


def test_rra_elite_and_roulette_reproduction_update_population(monkeypatch):
    space = _space(3)
    space.population.initialize_static(torch.tensor([[0.0], [1.0], [2.0]]))
    function = Function(_sphere)
    optimizer = RRA({"d_runner": 2.0, "tol": 0.0})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.ones(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probs, _n, replacement: torch.tensor([2, 1]),
    )

    optimizer.update(_context(space, function))

    assert torch.allclose(space.population.positions.flatten(), torch.tensor([0.0, 3.0, 2.0]))
    assert torch.allclose(space.population.fitness, torch.tensor([0.0, 9.0, 4.0]))


def test_rra_stall_restart_evaluates_new_population(monkeypatch):
    def constant(x):
        return x.new_tensor(1.0)

    space = _space(2, lower=-5, upper=5)
    function = Function(constant)
    optimizer = RRA({"tol": 1.0, "max_stall": 1})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.full(shape, 0.5, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.zeros(shape, dtype=torch.long, device=device),
    )
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probs, n, replacement: torch.zeros(n, dtype=torch.long),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))

    optimizer.update(_context(space, function))

    assert optimizer.n_stall == 0
    assert torch.allclose(space.population.positions, torch.full_like(space.population.positions, -2.5))
    assert torch.allclose(space.population.fitness, torch.ones_like(space.population.fitness))


def test_rra_archives_daughter_best_before_stall_restart(monkeypatch):
    def sum_objective(x):
        return x.sum()

    space = _space(2, lower=0, upper=200)
    space.population.initialize_static(torch.tensor([[100.0], [101.0]]))
    function = Function(sum_objective)
    optimizer = RRA({"d_runner": 4.0, "d_root": 0.0, "tol": 0.01, "max_stall": 1})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)
    random_values = iter([0.125, 0.5])

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.full(
            shape,
            next(random_values),
            device=device,
            dtype=dtype,
        ),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.zeros(shape, dtype=torch.long, device=device),
    )
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probs, n, replacement: torch.zeros(n, dtype=torch.long),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))

    optimizer.update(_context(space, function))

    assert torch.allclose(space.population.positions, torch.full_like(space.population.positions, 150))
    assert space.population.best_position.item() == pytest.approx(99.5)
    assert space.population.best_fitness.item() == pytest.approx(99.5)


def test_rra_restart_candidates_can_improve_global_best(monkeypatch):
    def sum_objective(x):
        return x.sum()

    space = _space(2, lower=0, upper=200)
    space.population.initialize_static(torch.tensor([[100.0], [101.0]]))
    function = Function(sum_objective)
    optimizer = RRA({"d_runner": 4.0, "d_root": 0.0, "tol": 0.01, "max_stall": 1})
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.full(shape, 0.5, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.zeros(shape, dtype=torch.long, device=device),
    )
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probs, n, replacement: torch.zeros(n, dtype=torch.long),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))

    optimizer.update(_context(space, function))

    assert space.population.best_position.item() == pytest.approx(50.0)
    assert space.population.best_fitness.item() == pytest.approx(50.0)


def test_rra_roulette_is_uniform_for_large_equal_fitness(monkeypatch):
    captured = {}

    def capture(probabilities, n, replacement):
        captured["probabilities"] = probabilities
        return torch.zeros(n, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", capture)

    selected = RRA._roulette_selection(torch.tensor([1e8, 1e8], dtype=torch.float32), 2)

    assert selected.shape == (2,)
    assert torch.isfinite(captured["probabilities"]).all()
    assert torch.allclose(captured["probabilities"], torch.tensor([0.5, 0.5]))


def test_ppa_population_partition_matches_canonical_schedule():
    assert PPA._calculate_population(20, 1, 10) == (13, 1, 6)


def test_ppa_requires_two_agents():
    optimizer = PPA()
    space = _space(1)

    with pytest.raises(e.ValueError, match="at least 2"):
        optimizer.compile(space.population)


def test_ppa_nesting_uses_difference_without_adding_current_position(monkeypatch):
    space = _space(2)
    space.population.initialize_static(torch.tensor([[2.0], [5.0]]))
    optimizer = PPA()
    optimizer.compile(space.population)

    monkeypatch.setattr(
        optimizer,
        "_sample_other",
        lambda indices, _n_agents: torch.ones_like(indices),
    )
    monkeypatch.setattr(
        distribution,
        "generate_levy_distribution",
        _ones_distribution,
    )

    optimizer._nesting_phase(space.population, 1)

    assert space.population.positions[0].item() == pytest.approx(0.03)


def test_ppa_parasitism_selects_one_winner_per_cuckoo_with_pool_offset(monkeypatch):
    space = _space(4)
    space.population.initialize_static(torch.tensor([[0.0], [10.0], [2.0], [6.0]]))
    space.population.fitness[:] = torch.tensor([0.0, 9.0, 1.0, 36.0])
    optimizer = PPA()
    optimizer.compile(space.population)

    def winners(_fitness, n):
        assert n == 2
        return torch.tensor([1, 0])

    monkeypatch.setattr(general, "tournament_selection", winners)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, shape, device=None: torch.full(shape, 3, dtype=torch.long, device=device),
    )
    monkeypatch.setattr(
        optimizer,
        "_sample_other",
        lambda indices, _n_agents: torch.zeros_like(indices),
    )
    monkeypatch.setattr(torch, "rand", _queued_factory([0.0, 0.0]))

    optimizer._parasitism_phase(space.population, n_crows=1, n_cuckoos=2, progress=0.0)

    assert torch.allclose(space.population.positions[1:3].flatten(), torch.tensor([2.0, 10.0]))


def test_ppa_predation_updates_velocity_and_cat_position(monkeypatch):
    space = _space(4)
    space.population.initialize_static(torch.tensor([[0.0], [1.0], [2.0], [4.0]]))
    space.population.best_position.zero_()
    optimizer = PPA()
    optimizer.compile(space.population)

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, device=None, dtype=None: torch.full(shape, 0.5, device=device, dtype=dtype),
    )

    optimizer._predation_phase(space.population, n_crows=1, n_cuckoos=2, progress=0.5)

    assert optimizer.velocity[3].item() == pytest.approx(-3.0)
    assert space.population.positions[3].item() == pytest.approx(1.0)


@pytest.mark.parametrize("optimizer_type", [EP, FOA, RRA, AO, PPA])
def test_completed_migrations_support_multidimensional_float64(optimizer_type):
    torch.manual_seed(11)
    space = _space(
        6,
        n_variables=3,
        n_dimensions=2,
        dtype=torch.float64,
    )
    function = Function(_sphere)
    optimizer = optimizer_type()
    optimizer.compile(space.population)
    optimizer.evaluate(space.population, function)

    optimizer.update(_context(space, function, iteration=2))

    assert space.population.positions.shape[1:] == (3, 2)
    assert space.population.positions.dtype == torch.float64
    assert space.population.fitness.dtype == torch.float64
    if isinstance(optimizer, EP):
        assert optimizer.strategy.shape == space.population.positions.shape
    if isinstance(optimizer, FOA):
        assert optimizer.age.shape == (space.population.n_agents,)
    if isinstance(optimizer, PPA):
        assert optimizer.velocity.shape == space.population.positions.shape
