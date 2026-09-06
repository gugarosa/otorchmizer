# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Deterministic behavioral tests for Lion Optimization Algorithm."""

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.population import Population
from otorchmizer.core.space import Space
from otorchmizer.optimizers.population import LOA
from otorchmizer.optimizers.population.loa import _LionBatch
from otorchmizer.otorchmizer import Otorchmizer


def _population(
    n_agents: int = 20,
    n_variables: int = 2,
    n_dimensions: int = 2,
    lower: float = -20,
    upper: float = 20,
    dtype: torch.dtype = torch.float64,
) -> Population:
    population = Population(
        n_agents=n_agents,
        n_variables=n_variables,
        n_dimensions=n_dimensions,
        lower_bound=torch.full((n_variables, n_dimensions), lower, dtype=dtype),
        upper_bound=torch.full((n_variables, n_dimensions), upper, dtype=dtype),
        dtype=dtype,
    )
    values = torch.linspace(lower / 2, upper / 2, population.positions.numel(), dtype=dtype)
    population.initialize_static(values.reshape_as(population.positions))
    return population


def _sphere(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum()


def _batch(
    population: Population,
    positions: torch.Tensor,
    female: list[bool],
    pride: list[int],
) -> _LionBatch:
    positions = positions.to(device=population.device, dtype=population.dtype).reshape(
        len(female),
        population.n_variables,
        population.n_dimensions,
    )
    fitness = torch.vmap(_sphere)(positions)
    return _LionBatch(
        positions=positions,
        fitness=fitness,
        best_positions=-positions,
        best_fitness=fitness.clone(),
        female=torch.tensor(female, dtype=torch.bool, device=population.device),
        pride=torch.tensor(pride, dtype=torch.long, device=population.device),
        group=torch.zeros(len(female), dtype=torch.long, device=population.device),
        success=torch.ones(len(female), dtype=torch.bool, device=population.device),
        improved=torch.zeros(len(female), dtype=torch.bool, device=population.device),
    )


def _constant_scalar_sequence(values: list[float]):
    samples = iter(values)

    def sample(*_size, device=None, dtype=None, **_kwargs):
        return torch.tensor(next(samples), device=device, dtype=dtype)

    return sample


def test_loa_defaults_and_parameter_validation():
    optimizer = LOA()

    assert (optimizer.N, optimizer.P, optimizer.S, optimizer.R) == (0.2, 4, 0.8, 0.2)
    assert (optimizer.I, optimizer.Ma, optimizer.Mu) == (0.4, 0.3, 0.2)

    for name in ("N", "S", "R", "I", "Ma", "Mu"):
        with pytest.raises(e.TypeError):
            setattr(optimizer, name, "invalid")
        with pytest.raises(e.ValueError):
            setattr(optimizer, name, 1.1)

    with pytest.raises(e.TypeError):
        optimizer.P = 1.5
    with pytest.raises(e.ValueError):
        optimizer.P = 0


def test_compile_creates_exact_device_local_demographics_and_rejects_small_populations():
    torch.manual_seed(4)
    population = _population()
    optimizer = LOA()

    optimizer.compile(population)

    assert optimizer.local_position.shape == population.positions.shape
    assert optimizer.local_position.dtype == population.dtype
    assert optimizer.local_position.device == population.device
    assert optimizer.local_fitness.dtype == population.dtype
    assert optimizer.nomad.sum().item() == 4
    assert (optimizer.nomad & optimizer.female).sum().item() == 1
    for pride_index in range(optimizer.P):
        members = optimizer.pride == pride_index
        assert members.sum().item() == 4
        assert (members & optimizer.female).sum().item() == 3
        assert (members & ~optimizer.female).sum().item() == 1

    with pytest.raises(e.ValueError, match="at least 2 nomad"):
        LOA({"P": 1, "N": 0.1}).compile(_population(n_agents=10))
    with pytest.raises(e.ValueError, match="at least 2 lions per pride"):
        LOA().compile(_population(n_agents=9))


def test_hunting_uses_hunters_for_prey_and_highest_cost_group_as_center(monkeypatch):
    population = _population(n_agents=6, n_variables=1, n_dimensions=1, lower=-200, upper=200)
    positions = population.positions.new_tensor([8.0, 2.0, 0.0, 100.0, 10.0, 20.0])
    lions = _batch(population, positions, [True, True, True, False, True, False], [0, 0, 0, 0, -1, -1])
    population.best_fitness = lions.fitness.min()
    optimizer = LOA({"P": 1})

    monkeypatch.setattr(
        torch,
        "randint",
        lambda _low, _high, size, device=None: torch.tensor([1, 2, 3], device=device),
    )
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *_size, device=None, dtype=None: torch.tensor(0, device=device, dtype=dtype),
    )

    optimizer._hunting(lions, population, Function(_sphere))

    expected = population.positions.new_tensor([17 / 3, 4, 5, 100, 10, 20]).reshape_as(lions.positions)
    torch.testing.assert_close(lions.positions, expected)
    assert lions.group.tolist() == [1, 2, 3, 0, 0, 0]
    torch.testing.assert_close(lions.fitness, torch.vmap(_sphere)(lions.positions))


def test_safe_place_and_roaming_follow_multidimensional_direction_equations(monkeypatch):
    optimizer = LOA()
    position = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: tensor.new_tensor([-0.8, 0.6]))

    monkeypatch.setattr(torch, "rand", _constant_scalar_sequence([0.25, 0.75, 0.75]))
    safe = optimizer._safe_place_candidate(position, target)
    lateral = 2.5 * torch.tan(torch.tensor(torch.pi / 12, dtype=position.dtype))
    expected_safe = position.new_tensor([[1.5 - 0.8 * lateral, 2.0 + 0.6 * lateral]])
    torch.testing.assert_close(safe, expected_safe)

    monkeypatch.setattr(torch, "rand", _constant_scalar_sequence([0.5, 0.5]))
    roaming = optimizer._roaming_candidate(position, target)
    torch.testing.assert_close(roaming, target)


def test_mating_uses_selected_male_average_and_creates_opposite_sexes(monkeypatch):
    population = _population(n_agents=3, n_variables=2, n_dimensions=2, lower=-10, upper=10)
    positions = population.positions.new_tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[4.0, 4.0], [4.0, 4.0]],
        ]
    )
    lions = _batch(population, positions, [True, False, False], [0, 0, 0])
    optimizer = LOA({"P": 1, "Mu": 0})
    function = Function(_sphere)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *_size, device=None, dtype=None: torch.tensor(1, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.ones_like(tensor))

    cubs = optimizer._mating_operator(
        lions,
        torch.tensor(0),
        torch.tensor([1, 2]),
        0,
        population,
        function,
    )

    torch.testing.assert_close(cubs.positions[0], torch.full_like(cubs.positions[0], 1.2))
    torch.testing.assert_close(cubs.positions[1], torch.full_like(cubs.positions[1], 1.8))
    assert cubs.female.sum().item() == 1
    assert (~cubs.female).sum().item() == 1
    assert torch.equal(cubs.best_positions, cubs.positions)
    assert torch.equal(cubs.best_fitness, cubs.fitness)
    torch.testing.assert_close(cubs.fitness, function(cubs.positions))


def test_nomad_roaming_uses_adaptive_probability_per_coordinate(monkeypatch):
    population = _population(n_agents=2, n_variables=1, n_dimensions=2, lower=-10, upper=10)
    positions = population.positions.new_tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
    lions = _batch(population, positions, [True, False], [-1, -1])
    population.best_fitness = lions.fitness.min()
    optimizer = LOA({"P": 1})
    samples = iter(
        [
            population.positions.new_tensor([[0.5, 0.5]]),
            population.positions.new_tensor([[0.5, 0.5]]),
            population.positions.new_tensor([[0.95, 0.05]]),
            population.positions.new_tensor([[0.5, 0.7]]),
        ]
    )
    monkeypatch.setattr(torch, "rand_like", lambda _tensor: next(samples))

    optimizer._nomad_roaming(lions, population, Function(_sphere))

    torch.testing.assert_close(lions.positions[0], positions[0])
    torch.testing.assert_close(lions.positions[1], population.positions.new_tensor([[9.0, 2.0]]))
    torch.testing.assert_close(lions.fitness, torch.vmap(_sphere)(lions.positions))


def test_personal_and_global_bests_survive_a_later_worse_move():
    population = _population(n_agents=4, n_variables=1, n_dimensions=1, lower=-10, upper=10)
    population.positions.fill_(4)
    optimizer = LOA({"P": 1, "N": 0.5})
    function = Function(_sphere)
    optimizer.compile(population)
    optimizer.evaluate(population, function)
    lions = optimizer._make_lions(population)
    index = torch.tensor([0])

    optimizer._evaluate_move(lions, index, population.positions.new_zeros((1, 1, 1)), population, function)
    optimizer._evaluate_move(lions, index, population.positions.new_full((1, 1, 1), 3), population, function)

    assert lions.positions[0].item() == 3
    assert lions.fitness[0].item() == 9
    assert lions.best_positions[0].item() == 0
    assert lions.best_fitness[0].item() == 0
    assert population.best_position.item() == 0
    assert population.best_fitness.item() == 0


def test_defense_attack_migration_and_control_preserve_aligned_lion_state(monkeypatch):
    population = _population(n_agents=4, n_variables=1, n_dimensions=1, lower=-20, upper=20)
    optimizer = LOA({"P": 1, "N": 0.5, "S": 0.5, "I": 1.0})
    optimizer.pride_sizes = torch.tensor([2], device=population.device)
    optimizer.pride_females = torch.tensor([1], device=population.device)
    optimizer.n_nomads = 2
    optimizer.nomad_females = 1

    lions = _batch(
        population,
        population.positions.new_tensor([10.0, 8.0, 7.0, 0.0]),
        [True, False, True, False],
        [0, 0, -1, -1],
    )
    cubs = _batch(
        population,
        population.positions.new_tensor([5.0, 1.0]),
        [True, False],
        [0, 0],
    )

    defended = optimizer._defense(lions, cubs)
    pride_males = ((defended.pride == 0) & ~defended.female).nonzero(as_tuple=True)[0]
    assert defended.positions[pride_males].item() == 1

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, device=None, dtype=None: torch.zeros(
            size[0] if len(size) == 1 and isinstance(size[0], tuple) else size,
            device=device,
            dtype=dtype,
        ),
    )
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))
    optimizer._nomad_attack(defended, population)
    optimizer._migration(defended, population)
    controlled = optimizer._population_control(defended, population)

    assert controlled.size == population.n_agents
    assert ((controlled.pride == 0) & controlled.female).sum().item() == 1
    assert ((controlled.pride == 0) & ~controlled.female).sum().item() == 1
    assert ((controlled.pride < 0) & controlled.female).sum().item() == 1
    assert ((controlled.pride < 0) & ~controlled.female).sum().item() == 1
    assert controlled.positions[(controlled.pride == 0) & controlled.female].item() == 5
    assert controlled.positions[(controlled.pride == 0) & ~controlled.female].item() == 0
    torch.testing.assert_close(controlled.best_positions, -controlled.positions)
    torch.testing.assert_close(controlled.fitness, torch.vmap(_sphere)(controlled.positions))


def test_otorchmizer_engine_runs_all_loa_phases_on_multidimensional_tensors():
    torch.manual_seed(12)
    space = Space(
        n_agents=20,
        n_variables=3,
        n_dimensions=2,
        lower_bound=-4,
        upper_bound=6,
        device="cpu",
    )
    space.build()
    space.population.to(torch.device("cpu"), torch.float64)

    def tilted_bowl(x):
        weights = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype).reshape_as(x)
        return (weights * (x - 0.75).square()).sum() + 0.05 * x.prod(dim=-1).sum()

    function = Function(tilted_bowl)
    optimizer = LOA()
    engine = Otorchmizer(space, optimizer, function)

    engine.start(n_iterations=3)

    population = space.population
    assert population.positions.shape == (20, 3, 2)
    assert population.positions.dtype == torch.float64
    assert torch.isfinite(population.positions).all()
    assert (population.positions >= population.lb).all()
    assert (population.positions <= population.ub).all()
    torch.testing.assert_close(population.fitness, function(population.positions))
    assert torch.all(optimizer.local_fitness <= population.fitness)
    assert population.best_fitness <= population.fitness.min()
    for pride_index in range(optimizer.P):
        members = optimizer.pride == pride_index
        assert members.sum() == optimizer.pride_sizes[pride_index]
        assert (members & optimizer.female).sum() == optimizer.pride_females[pride_index]
    assert optimizer.nomad.sum().item() == optimizer.n_nomads
