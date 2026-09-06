# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""State and equation contracts for swarm optimizers."""

from types import SimpleNamespace

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.optimizers.swarm import (
    ABC,
    ABO,
    AF,
    AIWPSO,
    BA,
    BOA,
    BWO,
    CS,
    CSA,
    EHO,
    FA,
    FFOA,
    FPA,
    FSO,
    JS,
    KH,
    MFO,
    MRFO,
    PIO,
    PSO,
    RPSO,
    SAVPSO,
    SBO,
    SCA,
    SFO,
    SOS,
    SSA,
    SSO,
    STOA,
    VPSO,
    WAOA,
    WOA,
)
from otorchmizer.optimizers.swarm.goa import GOA


def _population(
    positions: list[float],
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
    dtype: torch.dtype = torch.float32,
) -> Population:
    population = Population(
        len(positions),
        1,
        1,
        torch.tensor([lower_bound]),
        torch.tensor([upper_bound]),
        dtype=dtype,
    )
    population.positions[:, 0, 0] = torch.tensor(positions, dtype=dtype)
    return population


def _context(population: Population, function: Function, iteration: int = 0, n_iterations: int = 10) -> UpdateContext:
    return UpdateContext(
        space=SimpleNamespace(population=population),
        function=function,
        iteration=iteration,
        n_iterations=n_iterations,
        device=population.device,
    )


@pytest.mark.parametrize("optimizer_class", [PSO, AIWPSO, RPSO, SAVPSO, VPSO])
def test_pso_keeps_current_and_personal_fitness_separate(optimizer_class):
    population = _population([1.0, 2.0])
    function = Function(lambda position: (position**2).sum())
    optimizer = optimizer_class()
    optimizer.compile(population)
    optimizer.evaluate(population, function)

    population.positions[:, 0, 0] = torch.tensor([10.0, 0.5])
    optimizer.evaluate(population, function)

    torch.testing.assert_close(population.fitness, torch.tensor([100.0, 0.25]))
    torch.testing.assert_close(optimizer.local_fitness, torch.tensor([1.0, 0.25]))
    torch.testing.assert_close(optimizer.local_position[:, 0, 0], torch.tensor([1.0, 0.5]))
    assert population.best_fitness.item() == 0.25
    assert population.best_position.item() == 0.5


def test_csa_evaluation_initializes_and_preserves_personal_memory():
    population = _population([1.0, 2.0])
    function = Function(lambda position: (position**2).sum())
    optimizer = CSA()
    optimizer.compile(population)
    optimizer.evaluate(population, function)

    population.positions[:, 0, 0] = torch.tensor([3.0, 0.5])
    optimizer.evaluate(population, function)

    torch.testing.assert_close(population.fitness, torch.tensor([9.0, 0.25]))
    torch.testing.assert_close(optimizer.memory_fitness, torch.tensor([1.0, 0.25]))
    torch.testing.assert_close(optimizer.memory[:, 0, 0], torch.tensor([1.0, 0.5]))


def test_ssa_uses_one_leader_then_the_sequential_follower_recurrence(monkeypatch):
    population = _population([0.0, 2.0, -1.0], lower_bound=-2.0, upper_bound=3.0)
    population.best_position.fill_(0.5)
    function = Function(lambda position: (position**2).sum())
    optimizer = SSA()

    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([-1.0, 0.5, -0.25]))


def test_ssa_updates_a_single_agent_as_the_leader(monkeypatch):
    population = _population([0.0], lower_bound=-2.0, upper_bound=3.0)
    population.best_position.fill_(0.5)
    function = Function(lambda position: (position**2).sum())
    optimizer = SSA()

    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))

    optimizer.update(_context(population, function))

    assert population.positions.item() == -1.0


def test_goa_uses_remapped_social_distance_neighbor_direction_and_bound_scale():
    population = _population([0.0, 1.0], lower_bound=-2.0, upper_bound=2.0, dtype=torch.float64)
    population.best_position.zero_()
    optimizer = GOA()

    optimizer.update(_context(population, Function(lambda position: (position**2).sum())))

    social_force = 0.5 * torch.exp(torch.tensor(-2.0, dtype=torch.float64)) - torch.exp(
        torch.tensor(-3.0, dtype=torch.float64)
    )
    expected = torch.tensor([2 * social_force, -2 * social_force], dtype=torch.float64)
    torch.testing.assert_close(population.positions[:, 0, 0], expected)


def test_sfo_applies_prey_density_equation_and_promotes_better_sardines(monkeypatch):
    population = _population([4.0, 5.0])
    population.fitness[:] = torch.tensor([4.0, 5.0])
    population.best_position.fill_(4.0)
    population.best_fitness.fill_(4.0)
    function = Function(lambda positions: positions[:, 0, 0], batch=True)
    optimizer = SFO({"PP": 0.5, "A": 0.0})
    optimizer.compile(population)
    optimizer.sardine_positions[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    optimizer.sardine_fitness = function(optimizer.sardine_positions)

    draws = iter([0.25, 0.75])

    def controlled_rand(*size, **kwargs):
        return torch.full(
            size if len(size) > 1 else (size[0],),
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)

    optimizer.update(_context(population, function))

    expected = torch.tensor([-7 / 24, 1 / 24])
    torch.testing.assert_close(population.positions[:, 0, 0], expected)
    torch.testing.assert_close(population.fitness, expected)


def test_sfo_uses_new_sailfish_best_during_sardine_phase(monkeypatch):
    population = _population([4.0, 5.0])
    population.fitness[:] = torch.tensor([4.0, 5.0])
    population.best_position.fill_(4.0)
    population.best_fitness.fill_(4.0)
    function = Function(lambda positions: positions[:, 0, 0], batch=True)
    optimizer = SFO({"PP": 0.5, "A": 1.0})
    optimizer.compile(population)
    optimizer.sardine_positions[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    optimizer.sardine_fitness = function(optimizer.sardine_positions)
    draws = iter([0.5, 0.75, 0.5])

    def controlled_rand(*size, **kwargs):
        return torch.full(
            size if len(size) > 1 else (size[0],),
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([-1.0, -0.5]))
    assert population.best_fitness.item() == -1.0
    assert population.best_position.item() == -1.0


def test_sfo_low_attack_power_updates_selected_sardine_variables():
    torch.manual_seed(11)
    population = Population(
        2,
        4,
        1,
        torch.full((4,), -10.0),
        torch.full((4,), 10.0),
    )
    population.positions.fill_(2.0)
    population.fitness.fill_(8.0)
    population.best_position.fill_(2.0)
    population.best_fitness.fill_(8.0)
    function = Function(lambda position: position.sum())
    optimizer = SFO({"PP": 0.5, "A": 0.25})
    optimizer.compile(population)

    optimizer.update(_context(population, function))

    assert torch.isfinite(optimizer.sardine_positions).all()
    assert torch.isfinite(optimizer.sardine_fitness).all()


def test_eho_rejects_more_clans_than_agents():
    optimizer = EHO({"n_clans": 3})

    with pytest.raises(e.ValueError, match="n_clans"):
        optimizer.compile(_population([0.0, 1.0]))


@pytest.mark.parametrize(
    ("optimizer", "state_names"),
    [
        (ABC(), ("trial",)),
        (ABO(), ("w1", "w2")),
        (AF(), ("branch",)),
        (BA(), ("velocity", "frequency", "loudness", "pulse_rate")),
        (CSA(), ("memory", "memory_fitness")),
        (FSO(), ("weight",)),
        (KH(), ("induced_motion", "foraging_motion")),
        (MFO(), ("flames", "flame_fitness")),
        (PIO(), ("velocity",)),
        (PSO(), ("local_position", "local_fitness", "velocity")),
        (RPSO(), ("local_position", "local_fitness", "velocity", "mass")),
        (VPSO(), ("local_position", "local_fitness", "velocity", "v_velocity")),
        (SFO({"PP": 0.5}), ("sardine_positions", "sardine_fitness")),
        (SSO(), ("weight",)),
    ],
)
def test_compiled_floating_state_uses_population_dtype(optimizer, state_names):
    population = _population([0.0, 1.0], dtype=torch.float64)

    optimizer.compile(population)

    for state_name in state_names:
        state = getattr(optimizer, state_name)
        if state.is_floating_point():
            assert state.dtype == population.dtype


@pytest.mark.parametrize(
    "optimizer",
    [
        ABC(),
        ABO(),
        AF(),
        BA(),
        BOA(),
        BWO(),
        CS(),
        CSA(),
        EHO(),
        FA(),
        FFOA(),
        FPA(),
        FSO(),
        JS(),
        MFO(),
        MRFO(),
        PIO(),
        PSO(),
        AIWPSO(),
        RPSO(),
        SAVPSO(),
        VPSO(),
        SBO(),
        SCA(),
        SFO({"PP": 0.5}),
        SOS(),
        SSA(),
        SSO(),
        STOA(),
        WAOA(),
        WOA(),
    ],
)
def test_cpu_float16_update_preserves_dtype_when_operations_are_supported(optimizer):
    torch.manual_seed(7)
    population = Population(
        6,
        3,
        1,
        torch.full((3,), -1.0),
        torch.ones(3),
        dtype=torch.float16,
    )
    population.initialize_uniform()
    function = Function(lambda position: (position**2).sum())
    optimizer.compile(population)
    optimizer.evaluate(population, function)

    optimizer.update(_context(population, function))
    population.clip()
    optimizer.evaluate(population, function)

    assert population.positions.dtype == torch.float16
    assert population.fitness.dtype == torch.float16
    assert torch.isfinite(population.positions).all()
    assert torch.isfinite(population.fitness).all()
