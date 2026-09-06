# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Current-upstream parity and corrected-equation contracts for swarm optimizers."""

import math
from types import SimpleNamespace

import pytest
import torch

from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.optimizers.swarm import (
    AF,
    AIWPSO,
    BA,
    BOA,
    BWO,
    CS,
    EHO,
    FA,
    FFOA,
    GOA,
    JS,
    KH,
    MFO,
    MRFO,
    PSO,
    SBO,
    SCA,
    STOA,
    WAOA,
)
from otorchmizer.optimizers.swarm import cs as cs_module


def _population(
    positions: torch.Tensor,
    lower_bound: float = -100.0,
    upper_bound: float = 100.0,
) -> Population:
    if positions.ndim == 2:
        positions = positions.unsqueeze(-1)
    n_agents, n_variables, n_dimensions = positions.shape
    population = Population(
        n_agents,
        n_variables,
        n_dimensions,
        torch.full((n_variables,), lower_bound),
        torch.full((n_variables,), upper_bound),
        dtype=positions.dtype,
    )
    population.positions = positions.clone()
    return population


def _context(
    population: Population,
    function: Function,
    iteration: int = 0,
    n_iterations: int = 10,
) -> UpdateContext:
    return UpdateContext(
        space=SimpleNamespace(population=population),
        function=function,
        iteration=iteration,
        n_iterations=n_iterations,
        device=population.device,
    )


def _shape(size):
    return size[0] if len(size) == 1 and isinstance(size[0], tuple) else size


def test_waoa_runs_all_three_greedy_phases(monkeypatch):
    population = _population(torch.tensor([[1.0], [3.0]]), lower_bound=-10.0, upper_bound=10.0)
    calls = []

    def objective(positions):
        calls.append(positions.shape[0])
        return positions.square().sum(dim=(1, 2))

    function = Function(objective, batch=True)
    population.fitness = function(population.positions)
    population.update_best()
    calls.clear()
    optimizer = WAOA()
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.full(
            size,
            int(high == 3),
            device=kwargs["device"],
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))

    optimizer.update(_context(population, function))

    assert calls == [2, 1, 1, 2]
    torch.testing.assert_close(population.fitness, function(population.positions))


def test_current_swarm_defaults_and_canonical_surfaces():
    af = AF()
    kh = KH()

    assert (af.c1, af.c2, af.m, af.Q) == (0.75, 1.25, 10, 0.75)
    assert not hasattr(af, "g")
    assert EHO().n_clans == 10
    assert GOA().c_min == 0.00001
    assert SBO().alpha == 0.9
    assert (kh.w_n, kh.NN, kh.w_f, kh.C_t) == (0.42, 5, 0.38, 0.5)
    assert (STOA().Cf, STOA().u, STOA().v) == (2.0, 1.0, 1.0)
    assert "Walrus" in WAOA.__doc__


def test_waoa_supports_a_single_walrus():
    population = _population(torch.tensor([[1.0]]), lower_bound=-2.0, upper_bound=2.0)
    function = Function(lambda position: position.square().sum())
    optimizer = WAOA()
    optimizer.evaluate(population, function)

    optimizer.update(_context(population, function))

    assert torch.isfinite(population.positions).all()
    torch.testing.assert_close(population.fitness, function(population.positions))


def test_waoa_local_exploration_samples_the_scaled_interval(monkeypatch):
    population = _population(torch.tensor([[0.25]]), lower_bound=0.0, upper_bound=1.0)
    function = Function(lambda position: (position - 0.5).square().sum())
    optimizer = WAOA()
    optimizer.evaluate(population, function)
    draws = iter([0.0, 0.25])
    monkeypatch.setattr(
        torch,
        "rand_like",
        lambda tensor: torch.full_like(tensor, next(draws)),
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.ones(size, device=kwargs["device"], dtype=torch.long),
    )

    optimizer.update(_context(population, function))

    assert population.positions.item() == 0.5
    assert population.fitness.item() == 0.0


def test_af_restores_canonical_distance_state_and_knob_effect():
    function = Function(lambda position: position.square().sum())

    def run(c1):
        torch.manual_seed(8)
        population = _population(torch.tensor([[0.5], [1.0]]), lower_bound=-2.0, upper_bound=2.0)
        optimizer = AF({"c1": c1, "c2": 0.0, "m": 2, "Q": 1.0})
        optimizer.compile(population)
        optimizer.evaluate(population, function)
        torch.manual_seed(9)
        optimizer.update(_context(population, function))
        return population.positions, optimizer

    zero_positions, zero = run(0.0)
    moving_positions, moving = run(2.0)

    assert zero.c1 == 0.0 and moving.c1 == 2.0
    assert zero.p_distance.shape == zero.g_distance.shape == (2,)
    assert not torch.equal(zero_positions, moving_positions)


def test_af_scalar_lineage_distance_remains_nonzero_across_updates(monkeypatch):
    population = _population(torch.tensor([[1.0]]))
    function = Function(lambda position: -position.sum())
    optimizer = AF({"c1": 0.0, "c2": 1.0, "m": 1, "Q": 1.0})
    optimizer.compile(population)
    optimizer.p_distance.fill_(1.0)
    optimizer.g_distance.fill_(1.0)
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.ones(_shape(size), device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *size, **kwargs: torch.ones(size, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.zeros_like(tensor))

    optimizer.update(_context(population, function))
    first_distance = optimizer.p_distance.clone()
    optimizer.update(_context(population, function))

    assert first_distance.item() == 1.0
    assert optimizer.p_distance.item() == 1.0
    assert optimizer.g_distance.item() == 1.0
    assert population.positions.item() == 3.0


def test_af_selected_descendants_gather_their_parent_lineage(monkeypatch):
    population = _population(torch.tensor([[0.0], [10.0]]), lower_bound=-20.0, upper_bound=20.0)
    function = Function(lambda position: position.square().sum())
    optimizer = AF({"c1": 0.0, "c2": 1.0, "m": 2, "Q": 1.0})
    optimizer.compile(population)
    optimizer.p_distance = torch.tensor([2.0, 8.0])
    optimizer.g_distance = torch.tensor([1.0, 1.0])
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.ones(_shape(size), device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    noise = torch.tensor([[[[-1.0]], [[1.0]]], [[[0.5]], [[0.25]]]])
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: noise.to(**kwargs))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.tensor([1.0, 1.0, 0.0, 0.0]))

    optimizer.update(_context(population, function))

    torch.testing.assert_close(optimizer.g_distance, torch.tensor([8.0, 8.0]))
    torch.testing.assert_close(optimizer.p_distance, torch.tensor([4.0, 2.0]))


def test_af_archives_best_offspring_before_stochastic_selection(monkeypatch):
    population = _population(torch.tensor([[1.0]]))
    function = Function(lambda position: position.square().sum())
    optimizer = AF({"c1": 0.0, "c2": 1.0, "m": 2, "Q": 1.0})
    optimizer.compile(population)
    optimizer.p_distance.fill_(1.0)
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.ones(_shape(size), device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *args, **kwargs: torch.tensor([[[[-1.0]], [[1.0]]]], **kwargs),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.tensor([1.0, 0.0]))

    optimizer.update(_context(population, function))

    assert population.positions.item() == 2.0
    assert population.best_position.item() == 0.0
    assert population.best_fitness.item() == 0.0


def test_bwo_mutation_swaps_variables_instead_of_adding_gaussian_noise(monkeypatch):
    positions = torch.tensor([[[1.0], [2.0]], [[5.0], [5.0]]])
    population = _population(positions)
    function = Function(lambda position: position[0].sum() + 10 * position[1].sum())
    optimizer = BWO({"pp": 0.5, "cr": 0.0, "pm": 1.0})
    optimizer.evaluate(population, function)
    monkeypatch.setattr(torch, "randperm", lambda n, **kwargs: torch.tensor([1, 0], device=kwargs["device"]))
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size=(), **kwargs: torch.zeros(size, device=kwargs["device"], dtype=torch.long),
    )

    optimizer.update(_context(population, function))

    matches = (population.positions == population.positions.new_tensor([[2.0], [1.0]])).flatten(start_dim=1)
    assert matches.all(dim=1).any()
    torch.testing.assert_close(population.fitness, function(population.positions))


def test_bwo_one_variable_population_still_generates_offspring(monkeypatch):
    population = _population(torch.tensor([[1.0], [2.0], [4.0], [5.0], [6.0]]))
    function = Function(lambda position: (position - 3).square().sum())
    optimizer = BWO({"pp": 0.6, "cr": 0.44, "pm": 0.0})
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size=(), **kwargs: torch.tensor([1, 2], device=kwargs["device"]),
    )
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(_shape(size), 0.5, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    optimizer.update(_context(population, function))

    assert torch.any(population.positions[:, 0, 0] == 3.0)
    assert population.best_fitness.item() == 0.0


def test_bwo_positive_cannibal_fraction_retains_offspring_for_two_agents(monkeypatch):
    population = _population(torch.tensor([[-1.0], [1.0]]))
    function = Function(lambda position: position.square().sum())
    optimizer = BWO()
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size=(), **kwargs: torch.tensor([0, 1], device=kwargs["device"]),
    )
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(_shape(size), 0.5, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    for _ in range(3):
        optimizer.update(_context(population, function))

    assert torch.any(population.positions[:, 0, 0] == 0.0)
    assert population.best_fitness.item() == 0.0
    assert population.best_position.item() == 0.0


def test_bwo_archives_offspring_even_when_cannibal_survival_is_disabled(monkeypatch):
    population = _population(torch.tensor([[-1.0], [1.0]]))
    function = Function(lambda position: position.square().sum())
    optimizer = BWO({"cr": 0.0, "pm": 0.0})
    optimizer.evaluate(population, function)
    original = population.positions.clone()
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size=(), **kwargs: torch.tensor([0, 1], device=kwargs["device"]),
    )
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(_shape(size), 0.5, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    optimizer.update(_context(population, function))

    assert torch.equal(population.positions, original)
    assert population.best_fitness.item() == 0.0
    assert population.best_position.item() == 0.0


def test_ffoa_retains_axis_state_and_computes_elementwise_smell(monkeypatch):
    positions = torch.tensor([[[1.0], [2.0]]])
    population = _population(positions, lower_bound=0.0, upper_bound=10.0)
    function = Function(lambda position: position.sum())
    optimizer = FFOA()
    optimizer.compile(population)
    population.fitness.fill_(10.0)
    draws = iter([1.0, 2.0])
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, next(draws), device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    optimizer.update(_context(population, function))

    expected = torch.tensor([[[1 / math.sqrt(13)], [0.2]]])
    torch.testing.assert_close(population.positions, expected)
    torch.testing.assert_close(optimizer.x_axis, torch.tensor([[[2.0], [3.0]]]))
    torch.testing.assert_close(optimizer.y_axis, torch.tensor([[[3.0], [4.0]]]))


def test_kh_restores_canonical_knobs_and_persistent_motion():
    population = _population(torch.tensor([[1.0], [2.0]], dtype=torch.float64))
    function = Function(lambda position: position.square().sum())
    optimizer = KH(
        {
            "N_max": 0.0,
            "w_n": 0.5,
            "NN": 0,
            "V_f": 0.0,
            "w_f": 0.25,
            "D_max": 0.0,
            "C_t": 0.0,
            "Cr": 0.0,
            "Mu": 0.0,
        }
    )
    optimizer.compile(population)
    optimizer.evaluate(population, function)
    optimizer.motion.fill_(1.0)
    optimizer.foraging.fill_(2.0)

    optimizer.update(_context(population, function))

    torch.testing.assert_close(optimizer.motion, torch.full_like(population.positions, 0.5))
    torch.testing.assert_close(optimizer.foraging, torch.full_like(population.positions, 0.5))
    assert not hasattr(optimizer, "W_n")
    assert not hasattr(optimizer, "W_f")
    assert not hasattr(optimizer, "nn")
    assert not hasattr(optimizer, "d_s")


def test_kh_equal_zero_fitness_remains_finite():
    population = _population(torch.tensor([[-1.0], [1.0]]))
    function = Function(lambda position: position.sum() * 0)
    optimizer = KH({"NN": 0, "Cr": 0.0, "Mu": 0.0})
    optimizer.compile(population)
    optimizer.evaluate(population, function)

    optimizer.update(_context(population, function))

    assert torch.isfinite(population.positions).all()
    assert torch.isfinite(optimizer.motion).all()
    assert torch.isfinite(optimizer.foraging).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kh_tied_fitness_does_not_create_infinite_foraging(dtype):
    positions = torch.tensor([[[-10.0], [0.0]], [[10.0], [0.0]]], dtype=dtype)
    population = _population(positions)
    function = Function(lambda position: position.square().sum())
    optimizer = KH({"Cr": 0.0, "Mu": 0.0})
    optimizer.compile(population)
    optimizer.evaluate(population, function)

    optimizer.update(_context(population, function))

    assert torch.isfinite(population.positions).all()
    assert torch.isfinite(optimizer.motion).all()
    assert torch.isfinite(optimizer.foraging).all()


def test_kh_archives_a_better_temporary_food_location():
    positions = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    population = _population(positions)
    function = Function(lambda position: (position - 1.2).square().sum())
    optimizer = KH({"C_t": 0.0, "Cr": 0.0, "Mu": 0.0})
    optimizer.compile(population)
    optimizer.evaluate(population, function)
    fitness = population.fitness.clone()
    weights = fitness.reciprocal()
    food_position = (weights[:, None, None] * population.positions).sum(dim=0) / weights.sum()
    food_fitness = function(food_position.unsqueeze(0))[0]

    optimizer.update(_context(population, function))

    assert population.best_fitness.item() == pytest.approx(food_fitness.item())
    torch.testing.assert_close(population.best_position, food_position)


def test_stoa_uses_spiral_radius_parameters(monkeypatch):
    population = _population(torch.tensor([[2.0], [3.0]]))
    population.best_position.fill_(1.0)
    optimizer = STOA({"u": 0.0, "v": 5.0})
    monkeypatch.setattr(
        torch, "rand", lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs["device"], dtype=kwargs["dtype"])
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))

    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    torch.testing.assert_close(population.positions, torch.ones_like(population.positions))


def test_stoa_zero_best_coordinate_is_not_absorbing(monkeypatch):
    population = _population(torch.tensor([[-2.0], [-1.0], [0.0]]))
    function = Function(lambda position: (position - 1).square().sum())
    optimizer = STOA({"Cf": 2.0, "u": 1.0, "v": 0.0})
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.5))

    optimizer.update(_context(population, function))

    assert torch.count_nonzero(population.positions) > 0


def test_boa_uses_current_fragrance_equation(monkeypatch):
    population = _population(torch.tensor([[2.0]]), lower_bound=-10.0, upper_bound=10.0)
    population.fitness.fill_(4.0)
    population.best_position.fill_(1.0)
    optimizer = BOA({"c": 1.0, "a": 1.0, "p": 1.0})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch, "rand", lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs["device"], dtype=kwargs["dtype"])
    )

    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    assert population.positions.item() == -5.0
    assert optimizer.fragrance.item() == 4.0


def test_eho_separates_the_worst_elephant_in_every_clan(monkeypatch):
    population = _population(torch.tensor([[0.0], [1.0], [2.0], [3.0]]), lower_bound=-10.0, upper_bound=10.0)
    function = Function(lambda position: position.square().sum())
    optimizer = EHO({"n_clans": 2, "alpha": 0.0, "beta": 0.0})
    optimizer.compile(population)
    assert optimizer.n_ci == 2
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch, "rand", lambda *size, **kwargs: torch.full(size, 0.25, device=kwargs["device"], dtype=kwargs["dtype"])
    )

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[[1, 3], 0, 0], torch.tensor([-5.0, -5.0]))
    torch.testing.assert_close(population.fitness[[1, 3]], torch.tensor([25.0, 25.0]))


def test_eho_archives_clan_improvement_before_separation(monkeypatch):
    population = _population(torch.tensor([[2.0], [4.0]]), lower_bound=0.0, upper_bound=10.0)
    function = Function(lambda position: position.square().sum())
    optimizer = EHO({"n_clans": 2, "alpha": 0.0, "beta": 0.0})
    optimizer.compile(population)
    optimizer.evaluate(population, function)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([5.0, 5.0]))
    assert population.best_position.item() == 0.0
    assert population.best_fitness.item() == 0.0


def test_sca_uses_r_max_as_the_shared_target_weight(monkeypatch):
    population = _population(torch.tensor([[1.0], [2.0]]))
    population.best_position.fill_(2.0)
    optimizer = SCA({"r_min": 0.0, "r_max": 3.0, "a": 1.0})
    draws = iter([0.25, 1.0, 0.25])

    def controlled_rand(*size, **kwargs):
        return torch.full(
            size,
            next(draws),
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(
        torch,
        "rand",
        controlled_rand,
    )

    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([6.0, 6.0]))


def test_sca_independent_coordinate_draws_break_diagonal_lock(monkeypatch):
    positions = torch.ones(2, 2, 1)
    population = _population(positions)
    population.best_position.fill_(2.0)
    optimizer = SCA({"r_min": 0.0, "r_max": 2.0, "a": 1.0})
    draws = iter(
        [
            torch.tensor([[[0.25], [0.0]], [[0.0], [0.25]]]),
            torch.full((2, 2, 1), 0.5),
            torch.zeros(2),
        ]
    )
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(draws).to(**kwargs))

    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    assert population.positions[0, 0] != population.positions[0, 1]
    assert population.positions[1, 0] != population.positions[1, 1]


def test_mfo_uses_iteration_dependent_spiral_lower_bound(monkeypatch):
    population = _population(torch.tensor([[1.0], [3.0]]))
    population.fitness = torch.tensor([1.0, 9.0])
    optimizer = MFO({"b": 1.0})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch, "rand", lambda *size, **kwargs: torch.zeros(size, device=kwargs["device"], dtype=kwargs["dtype"])
    )

    optimizer.update(
        _context(population, Function(lambda position: position.square().sum()), iteration=5, n_iterations=10)
    )

    expected = 1 - 2 * math.exp(-1.5)
    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([1.0, expected]))


def test_sbo_uses_selected_bower_probability(monkeypatch):
    population = _population(torch.tensor([[0.0], [2.0]]))
    population.fitness = torch.tensor([0.0, 3.0])
    population.best_position.zero_()
    optimizer = SBO({"alpha": 0.9, "p_mutation": 0.0})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda *args, **kwargs: torch.tensor([1, 1]),
    )

    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([0.75, 1.25]))
    torch.testing.assert_close(population.fitness, torch.tensor([0.5625, 1.5625]))


def test_mrfo_chain_uses_independent_alpha_and_movement_draws(monkeypatch):
    population = _population(torch.tensor([[2.0]]))
    population.best_position.zero_()
    population.best_fitness.zero_()
    population.fitness.fill_(4.0)
    optimizer = MRFO({"S": 0.0})
    draws = iter([0.75, 0.25, 0.75, 0.0, 0.0])

    def controlled_rand(*size, **kwargs):
        return torch.full(size, next(draws), device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr(torch, "rand", controlled_rand)
    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    alpha = 0.5 * math.sqrt(abs(math.log(0.25)))
    expected = 0.5 - 2 * alpha
    assert population.positions.item() == pytest.approx(expected)
    assert population.fitness.item() == pytest.approx(expected**2)


def test_fa_preserves_frozen_population_order(monkeypatch):
    population = _population(torch.tensor([[5.0], [1.0], [10.0]]))
    population.fitness = torch.tensor([1.0, 0.0, 2.0])
    optimizer = FA({"alpha": 0.0, "beta": 0.5, "gamma": 0.0})
    monkeypatch.setattr(
        torch, "rand", lambda *args, **kwargs: torch.tensor(0.5, device=kwargs["device"], dtype=kwargs["dtype"])
    )

    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([3.0, 1.0, 4.25]))


def test_fa_random_perturbations_do_not_lock_identical_agents_together(monkeypatch):
    positions = torch.tensor([[[0.0], [0.0]], [[1.0], [1.0]], [[1.0], [1.0]]])
    population = _population(positions)
    population.fitness = torch.tensor([0.0, 2.0, 2.0])
    optimizer = FA({"alpha": 1.0, "beta": 0.0, "gamma": 0.0})
    random = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    monkeypatch.setattr(
        torch,
        "rand_like",
        lambda tensor: random.reshape_as(tensor).to(device=tensor.device, dtype=tensor.dtype),
    )

    optimizer.update(_context(population, Function(lambda position: position.square().sum())))

    assert not torch.equal(population.positions[1], population.positions[2])


def test_cs_retains_gaussian_levy_multiplier_and_p_is_retention(monkeypatch):
    population = _population(torch.tensor([[2.0], [4.0]]))
    function = Function(lambda position: -position.sum())
    population.fitness = function(population.positions)
    population.best_position.zero_()
    optimizer = CS({"alpha": 1.0, "beta": 1.5, "p": 1.0})
    monkeypatch.setattr(
        cs_module.d,
        "generate_levy_distribution",
        lambda **kwargs: torch.ones(kwargs["size"], device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *size, **kwargs: torch.full(size, 2.0, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        torch, "rand", lambda *size, **kwargs: torch.zeros(size, device=kwargs["device"], dtype=kwargs["dtype"])
    )

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([6.0, 12.0]))
    torch.testing.assert_close(population.fitness, torch.tensor([-6.0, -12.0]))


def test_ba_initializes_state_ranges_and_rejects_a_worse_candidate(monkeypatch):
    population = _population(torch.tensor([[1.0]]), lower_bound=-10.0, upper_bound=10.0)
    function = Function(lambda position: position.square().sum())
    population.fitness = function(population.positions)
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = BA({"f_min": 0.0, "f_max": 2.0, "A": 0.5, "r": 0.5})
    draws = iter([0.5, 0.5, 0.5, 0.5, 0.0, 0.5])
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(
            size,
            next(draws),
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        ),
    )
    optimizer.compile(population)

    optimizer.update(_context(population, function))

    assert optimizer.frequency.item() == 1.0
    assert optimizer.loudness.item() == 0.25
    assert optimizer.pulse_rate.item() == 0.25
    assert optimizer.velocity.item() == 1.0
    assert population.positions.item() == 1.0
    assert population.fitness.item() == 1.0


def test_ba_local_walk_uses_independent_coordinate_noise(monkeypatch):
    population = _population(torch.tensor([[[0.0], [0.0]]]), lower_bound=-2.0, upper_bound=2.0)
    function = Function(lambda position: (position[0] - 1).square().sum() + (position[1] + 1).square().sum())
    population.fitness = function(population.positions)
    population.best_position.zero_()
    population.best_fitness.fill_(2.0)
    optimizer = BA({"f_min": 0.0, "f_max": 0.0, "A": 1.0, "r": 0.0})
    optimizer.compile(population)
    optimizer.loudness.fill_(1.0)
    optimizer.pulse_rate.zero_()
    draws = iter([0.0, 1.0, 0.0])
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, next(draws), device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        torch,
        "randn_like",
        lambda tensor: torch.tensor([[[1.0], [-1.0]]], device=tensor.device, dtype=tensor.dtype),
    )

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[0, :, 0], torch.tensor([0.001, -0.001]))
    assert population.fitness.item() < 2.0


@pytest.mark.parametrize(
    ("optimizer_class", "params", "name"),
    [
        (PSO, {"w": float("nan")}, "w"),
        (PSO, {"c1": float("inf")}, "c1"),
        (AIWPSO, {"w_min": float("nan")}, "w_min"),
        (JS, {"beta": float("inf")}, "beta"),
        (JS, {"gamma": float("nan")}, "gamma"),
    ],
)
def test_recent_current_parameter_validation_is_preserved(optimizer_class, params, name):
    with pytest.raises(ValueError, match=name):
        optimizer_class(params)


def test_aiwpso_limits_are_order_independent_and_may_exceed_one():
    optimizer = AIWPSO({"w_max": 3.0, "w_min": 2.0})

    assert optimizer.w_min == 2.0
    assert optimizer.w_max == 3.0
