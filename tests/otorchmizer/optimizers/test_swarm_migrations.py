# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Canonical migration contracts for swarm optimizers."""

from types import SimpleNamespace

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.optimizers.swarm import ABO, FSO, JS, NBJS, PIO, SFO, SSO
from otorchmizer.optimizers.swarm import fso as fso_module


def _population(
    positions: torch.Tensor,
    lower_bound: float | torch.Tensor = -10.0,
    upper_bound: float | torch.Tensor = 10.0,
) -> Population:
    if positions.ndim == 2:
        positions = positions.unsqueeze(-1)
    n_agents, n_variables, n_dimensions = positions.shape
    lower = torch.as_tensor(lower_bound).expand(n_variables)
    upper = torch.as_tensor(upper_bound).expand(n_variables)
    population = Population(
        n_agents,
        n_variables,
        n_dimensions,
        lower,
        upper,
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


def test_abo_exposes_canonical_parameters_only():
    optimizer = ABO({"sunspot_ratio": 0.75, "a": 1.5})

    assert optimizer.sunspot_ratio == 0.75
    assert optimizer.a == 1.5
    assert not hasattr(optimizer, "starvation_ratio")


def test_canonical_migrations_remove_mislabeled_parameter_surfaces():
    assert not hasattr(FSO(), "step_individual")
    assert not hasattr(SSO(), "female_percentage")
    assert not hasattr(PIO(), "n_c")


@pytest.mark.parametrize(
    ("optimizer_class", "params", "name"),
    [
        (ABO, {"sunspot_ratio": 1.1}, "sunspot_ratio"),
        (ABO, {"a": -1.0}, "a"),
        (FSO, {"beta": 0.0}, "beta"),
        (PIO, {"n_c1": 0}, "n_c1"),
        (PIO, {"n_c2": 100}, "n_c2"),
        (SFO, {"PP": 0.0}, "PP"),
        (SFO, {"A": -1.0}, "A"),
        (SFO, {"e": -1.0}, "e"),
        (JS, {"eta": 4.1}, "eta"),
        (JS, {"beta": 0.0}, "beta"),
        (JS, {"gamma": 0.0}, "gamma"),
    ],
)
def test_canonical_parameters_reject_invalid_ranges(optimizer_class, params, name):
    with pytest.raises(e.ValueError, match=name):
        optimizer_class(params)


def test_abo_sunspot_flight_updates_one_variable_greedily(monkeypatch):
    population = _population(torch.tensor([[1.0], [3.0]]))
    function = Function(lambda position: -position.sum())
    population.fitness = function(population.positions)
    population.update_best()
    optimizer = ABO({"sunspot_ratio": 1.0})
    integers = iter([1, 0, 0, 0])

    def controlled_randint(*args, **kwargs):
        return torch.tensor(next(integers), device=kwargs.get("device"))

    monkeypatch.setattr(torch, "randint", controlled_randint)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: torch.tensor(0.75, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([4.0, 1.0]))
    torch.testing.assert_close(population.fitness, function(population.positions))
    assert population.best_position.item() == 4.0


def test_abo_zero_sunspots_uses_the_full_canopy_pool(monkeypatch):
    population = _population(torch.tensor([[1.0], [2.0]]))
    population.fitness.zero_()
    optimizer = ABO({"sunspot_ratio": 0.0})
    upper_bounds = []

    def controlled_randint(low, high, size, **kwargs):
        upper_bounds.append(high)
        return torch.tensor(0, device=kwargs.get("device"))

    monkeypatch.setattr(torch, "randint", controlled_randint)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: torch.tensor(0.5, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    optimizer.update(_context(population, Function(lambda position: position.sum() * 0)))

    assert upper_bounds == [2, 1, 2, 2, 1, 2]


def test_abo_one_agent_default_runs_canopy_free_flight(monkeypatch):
    population = _population(torch.tensor([[2.0]]))
    function = Function(lambda position: (position**2).sum())
    population.fitness = function(population.positions)
    population.update_best()
    optimizer = ABO()
    draws = iter([0.5, 0.0, 0.0])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(0, device=kwargs.get("device")))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: torch.tensor(
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        ),
    )

    optimizer.update(_context(population, function))

    assert population.positions.item() == -2.0
    torch.testing.assert_close(population.fitness, function(population.positions))


def test_fso_uses_sigma_reduction_and_beta_expansion(monkeypatch):
    population = _population(torch.tensor([[2.0], [4.0]]))
    population.fitness = -population.positions[:, 0, 0].clone()
    population.best_position.fill_(5.0)
    population.best_fitness.fill_(-5.0)
    function = Function(lambda position: -position.sum())
    optimizer = FSO({"beta": 0.5})
    observed = {}

    monkeypatch.setattr(
        torch,
        "randn",
        lambda size, **kwargs: torch.zeros(size, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    def controlled_levy(beta, size, device, dtype):
        observed["beta"] = beta
        observed["size"] = size
        observed["dtype"] = dtype
        return torch.full(size, -0.5, device=device, dtype=dtype)

    monkeypatch.setattr(fso_module.d, "generate_levy_distribution", controlled_levy)

    optimizer.update(_context(population, function, iteration=1, n_iterations=10))

    assert observed == {"beta": pytest.approx(0.8), "size": (2, 1, 1), "dtype": torch.float32}
    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([6.5, 5.5]))
    torch.testing.assert_close(population.fitness, function(population.positions))
    assert population.best_position.item() == 6.5
    assert population.best_fitness.item() == -6.5


def test_fso_preserves_multidimensional_shape_and_dtype():
    torch.manual_seed(3)
    positions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ],
        dtype=torch.float64,
    )
    population = _population(positions)
    function = Function(lambda position: (position**2).sum())
    optimizer = FSO()
    optimizer.evaluate(population, function)

    optimizer.update(_context(population, function))

    assert population.positions.shape == positions.shape
    assert population.positions.dtype == torch.float64
    assert torch.isfinite(population.positions).all()
    torch.testing.assert_close(population.fitness, function(population.positions))


def test_sso_evaluation_tracks_personal_bests_separately():
    population = _population(torch.tensor([[1.0], [2.0]]))
    function = Function(lambda position: (position**2).sum())
    optimizer = SSO()
    optimizer.compile(population)
    optimizer.evaluate(population, function)
    population.positions[:, 0, 0] = torch.tensor([3.0, 0.5])

    optimizer.evaluate(population, function)

    torch.testing.assert_close(population.fitness, torch.tensor([9.0, 0.25]))
    torch.testing.assert_close(optimizer.local_fitness, torch.tensor([1.0, 0.25]))
    torch.testing.assert_close(optimizer.local_position[:, 0, 0], torch.tensor([1.0, 0.5]))


def test_sso_applies_all_four_canonical_sources(monkeypatch):
    positions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    population = _population(positions, lower_bound=0.0, upper_bound=200.0)
    population.best_position = torch.tensor([[100.0], [110.0], [120.0], [130.0]])
    optimizer = SSO()
    optimizer.local_position = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    optimizer.local_fitness = torch.tensor([0.0])
    threshold = torch.tensor([[[0.05], [0.2], [0.6], [0.95]]])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: threshold.to(**kwargs))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))

    optimizer.update(_context(population, Function(lambda position: position.sum())))

    expected = torch.tensor([[[1.0], [20.0], [120.0], [50.0]]])
    torch.testing.assert_close(population.positions, expected)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("C_w", -0.1),
        ("C_p", 0.05),
        ("C_g", 0.2),
    ],
)
def test_sso_rejects_invalid_probability_thresholds(name, value):
    optimizer = SSO()

    with pytest.raises(e.ValueError, match=name):
        setattr(optimizer, name, value)


def test_sso_coupled_thresholds_are_independent_of_parameter_order():
    optimizer = SSO({"C_g": 0.2, "C_p": 0.15, "C_w": 0.05})

    assert (optimizer.C_w, optimizer.C_p, optimizer.C_g) == (0.05, 0.15, 0.2)
    optimizer.build({"C_g": 0.3, "C_p": 0.2, "C_w": 0.1})
    assert (optimizer.C_w, optimizer.C_p, optimizer.C_g) == (0.1, 0.2, 0.3)


def test_sso_build_rejects_an_invalid_single_coupled_override_atomically():
    optimizer = SSO()
    before = (optimizer.C_w, optimizer.C_p, optimizer.C_g)

    with pytest.raises(e.ValueError, match="C_p"):
        optimizer.build({"C_w": 0.5})

    assert (optimizer.C_w, optimizer.C_p, optimizer.C_g) == before


def test_pio_map_and_compass_operator(monkeypatch):
    population = _population(torch.tensor([[2.0], [4.0]]))
    population.best_position.zero_()
    optimizer = PIO({"n_c1": 2, "n_c2": 4, "R": 0.0})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    optimizer.update(_context(population, Function(lambda position: position.sum()), iteration=0))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(optimizer.velocity[:, 0, 0], torch.tensor([-1.0, -2.0]))


def test_pio_landmark_operator_reduces_active_pigeons(monkeypatch):
    population = _population(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    population.fitness = torch.tensor([1.0, 2.0, 3.0, 4.0])
    optimizer = PIO({"n_c1": 2, "n_c2": 4})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    optimizer.update(_context(population, Function(lambda position: position.sum()), iteration=2))

    center = torch.tensor(14 / 18)
    torch.testing.assert_close(population.positions[:, 0, 0], (torch.arange(1.0, 5.0) + center) / 2)
    assert optimizer.n_p == 3


def test_pio_stops_after_landmark_threshold():
    population = _population(torch.tensor([[1.0], [2.0]]))
    optimizer = PIO({"n_c1": 2, "n_c2": 4})
    optimizer.compile(population)
    before = population.positions.clone()

    optimizer.update(_context(population, Function(lambda position: position.sum()), iteration=4))

    assert torch.equal(population.positions, before)


def test_pio_landmark_center_remains_finite_when_fitness_sum_is_zero(monkeypatch):
    population = _population(torch.tensor([[2.0], [4.0]]))
    population.fitness = torch.zeros(2)
    optimizer = PIO({"n_c1": 1, "n_c2": 2})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    optimizer.update(_context(population, Function(lambda position: position.sum()), iteration=1))

    assert torch.isfinite(population.positions).all()
    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([1.75, 2.75]))


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_pio_landmark_center_is_invariant_to_positive_fitness_scale(monkeypatch, scale):
    positions = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    population = _population(positions)
    population.fitness = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64) * scale
    optimizer = PIO({"n_c2": 4, "n_c1": 2})
    optimizer.compile(population)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    optimizer.update(_context(population, Function(lambda position: position.sum()), iteration=2))

    center = torch.tensor(14 / 18, dtype=torch.float64)
    expected = (torch.arange(1.0, 5.0, dtype=torch.float64) + center) / 2
    torch.testing.assert_close(population.positions[:, 0, 0], expected)


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_pio_rejects_nonfinite_population_fitness(invalid):
    population = _population(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    population.fitness = torch.tensor([1.0, 2.0, 3.0, invalid])
    optimizer = PIO({"n_c1": 1, "n_c2": 2})
    optimizer.compile(population)

    with pytest.raises(e.ValueError, match="population.fitness"):
        optimizer.update(_context(population, Function(lambda position: position.sum()), iteration=1))


def test_pio_coupled_thresholds_are_independent_of_parameter_order():
    optimizer = PIO({"n_c2": 20, "n_c1": 10})

    assert (optimizer.n_c1, optimizer.n_c2) == (10, 20)
    optimizer.build({"n_c2": 40, "n_c1": 30})
    assert (optimizer.n_c1, optimizer.n_c2) == (30, 40)


def test_pio_build_rejects_an_invalid_single_coupled_override_atomically():
    optimizer = PIO({"n_c1": 10, "n_c2": 20})
    before = (optimizer.n_c1, optimizer.n_c2)

    with pytest.raises(e.ValueError, match="n_c2"):
        optimizer.build({"n_c1": 30})

    assert (optimizer.n_c1, optimizer.n_c2) == before


def test_js_compile_maps_logistic_sequence_to_bounds(monkeypatch):
    positions = torch.zeros(3, 1, 1, dtype=torch.float64)
    population = _population(positions, lower_bound=-2.0, upper_bound=2.0)
    optimizer = JS()
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.25))

    optimizer.compile(population)

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))


def test_js_ocean_current_phase_uses_population_mean(monkeypatch):
    population = _population(torch.tensor([[1.0], [3.0]]), lower_bound=-10.0, upper_bound=10.0)
    population.best_position.fill_(1.0)
    population.fitness = torch.tensor([1.0, 3.0])
    optimizer = JS({"beta": 1.0})
    draws = iter([1.0, 0.0, 0.5, 0.0, 0.0, 0.0])

    def controlled_rand(*size, **kwargs):
        return torch.full(
            size,
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.zeros_like(tensor))
    monkeypatch.setattr(
        torch, "randint", lambda *args, **kwargs: torch.zeros(args[2], device=kwargs.get("device"), dtype=torch.long)
    )

    optimizer.update(_context(population, Function(lambda position: position.sum())))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([1.5, 3.5]))


def test_nbjs_changes_only_passive_motion_scale(monkeypatch):
    population = _population(torch.zeros(2, 2, 1), lower_bound=-5.0, upper_bound=5.0)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.full(size, 0.5, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
    )

    js_motion = JS({"gamma": 0.2})._motion_a(population)
    nbjs_motion = NBJS({"gamma": 0.2})._motion_a(population)

    torch.testing.assert_close(js_motion, torch.ones_like(population.positions))
    torch.testing.assert_close(nbjs_motion, torch.full((2, 1, 1), 0.1))


def test_sfo_consumes_promoted_sardines_and_replenishes_prey(monkeypatch):
    population = _population(torch.tensor([[10.0], [11.0]]), lower_bound=-100.0, upper_bound=100.0)
    function = Function(lambda positions: positions[:, 0, 0], batch=True)
    population.fitness = function(population.positions)
    population.best_position.fill_(10.0)
    population.best_fitness.fill_(10.0)
    optimizer = SFO({"PP": 0.5, "A": 0.0})
    optimizer.compile(population)
    optimizer.sardine_positions[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    optimizer.sardine_fitness = function(optimizer.sardine_positions)
    draws = iter([1.0, 0.0])

    def controlled_rand(*size, **kwargs):
        return torch.full(
            size,
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))

    optimizer.update(_context(population, function))

    torch.testing.assert_close(population.positions[:, 0, 0], torch.tensor([1.0, 2.0]))
    assert optimizer.sardine_positions.shape[0] == optimizer.n_sardines == 4
    assert not torch.isin(torch.tensor([1.0, 2.0]), optimizer.sardine_positions[:, 0, 0]).any()
    torch.testing.assert_close(optimizer.sardine_fitness, function(optimizer.sardine_positions))
    assert population.best_position.item() == 1.0


def test_sfo_archives_initial_prey_best_before_the_prey_moves(monkeypatch):
    population = _population(torch.tensor([[10.0], [11.0]]), lower_bound=-100.0, upper_bound=100.0)
    function = Function(lambda positions: positions[:, 0, 0], batch=True)
    population.fitness = function(population.positions)
    population.best_position.fill_(10.0)
    population.best_fitness.fill_(10.0)
    optimizer = SFO({"PP": 0.5, "A": 4.0})
    optimizer.compile(population)
    optimizer.sardine_positions[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    optimizer.sardine_fitness.fill_(torch.inf)
    draws = iter([1.0, 0.0, 1.0])

    def controlled_rand(*size, **kwargs):
        return torch.full(
            size,
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))

    optimizer.update(_context(population, function))

    assert population.best_fitness.item() == 1.0
    assert population.best_position.item() == 1.0


def test_sfo_archives_replenished_prey_before_the_next_iteration(monkeypatch):
    population = _population(torch.tensor([[10.0], [11.0]]), lower_bound=-100.0, upper_bound=100.0)
    function = Function(lambda positions: positions[:, 0, 0], batch=True)
    population.fitness = function(population.positions)
    population.best_position.fill_(10.0)
    population.best_fitness.fill_(10.0)
    optimizer = SFO({"PP": 0.5, "A": 0.0})
    optimizer.compile(population)
    optimizer.sardine_positions[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    optimizer.sardine_fitness = function(optimizer.sardine_positions)
    draws = iter([1.0, 0.0])

    def controlled_rand(*size, **kwargs):
        return torch.full(
            size,
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )

    monkeypatch.setattr(torch, "rand", controlled_rand)
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)

    optimizer.update(_context(population, function))

    assert population.best_fitness.item() == -100.0
    assert population.best_position.item() == -100.0
    torch.testing.assert_close(optimizer.sardine_fitness, function(optimizer.sardine_positions))


def test_canonical_swarm_exports_include_nbjs():
    from otorchmizer.optimizers import swarm

    assert "NBJS" in swarm.__all__
    assert swarm.NBJS is NBJS


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize(
    ("optimizer_class", "params"),
    [
        (ABO, None),
        (FSO, None),
        (SSO, None),
        (PIO, {"n_c1": 2, "n_c2": 4}),
        (SFO, {"PP": 0.5}),
        (JS, None),
        (NBJS, None),
    ],
)
def test_migrated_swarm_optimizers_preserve_dtype_shape_and_matching_fitness(
    optimizer_class,
    params,
    dtype,
):
    torch.manual_seed(4)
    population = Population(
        6,
        3,
        2,
        torch.full((3,), -2.0),
        torch.full((3,), 3.0),
        dtype=dtype,
    )
    population.initialize_uniform()
    function = Function(lambda position: (position**2).sum())
    optimizer = optimizer_class(params)
    optimizer.compile(population)
    optimizer.evaluate(population, function)

    optimizer.update(_context(population, function))
    population.clip()
    optimizer.evaluate(population, function)

    assert population.positions.shape == (6, 3, 2)
    assert population.positions.dtype == dtype
    assert population.fitness.dtype == dtype
    assert torch.isfinite(population.positions).all()
    assert torch.isfinite(population.fitness).all()
    torch.testing.assert_close(population.fitness, function(population.positions))
