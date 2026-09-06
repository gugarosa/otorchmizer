# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Current-upstream parity contracts for miscellaneous, science, social, and Boolean optimizers."""

import importlib
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.optimizers.boolean import UMDA
from otorchmizer.optimizers.misc import CEM
from otorchmizer.optimizers.science import AIG, MOA, MVO
from otorchmizer.optimizers.social import ISA


def _population(n_agents=4, n_variables=1, n_dimensions=1, lower=-10.0, upper=10.0):
    return Population(
        n_agents=n_agents,
        n_variables=n_variables,
        n_dimensions=n_dimensions,
        lower_bound=torch.full((n_variables,), lower),
        upper_bound=torch.full((n_variables,), upper),
    )


def _context(population, function, iteration=0, n_iterations=10):
    return UpdateContext(
        space=SimpleNamespace(population=population),
        function=function,
        iteration=iteration,
        n_iterations=n_iterations,
        device=population.device,
    )


def _sphere(position):
    return position.square().sum()


def _constant_random(value):
    def generate(*shape, **kwargs):
        size = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return torch.full(size, value, device=kwargs.get("device"), dtype=kwargs.get("dtype"))

    return generate


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("science.aig", "AIG"),
        ("science.cdo", "CDO"),
        ("science.efo", "EFO"),
        ("science.esa", "ESA"),
        ("science.hgso", "HGSO"),
        ("science.lsa", "LSA"),
        ("science.moa", "MOA"),
        ("science.sma", "SMA"),
        ("science.teo", "TEO"),
        ("science.two", "TWO"),
        ("science.weo", "WEO"),
        ("social.bso", "BSO"),
        ("social.ci", "CI"),
        ("social.isa", "ISA"),
        ("social.mvpa", "MVPA"),
        ("social.qsa", "QSA"),
        ("social.ssd", "SSD"),
        ("boolean.bmrfo", "BMRFO"),
        ("boolean.bpso", "BPSO"),
        ("boolean.umda", "UMDA"),
    ],
)
def test_current_module_paths_export_canonical_classes(module_name, class_name):
    module = importlib.import_module(f"otorchmizer.optimizers.{module_name}")
    family = module_name.split(".", 1)[0]
    package = importlib.import_module(f"otorchmizer.optimizers.{family}")

    assert getattr(module, class_name) is getattr(package, class_name)


@pytest.mark.parametrize(
    "module_name",
    [
        "otorchmizer.optimizers.science.science_extra",
        "otorchmizer.optimizers.social.social",
        "otorchmizer.optimizers.boolean.boolean",
    ],
)
def test_obsolete_grouped_modules_are_removed(module_name):
    assert importlib.util.find_spec(module_name) is None


def test_aig_uses_shared_random_limit_and_coordinate_gaussian_scale(monkeypatch):
    population = _population(n_agents=1, n_variables=2)
    population.positions.fill_(1)
    population.fitness.fill_(-2)
    population.best_position.fill_(1)
    population.best_fitness.fill_(-2)
    optimizer = AIG()
    monkeypatch.setattr(torch, "rand", _constant_random(0.5))
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    function = Function(lambda position: -position.sum())

    optimizer.update(_context(population, function, iteration=10, n_iterations=10))

    expected = 1 / torch.cos(torch.tensor(torch.pi / 6)) ** 2
    assert torch.allclose(population.positions, torch.full_like(population.positions, expected))


def test_aig_preserves_finite_near_singular_reciprocal(monkeypatch):
    population = Population(
        n_agents=1,
        n_variables=1,
        n_dimensions=1,
        lower_bound=torch.tensor([-1.0], dtype=torch.float64),
        upper_bound=torch.tensor([1.0], dtype=torch.float64),
        dtype=torch.float64,
    )
    population.positions.fill_(1e-20)
    population.fitness.fill_(-1e-20)
    population.best_position.copy_(population.positions[0])
    population.best_fitness.copy_(population.fitness[0])
    optimizer = AIG()
    monkeypatch.setattr(torch, "rand", _constant_random(1.0))
    corrections = iter([torch.full_like(population.positions[0], 1.5), torch.zeros_like(population.positions[0])])
    monkeypatch.setattr(torch, "randn_like", lambda _tensor: next(corrections))

    optimizer.update(_context(population, Function(lambda position: -position.abs().sum())))

    expected = population.positions.new_tensor(1e-20) / torch.cos(population.positions.new_tensor(torch.pi / 2))
    assert population.positions.item() == pytest.approx(expected.item())


def test_moa_requires_square_population():
    with pytest.raises(ValueError, match="perfect square"):
        MOA().compile(_population(n_agents=5))


def test_moa_uses_four_toroidal_neighbors(monkeypatch):
    population = _population(n_agents=9, lower=-20, upper=20)
    population.positions[:, 0, 0] = torch.arange(8, -1, -1)
    population.fitness = torch.arange(8, -1, -1, dtype=population.dtype)
    optimizer = MOA({"alpha": 1.0, "rho": 0.0})
    optimizer.compile(population)
    monkeypatch.setattr(torch, "rand", _constant_random(1.0))

    optimizer.update(_context(population, Function(_sphere)))

    assert population.positions[0].item() == pytest.approx(1.5)
    assert population.positions[4].item() == pytest.approx(5.0)


def test_moa_fitness_normalization_is_scale_invariant(monkeypatch):
    normalized = torch.linspace(-1, 1, 9, dtype=torch.float64)
    results = []
    monkeypatch.setattr(torch, "rand", _constant_random(1.0))

    for scale in (1.0, 1e308):
        population = Population(
            n_agents=9,
            n_variables=1,
            n_dimensions=1,
            lower_bound=torch.tensor([-20.0], dtype=torch.float64),
            upper_bound=torch.tensor([20.0], dtype=torch.float64),
            dtype=torch.float64,
        )
        population.positions[:, 0, 0] = torch.arange(9, dtype=torch.float64)
        population.fitness = normalized * scale
        optimizer = MOA({"alpha": 1.0, "rho": 0.0})
        optimizer.compile(population)
        optimizer.update(_context(population, Function(_sphere)))
        results.append(population.positions.clone())

    assert torch.isfinite(results[1]).all()
    assert torch.allclose(results[0], results[1])


def test_moa_equal_fitness_has_zero_force(monkeypatch):
    population = _population(n_agents=9, lower=-20, upper=20)
    population.positions[:, 0, 0] = torch.arange(9)
    population.fitness.fill_(3)
    optimizer = MOA()
    optimizer.compile(population)
    before = population.positions.clone()
    monkeypatch.setattr(torch, "rand", _constant_random(1.0))

    optimizer.update(_context(population, Function(_sphere)))

    assert torch.equal(population.positions, before)


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_moa_rejects_nonfinite_fitness_before_movement(invalid):
    population = _population(n_agents=9)
    population.positions[:, 0, 0] = torch.arange(9)
    population.fitness.zero_()
    population.fitness[4] = invalid
    optimizer = MOA()
    optimizer.compile(population)
    before = population.positions.clone()

    with pytest.raises(ValueError, match="finite values for MOA"):
        optimizer.update(_context(population, Function(_sphere)))

    assert torch.equal(population.positions, before)


def test_mvo_wormhole_draw_includes_lower_bound(monkeypatch):
    population = _population(n_agents=1, lower=-5, upper=5)
    population.positions.zero_()
    population.fitness.zero_()
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = MVO({"WEP_min": 1.0, "WEP_max": 1.0})
    draws = iter([1.0, 0.0, 0.0, 0.5])

    def random(*shape, **kwargs):
        size = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return torch.full(size, next(draws), device=kwargs.get("device"), dtype=kwargs.get("dtype"))

    monkeypatch.setattr(torch, "rand", random)
    optimizer.update(_context(population, Function(_sphere), iteration=0, n_iterations=16))

    assert torch.count_nonzero(population.positions) == 0


def test_isa_archives_weighted_position_independently(monkeypatch):
    population = _population(n_agents=2)
    population.positions[:, 0, 0] = population.positions.new_tensor([-2.0, 2.0])
    optimizer = ISA({"w": 0.0, "tau": 0.0})
    function = Function(_sphere)
    optimizer.compile(population)
    optimizer.evaluate(population, function)
    monkeypatch.setattr(torch, "rand", _constant_random(0.0))

    with patch("torch.randint", side_effect=[torch.tensor([1]), torch.tensor([0])]):
        optimizer.update(_context(population, function))

    assert torch.count_nonzero(optimizer.local_position) == 0
    assert torch.count_nonzero(optimizer.local_fitness) == 0
    assert population.best_fitness.item() == 0


def test_isa_equal_fitness_uses_valid_uniform_weighted_position(monkeypatch):
    population = _population(n_agents=2, lower=1, upper=2)
    population.positions.fill_(1)
    optimizer = ISA({"w": 0.0, "tau": 0.0})
    function = Function(lambda position: position.sum())
    optimizer.compile(population)
    optimizer.evaluate(population, function)
    monkeypatch.setattr(torch, "rand", _constant_random(0.0))

    with patch("torch.randint", side_effect=[torch.tensor([1]), torch.tensor([0])]):
        optimizer.update(_context(population, function))

    assert torch.equal(population.positions, torch.ones_like(population.positions))
    assert torch.equal(population.best_position, torch.ones_like(population.best_position))
    assert torch.equal(optimizer.local_position, torch.ones_like(optimizer.local_position))


def test_isa_weighted_position_is_scale_invariant_for_large_fitness():
    positions = [1.0, 2.0, 3.0, 4.0]
    weighted_positions = []

    for scale in (1.0, 1e38):
        population = _population(n_agents=4, lower=1, upper=4)
        population.positions[:, 0, 0] = population.positions.new_tensor(positions)
        population.fitness = (population.positions[:, 0, 0] - 1) * scale
        population.best_position.fill_(1)
        population.best_fitness.zero_()
        optimizer = ISA({"w": 0.0, "tau": 0.0})
        optimizer.compile(population)
        optimizer.local_position.copy_(population.positions)
        optimizer.local_fitness.copy_(population.fitness)
        draws = iter([0.0, 0.0, 0.0, 0.0] * 4)
        function = Function(lambda position, scale=scale: (position.sum() - 1) * scale)

        with (
            patch(
                "torch.rand",
                side_effect=lambda *shape, **kwargs: torch.full(
                    shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape,
                    next(draws),
                    device=kwargs.get("device"),
                    dtype=kwargs.get("dtype"),
                ),
            ),
            patch(
                "torch.randint",
                side_effect=[torch.tensor([1]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])],
            ),
        ):
            optimizer.update(_context(population, function))

        weighted_positions.append(optimizer.local_position[3].item())

    assert weighted_positions == pytest.approx([20 / 6, 20 / 6])


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_isa_rejects_nonfinite_fitness_before_movement(invalid):
    population = _population(n_agents=2)
    population.fitness.zero_()
    population.fitness[1] = invalid
    optimizer = ISA()
    optimizer.compile(population)
    before = population.positions.clone()

    with pytest.raises(ValueError, match="finite values for ISA"):
        optimizer.update(_context(population, Function(_sphere)))

    assert torch.equal(population.positions, before)


def test_isa_movement_uses_prior_memory_before_weighted_selection(monkeypatch):
    population = _population(n_agents=2, lower=-100, upper=100)
    population.positions[:, 0, 0] = population.positions.new_tensor([0.0, 2.0])
    population.fitness = population.positions.square().sum(dim=(-1, -2))
    population.best_position.zero_()
    population.best_fitness.zero_()
    optimizer = ISA({"w": 0.0, "tau": 0.0})
    optimizer.compile(population)
    optimizer.local_position.fill_(10)
    optimizer.local_fitness.fill_(100)
    draws = iter([0.5, 0.0, 0.5, 1.0] * 2)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, **kwargs: torch.full(
            shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape,
            next(draws),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        ),
    )

    with patch("torch.randint", side_effect=[torch.tensor([1]), torch.tensor([0])]):
        optimizer.update(_context(population, Function(_sphere)))

    assert population.positions[:, 0, 0].tolist() == pytest.approx([-20.0, -16.0])
    assert optimizer.local_position[:, 0, 0].tolist() == pytest.approx([2.0, 2.0])
    assert optimizer.local_fitness.tolist() == pytest.approx([4.0, 4.0])


def test_cem_accepts_current_numeric_scalars_and_extrapolation():
    optimizer = CEM({"n_updates": np.int64(2), "alpha": np.float32(2.0)})

    assert optimizer.n_updates == 2
    assert optimizer.alpha == 2.0


@pytest.mark.parametrize("alpha", [-0.1, np.nan, np.inf])
def test_cem_rejects_invalid_alpha_before_sampling(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CEM({"alpha": alpha})


def test_cem_reassignment_uses_same_numeric_contract():
    optimizer = CEM()
    optimizer.n_updates = np.int64(3)
    optimizer.alpha = np.float32(0.5)

    assert optimizer.n_updates == 3
    assert optimizer.alpha == 0.5
    with pytest.raises(ValueError, match="alpha"):
        optimizer.alpha = -1


def test_umda_accepts_current_real_scalar_parameters():
    optimizer = UMDA(
        {
            "p_selection": np.float32(0.5),
            "lower_bound": np.float64(0.1),
            "upper_bound": np.int64(1),
        }
    )

    assert optimizer.p_selection == 0.5
    assert optimizer.lower_bound == 0.1
    assert optimizer.upper_bound == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("p_selection", 0),
        ("lower_bound", np.nan),
        ("lower_bound", 2),
        ("upper_bound", -1),
        ("upper_bound", np.inf),
    ],
)
def test_umda_reassignment_rejects_invalid_parameters(name, value):
    optimizer = UMDA()

    with pytest.raises((TypeError, ValueError), match=name):
        setattr(optimizer, name, value)


@pytest.mark.parametrize(
    "params",
    [
        {"lower_bound": 1.0, "upper_bound": 1.0},
        {"upper_bound": 0.2, "lower_bound": 0.1},
    ],
)
def test_umda_build_applies_coupled_bounds_atomically(params):
    optimizer = UMDA()

    optimizer.build(params)

    assert optimizer.lower_bound == params["lower_bound"]
    assert optimizer.upper_bound == params["upper_bound"]
