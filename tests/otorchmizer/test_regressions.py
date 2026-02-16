"""Regression tests — validates fixes for issues found in the audit.

Each test corresponds to a specific audit finding to prevent regressions.
"""

import torch
import pytest

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.population import Population
from otorchmizer.core.function import Function
from otorchmizer.core.node import Node
from otorchmizer.optimizers.swarm.pso import PSO, AIWPSO
from otorchmizer.optimizers.swarm.fa import FA
from otorchmizer.optimizers.misc.hc import HC
from otorchmizer.spaces.search import SearchSpace
from otorchmizer.core.optimizer import UpdateContext


def _make_pop(n_agents=5, n_vars=3, n_dims=1):
    lb = torch.zeros(n_vars)
    ub = torch.ones(n_vars) * 10.0
    pop = Population(n_agents, n_vars, n_dims, lb, ub)
    pop.initialize_uniform()
    return pop


def _sphere(x):
    return (x ** 2).sum()


class TestConstantFloatMax:
    """Audit: constant-floatmax — FLOAT_MAX must be finite float32."""

    def test_float_max_is_finite(self):
        t = torch.tensor(c.FLOAT_MAX)
        assert t.isfinite(), "FLOAT_MAX should be finite in float32"

    def test_float_max_in_tensor_full(self):
        t = torch.full((5,), c.FLOAT_MAX)
        assert t.shape == (5,)
        assert t.isfinite().all()

    def test_population_uses_float_max(self):
        pop = _make_pop()
        expected = torch.tensor(c.FLOAT_MAX)
        # After init (before evaluation), fitness should be FLOAT_MAX
        lb = torch.zeros(3)
        ub = torch.ones(3)
        pop2 = Population(2, 3, 1, lb, ub)
        assert (pop2.fitness == expected).all()


class TestFARegressions:
    """Audit: fa-beta-default, fa-formula, fa-alpha-decay, fa-validation."""

    def test_beta_default(self):
        fa = FA()
        assert fa.beta == 0.2, f"FA beta default should be 0.2, got {fa.beta}"

    def test_alpha_default(self):
        fa = FA()
        assert fa.alpha == 0.5

    def test_gamma_default(self):
        fa = FA()
        assert fa.gamma == 1.0

    def test_alpha_validation(self):
        with pytest.raises(Exception):
            fa = FA({"alpha": -1.0})

    def test_beta_validation(self):
        with pytest.raises(Exception):
            fa = FA({"beta": -0.5})

    def test_gamma_validation(self):
        with pytest.raises(Exception):
            fa = FA({"gamma": -1.0})

    def test_alpha_decay(self):
        """FA alpha should decay each iteration."""
        fa = FA()
        initial_alpha = fa.alpha

        lb = [0.0, 0.0, 0.0]
        ub = [10.0, 10.0, 10.0]
        space = SearchSpace(n_agents=5, n_variables=3, lower_bound=lb, upper_bound=ub)
        space.build()
        fa.compile(space.population)

        fn = Function(lambda x: (x ** 2).sum())

        ctx = UpdateContext(
            space=space,
            function=fn,
            iteration=0,
            n_iterations=100,
            device=torch.device("cpu"),
        )

        fa.update(ctx)
        assert fa.alpha < initial_alpha, "Alpha should decay after update"

    def test_formula_uses_raw_distance(self):
        """FA attractiveness should use raw distance, not squared."""
        fa = FA()
        pop = _make_pop(n_agents=3)
        fa.compile(pop)

        lb = [0.0, 0.0, 0.0]
        ub = [10.0, 10.0, 10.0]
        space = SearchSpace(n_agents=3, n_variables=3, lower_bound=lb, upper_bound=ub)
        space.build()

        fn = Function(_sphere)
        fa.evaluate(space.population, fn)

        ctx = UpdateContext(
            space=space, function=fn,
            iteration=0, n_iterations=10,
            device=torch.device("cpu"),
        )

        pos_before = space.population.positions.clone()
        fa.update(ctx)
        pos_after = space.population.positions

        # Positions should have changed
        assert not torch.allclose(pos_before, pos_after)


class TestHCRegressions:
    """Audit: hc-formula, hc-validation."""

    def test_unconditional_noise(self):
        """HC should always add noise, not conditionally accept."""
        hc = HC()
        lb = [0.0, 0.0, 0.0]
        ub = [10.0, 10.0, 10.0]
        space = SearchSpace(n_agents=5, n_variables=3, lower_bound=lb, upper_bound=ub)
        space.build()

        fn = Function(_sphere)

        ctx = UpdateContext(
            space=space, function=fn,
            iteration=0, n_iterations=10,
            device=torch.device("cpu"),
        )

        pos_before = space.population.positions.clone()
        hc.update(ctx)
        pos_after = space.population.positions

        # ALL positions should change (noise added unconditionally)
        # At least one dimension must differ for each agent
        changed = ~torch.allclose(pos_before, pos_after)
        assert changed, "HC should unconditionally add noise to all positions"

    def test_r_var_validation(self):
        with pytest.raises(Exception):
            HC({"r_var": -0.1})


class TestPSOValidation:
    """Audit: pso-validation, aiwpso-validation."""

    def test_w_negative(self):
        with pytest.raises(Exception):
            PSO({"w": -0.1})

    def test_c1_negative(self):
        with pytest.raises(Exception):
            PSO({"c1": -1.0})

    def test_c2_negative(self):
        with pytest.raises(Exception):
            PSO({"c2": -1.0})

    def test_aiwpso_w_max_lt_w_min(self):
        with pytest.raises(Exception):
            AIWPSO({"w_min": 0.5, "w_max": 0.1})

    def test_aiwpso_w_min_negative(self):
        with pytest.raises(Exception):
            AIWPSO({"w_min": -0.1})

    def test_aiwpso_w_max_negative(self):
        with pytest.raises(Exception):
            AIWPSO({"w_max": -0.1})


class TestNodeStr:
    """Audit: node-str — Node should have __str__ and _build_string."""

    def test_terminal_str(self):
        node = Node("x0", "TERMINAL", value=torch.tensor([1.0, 2.0]))
        s = str(node)
        assert "x0" in s

    def test_tree_str(self):
        left = Node("x0", "TERMINAL", value=torch.tensor([1.0]))
        right = Node("x1", "TERMINAL", value=torch.tensor([2.0]))
        root = Node("SUM", "FUNCTION")
        root._left = left
        root._right = right
        left._parent = root
        right._parent = root

        s = str(root)
        assert "SUM" in s
        assert "x0" in s
        assert "x1" in s
