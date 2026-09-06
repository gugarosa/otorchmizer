# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Deterministic contracts for Genetic Programming optimizers."""

import dill
import pytest
import torch

import otorchmizer.optimizers.evolutionary.gp as gp_module
from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Node, Space, UpdateContext
from otorchmizer.optimizers.evolutionary import GP, GSGP
from otorchmizer.spaces import TreeSpace
from otorchmizer.utils.callback import Callback, DiscreteSearchCallback


def _terminal(
    name: int | str,
    values: list[float],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Node:
    return Node(name, "TERMINAL", torch.tensor(values, device=device, dtype=dtype).unsqueeze(-1))


def _binary(name: str, left: Node, right: Node) -> Node:
    root = Node(name, "FUNCTION")
    root.left = left
    root.right = right
    left.parent = root
    right.parent = root
    right.flag = False
    return root


def _unary(name: str, child: Node) -> Node:
    root = Node(name, "FUNCTION")
    root.left = child
    child.parent = root
    return root


def _assert_tree_ownership(tree: Node) -> None:
    seen = set()
    assert tree.parent is None

    for node in tree.pre_order:
        assert id(node) not in seen
        seen.add(id(node))

        if node.left is not None:
            assert node.left.parent is node
            assert node.left.flag is True
        if node.right is not None:
            assert node.right.parent is node
            assert node.right.flag is False


def _make_space(n_agents: int = 10, device: str = "cpu") -> TreeSpace:
    return TreeSpace(
        n_agents=n_agents,
        n_variables=2,
        lower_bound=[-2.0, -2.0],
        upper_bound=[2.0, 2.0],
        n_terminals=3,
        min_depth=1,
        max_depth=2,
        functions=["SUM", "SUB", "MUL"],
        device=device,
    )


class _TreeStateCallback(Callback):
    def __init__(self) -> None:
        self.updates = 0
        self.evaluations = 0

    def on_update_after(self, ctx: UpdateContext) -> None:
        self.updates += 1
        expected = torch.stack([ctx.space.evaluate_tree(tree) for tree in ctx.space.trees])
        assert torch.allclose(ctx.space.population.positions, expected)

    def on_evaluate_after(self, population, function) -> None:
        self.evaluations += 1
        assert population.fitness.shape == (population.n_agents,)


class _PositionMutationCallback(Callback):
    def __init__(self, route: str) -> None:
        self.route = route

    @staticmethod
    def _change(positions: torch.Tensor) -> None:
        positions.add_(0.125)

    def on_evaluate_before(self, population, function) -> None:
        if self.route == "evaluate_before":
            self._change(population.positions)

    def on_update_before(self, ctx: UpdateContext) -> None:
        if self.route == "update_before":
            self._change(ctx.space.population.positions)

    def on_update_after(self, ctx: UpdateContext) -> None:
        if self.route == "update_after":
            self._change(ctx.space.population.positions)


def test_gp_canonical_parameters_and_validation():
    optimizer = GP()

    assert optimizer.p_reproduction == 0.25
    assert optimizer.p_mutation == 0.1
    assert optimizer.p_crossover == 0.1
    assert optimizer.prunning_ratio == 0.0

    configured = GP(
        {
            "p_reproduction": 0.5,
            "p_mutation": 0.25,
            "p_crossover": 0.5,
            "prunning_ratio": 0.25,
        }
    )
    assert configured.p_reproduction == 0.5
    assert configured.p_mutation == 0.25
    assert configured.p_crossover == 0.5
    assert configured.prunning_ratio == 0.25

    with pytest.raises(ValueError, match="p_mutation"):
        GP({"p_mutation": 1.1})
    with pytest.raises(TypeError, match="p_crossover"):
        GP({"p_crossover": "invalid"})


def test_gsgp_mutation_step_validation():
    assert GSGP().mutation_step == 0.1
    assert GSGP({"mutation_step": 0.25}).mutation_step == 0.25

    with pytest.raises(ValueError, match="mutation_step"):
        GSGP({"mutation_step": -0.1})


def test_gsgp_rejects_nonzero_subtree_pruning():
    assert GSGP({"prunning_ratio": 0}).prunning_ratio == 0

    with pytest.raises(ValueError, match="complete trees"):
        GSGP({"prunning_ratio": 0.1})

    optimizer = GSGP()
    with pytest.raises(ValueError, match="complete trees"):
        optimizer.prunning_ratio = 0.1
    with pytest.raises(ValueError, match="complete trees"):
        optimizer.build({"prunning_ratio": 0.1})


def test_tree_space_initializes_positions_from_trees():
    torch.manual_seed(4)
    space = _make_space()

    expected = torch.stack([space.evaluate_tree(tree) for tree in space.trees])
    assert torch.equal(space.population.positions, expected)
    assert torch.equal(space.population.best_position, expected[0])
    assert all(value.device == space.device for value in space.terminals)
    assert all(tree.min_depth >= space.min_depth for tree in space.trees)
    assert all(tree.max_depth <= space.max_depth for tree in space.trees)


def test_tree_space_rejects_unsupported_functions():
    with pytest.raises(ValueError, match="functions"):
        TreeSpace(
            n_agents=2,
            n_variables=1,
            lower_bound=[0.0],
            upper_bound=[1.0],
            functions=["UNKNOWN"],
        )


def test_protected_division_returns_the_numerator_for_zero_denominator():
    tree = _binary("DIV", _terminal(0, [2.0]), _terminal(1, [0.0]))

    assert torch.equal(tree.position, torch.tensor([[2.0]]))


def test_integer_division_and_log_use_default_floating_dtype():
    division = _binary(
        "DIV",
        Node(0, "TERMINAL", torch.tensor([[6]])),
        Node(1, "TERMINAL", torch.tensor([[2]])),
    )
    logarithm = _unary("LOG", Node(0, "TERMINAL", torch.tensor([[1]])))

    assert division.position.dtype == torch.get_default_dtype()
    assert division.position.item() == pytest.approx(3.0)
    assert logarithm.position.dtype == torch.get_default_dtype()
    assert logarithm.position.item() == pytest.approx(0.0)


def test_integer_primitives_follow_changed_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        division = _binary(
            "DIV",
            Node(0, "TERMINAL", torch.tensor([[6]])),
            Node(1, "TERMINAL", torch.tensor([[2]])),
        )
        logarithm = _unary("LOG", Node(0, "TERMINAL", torch.tensor([[1]])))

        assert division.position.dtype == torch.float64
        assert logarithm.position.dtype == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_mixed_and_float64_primitives_preserve_higher_precision():
    division = _binary(
        "DIV",
        Node(0, "TERMINAL", torch.tensor([[6.0]], dtype=torch.float64)),
        Node(1, "TERMINAL", torch.tensor([[2]], dtype=torch.int64)),
    )
    logarithm = _unary("LOG", Node(0, "TERMINAL", torch.tensor([[1.0]], dtype=torch.float64)))

    assert division.position.dtype == torch.float64
    assert division.position.item() == pytest.approx(3.0)
    assert logarithm.position.dtype == torch.float64
    assert logarithm.position.item() == pytest.approx(0.0)


def test_tree_space_still_rejects_integer_tree_phenotypes():
    space = TreeSpace(
        n_agents=1,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[1.0],
        functions=[],
        device="cpu",
    )
    space.trees = [Node(0, "TERMINAL", torch.tensor([[1]]))]

    with pytest.raises(ValueError, match="tree.position.dtype"):
        space.sync_positions()


def test_nonfinite_expression_is_rejected_instead_of_becoming_midpoint():
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[-100.0],
        upper_bound=[100.0],
        functions=[],
        device="cpu",
    )
    finite_left = _terminal("left", [100.0])
    finite_right = _terminal("right", [100.0])
    invalid = _binary("SUB", _unary("EXP", finite_left), _unary("EXP", finite_right))
    space.trees = [invalid, _terminal("valid", [1.0])]

    assert torch.isfinite(finite_left.position).all()
    assert torch.isfinite(finite_right.position).all()
    assert not torch.isfinite(invalid.position).all()
    with pytest.raises(ValueError, match="finite"):
        space.sync_positions()


def test_gp_subtree_crossover_swaps_independent_branches(monkeypatch):
    points = iter([1, 1])
    monkeypatch.setattr(gp_module.r, "generate_integer_random_number", lambda *args, **kwargs: next(points))

    father = _binary("SUM", _terminal(0, [1.0]), _terminal(1, [2.0]))
    mother = _binary("MUL", _terminal(2, [3.0]), _terminal(3, [4.0]))
    first, second = GP()._cross(father, mother, father.n_nodes, mother.n_nodes)

    assert torch.equal(first.position, torch.tensor([[5.0]]))
    assert torch.equal(second.position, torch.tensor([[4.0]]))
    assert torch.equal(father.position, torch.tensor([[3.0]]))
    assert torch.equal(mother.position, torch.tensor([[12.0]]))
    assert {id(node) for node in first.pre_order}.isdisjoint(id(node) for node in second.pre_order)
    _assert_tree_ownership(first)
    _assert_tree_ownership(second)


def test_gp_subtree_mutation_replaces_selected_branch(monkeypatch):
    space = _make_space(n_agents=2)
    parent = _binary("SUM", _terminal(0, [1.0, 2.0]), _terminal(1, [3.0, 4.0]))
    monkeypatch.setattr(gp_module.r, "generate_integer_random_number", lambda *args, **kwargs: 2)
    monkeypatch.setattr(space, "grow", lambda *args, **kwargs: _terminal(2, [9.0, 8.0]))

    offspring = GP()._mutate(space, parent, parent.n_nodes)

    assert torch.equal(offspring.position, torch.tensor([[10.0], [10.0]]))
    assert torch.equal(parent.position, torch.tensor([[4.0], [6.0]]))
    assert not {id(node) for node in offspring.pre_order}.intersection(id(node) for node in parent.pre_order)
    _assert_tree_ownership(offspring)


def test_gp_generation_keeps_tree_position_and_fitness_state_coherent(monkeypatch):
    space = TreeSpace(
        n_agents=6,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[5.0],
        functions=[],
        device="cpu",
    )
    space.trees = [_terminal(index, [float(index)]) for index in range(6)]
    space.sync_positions()

    monkeypatch.setattr(
        gp_module.g,
        "tournament_selection",
        lambda fitness, n_individuals: torch.zeros(n_individuals, dtype=torch.long, device=fitness.device),
    )
    optimizer = GP({"p_reproduction": 0.5, "p_crossover": 0.0, "p_mutation": 0.0})
    optimizer.bind(space)
    space.population.fitness = space.population.positions.flatten().square()
    optimizer.update(UpdateContext(space, Function(lambda x: x.square().sum()), 0, 1, space.device))

    assert torch.equal(space.population.positions[3:], torch.zeros(3, 1, 1))
    assert torch.isinf(space.population.fitness).all()
    assert torch.equal(space.population.positions, torch.stack([space.evaluate_tree(tree) for tree in space.trees]))
    assert len({id(tree) for tree in space.trees}) == space.n_agents


def test_gp_singleton_mutation_runs_and_preserves_archived_best(monkeypatch):
    space = TreeSpace(
        n_agents=1,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[1.0],
        functions=[],
        device="cpu",
    )
    space.trees = [_terminal(0, [0.2])]
    space.sync_positions()
    optimizer = GP({"p_reproduction": 0.0, "p_crossover": 0.0, "p_mutation": 1.0})
    optimizer.bind(space)
    function = Function(lambda x: x.square().sum())
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(gp_module.r, "generate_integer_random_number", lambda *args, **kwargs: 0)
    monkeypatch.setattr(space, "grow", lambda *args, **kwargs: _terminal(1, [0.8]))
    optimizer.update(UpdateContext(space, function, 0, 1, space.device))

    assert space.population.positions.item() == pytest.approx(0.8)
    optimizer.evaluate(space.population, function)
    assert space.best_position.item() == pytest.approx(0.2)
    assert space.best_fitness.item() == pytest.approx(0.04)
    assert space.evaluate_tree(space.best_tree).item() == pytest.approx(0.2)


def test_gsgp_singleton_mutation_runs_and_preserves_archived_best(monkeypatch):
    space = TreeSpace(
        n_agents=1,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[1.0],
        functions=[],
        device="cpu",
    )
    space.trees = [_terminal(0, [0.2])]
    space.sync_positions()
    optimizer = GSGP({"p_reproduction": 0.0, "p_crossover": 0.0, "p_mutation": 1.0})
    optimizer.bind(space)
    function = Function(lambda x: x.square().sum())
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(
        optimizer,
        "_random_terminal",
        lambda reference, name: Node(
            name,
            "TERMINAL",
            torch.ones_like(reference) if name == "R1" else torch.zeros_like(reference),
        ),
    )
    optimizer.update(UpdateContext(space, function, 0, 1, space.device))

    assert space.population.positions.item() == pytest.approx(0.3)
    optimizer.evaluate(space.population, function)
    assert space.best_position.item() == pytest.approx(0.2)
    assert space.best_fitness.item() == pytest.approx(0.04)
    assert space.evaluate_tree(space.best_tree).item() == pytest.approx(0.2)


def test_gp_two_agent_crossover_runs_and_preserves_archived_best(monkeypatch):
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[1.0],
        functions=[],
        device="cpu",
    )
    space.trees = [
        _binary("SUM", _terminal(0, [0.1]), _terminal(1, [0.2])),
        _binary("SUB", _terminal(2, [0.9]), _terminal(3, [0.2])),
    ]
    space.sync_positions()
    optimizer = GP({"p_reproduction": 0.0, "p_crossover": 1.0, "p_mutation": 0.0})
    optimizer.bind(space)
    function = Function(lambda x: ((x - 0.5) ** 2).sum())
    optimizer.evaluate(space.population, function)

    monkeypatch.setattr(optimizer, "_select_pairs", lambda fitness, n_pairs: ([0], [1]))
    points = iter([1, 1])
    monkeypatch.setattr(gp_module.r, "generate_integer_random_number", lambda *args, **kwargs: next(points))
    optimizer.update(UpdateContext(space, function, 0, 1, space.device))

    assert torch.equal(space.population.positions.flatten().sort().values, torch.tensor([0.0, 1.0]))
    optimizer.evaluate(space.population, function)
    assert space.best_position.item() == pytest.approx(0.3)
    assert space.best_fitness.item() == pytest.approx(0.04)
    assert space.evaluate_tree(space.best_tree).item() == pytest.approx(0.3)


def test_gsgp_two_agent_crossover_runs_and_preserves_archived_best():
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[1.0],
        functions=[],
        device="cpu",
    )
    space.trees = [_terminal(0, [0.2]), _terminal(1, [0.8])]
    space.sync_positions()
    optimizer = GSGP({"p_reproduction": 0.0, "p_crossover": 1.0, "p_mutation": 0.0})
    optimizer.bind(space)
    function = Function(lambda x: x.square().sum())
    optimizer.evaluate(space.population, function)

    torch.manual_seed(9)
    optimizer.update(UpdateContext(space, function, 0, 1, space.device))

    assert torch.all(space.population.positions > 0.2)
    assert torch.all(space.population.positions < 0.8)
    optimizer.evaluate(space.population, function)
    assert space.best_position.item() == pytest.approx(0.2)
    assert space.best_fitness.item() == pytest.approx(0.04)
    assert space.evaluate_tree(space.best_tree).item() == pytest.approx(0.2)


def test_gsgp_mutation_matches_semantic_equation():
    space = _make_space(n_agents=2)
    parent = _terminal(0, [0.25, -0.5])
    optimizer = GSGP({"mutation_step": 0.2})

    torch.manual_seed(12)
    random_one = torch.rand_like(parent.position)
    random_two = torch.rand_like(parent.position)
    expected = parent.position + 0.2 * (random_one - random_two)

    torch.manual_seed(12)
    offspring = optimizer._mutate(space, parent, parent.n_nodes)

    assert torch.allclose(offspring.position, expected)
    assert offspring.n_nodes == 7
    assert not {id(node) for node in offspring.pre_order}.intersection(id(node) for node in parent.pre_order)
    _assert_tree_ownership(offspring)


def test_gsgp_crossover_matches_convex_semantic_equation():
    father = _terminal(0, [0.2, 0.8])
    mother = _terminal(1, [0.6, -0.4])
    optimizer = GSGP()

    torch.manual_seed(21)
    mask = torch.rand_like(father.position)
    expected_first = mask * father.position + (1 - mask) * mother.position
    expected_second = mask * mother.position + (1 - mask) * father.position

    torch.manual_seed(21)
    first, second = optimizer._cross(father, mother, father.n_nodes, mother.n_nodes)

    assert torch.allclose(first.position, expected_first)
    assert torch.allclose(second.position, expected_second)
    assert {id(node) for node in first.pre_order}.isdisjoint(id(node) for node in second.pre_order)
    _assert_tree_ownership(first)
    _assert_tree_ownership(second)


@pytest.mark.parametrize("optimizer", [GP(), GSGP()])
def test_tree_optimizers_run_through_engine_with_batching_history_and_callbacks(optimizer):
    torch.manual_seed(31)
    space = _make_space(n_agents=12)
    calls = []

    def batch_sphere(positions):
        calls.append(tuple(positions.shape))
        return positions.square().sum(dim=(1, 2))

    callback = _TreeStateCallback()
    model = Otorchmizer(space, optimizer, Function(batch_sphere, batch=True), save_agents=True)
    model.start(n_iterations=3, callbacks=[callback])

    assert calls == [(12, 2, 1)] * 4
    assert callback.updates == 3
    assert callback.evaluations == 4
    assert len(model.history.best_agent) == 3
    assert len(model.history.positions) == 3
    assert torch.equal(space.population.positions, torch.stack([space.evaluate_tree(tree) for tree in space.trees]))
    assert torch.equal(space.best_position, space.evaluate_tree(space.best_tree))
    assert torch.equal(space.population.fitness, batch_sphere(space.population.positions))


@pytest.mark.parametrize("route", ["evaluate_before", "update_before", "update_after"])
def test_tree_optimizer_rejects_position_changing_callbacks(route):
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[1.0],
        functions=[],
        device="cpu",
    )
    space.trees = [_terminal(0, [0.2]), _terminal(1, [0.8])]
    space.sync_positions()
    model = Otorchmizer(
        space,
        GP({"p_reproduction": 0.0, "p_crossover": 0.0, "p_mutation": 0.0}),
        Function(lambda x: x.square().sum()),
    )
    callback = DiscreteSearchCallback([[0.0, 1.0]]) if route == "evaluate_before" else _PositionMutationCallback(route)

    with pytest.raises(ValueError, match="observational callbacks"):
        model.start(n_iterations=0 if route == "evaluate_before" else 1, callbacks=[callback])

    space.validate_positions()


def test_gp_requires_explicit_tree_space_binding():
    population_space = Space(n_agents=2, n_variables=1, lower_bound=[0.0], upper_bound=[1.0])
    population_space.build()
    optimizer = GP()
    function = Function(lambda x: x.square().sum())

    with pytest.raises(RuntimeError, match="bound"):
        optimizer.evaluate(population_space.population, function)
    with pytest.raises(TypeError, match="TreeSpace"):
        Otorchmizer(population_space, optimizer, function)


def test_population_only_dtype_transfer_is_rejected():
    space = _make_space(n_agents=2)
    space.population.to(space.device, dtype=torch.float64)

    with pytest.raises(ValueError, match="tree.position.dtype"):
        Otorchmizer(space, GP(), Function(lambda x: x.square().sum()))


def test_bound_tree_state_survives_serialization():
    torch.manual_seed(42)
    model = Otorchmizer(
        _make_space(n_agents=6),
        GP({"p_reproduction": 0.5}),
        Function(lambda x: x.square().sum()),
    )
    model.start(n_iterations=1)

    restored = dill.loads(dill.dumps(model))

    assert restored.optimizer._space is restored.space
    assert torch.equal(restored.space.best_position, restored.space.evaluate_tree(restored.space.best_tree))


def test_tree_space_to_moves_all_state_and_revalidates_historical_best():
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[2.0],
        functions=[],
        device="cpu",
    )
    semantic_optimizer = GSGP(
        {
            "p_reproduction": 0.0,
            "p_crossover": 0.0,
            "p_mutation": 0.0,
        }
    )
    space.trees = [
        semantic_optimizer._mutate(space, _terminal(0, [1.0]), 1),
        _terminal(1, [2.0]),
    ]
    space.sync_positions()
    space.best_tree = _terminal("archive", [0.25])
    space.population.best_position = space.evaluate_tree(space.best_tree)
    space.population.best_fitness = torch.tensor(0.0625)

    semantic_optimizer.bind(space)
    semantic_optimizer._compiled_update = lambda ctx: None
    archived_tree = space.best_tree
    result = space.to("cpu", dtype=torch.float64)

    assert result is space
    assert space.best_tree is archived_tree
    assert space.device == torch.device("cpu")
    assert space.population.device == torch.device("cpu")
    assert space.population.dtype == torch.float64
    assert torch.isinf(space.population.fitness).all()
    assert torch.isinf(space.best_fitness)
    assert space.best_position.item() == pytest.approx(0.25)
    assert all(terminal.dtype == torch.float64 for terminal in space.terminals)
    assert all(
        node.value.dtype == torch.float64
        for tree in [*space.trees, space.best_tree]
        for node in tree.pre_order
        if node.value is not None
    )

    semantic_optimizer.rebind(space)
    assert semantic_optimizer._compiled_update is None
    calls = []

    def batch_sphere(positions):
        calls.append(tuple(positions.shape))
        return positions.square().sum(dim=(1, 2))

    function = Function(batch_sphere, batch=True)
    semantic_optimizer.evaluate(space.population, function)

    assert calls == [(3, 1, 1)]
    assert space.best_tree is archived_tree
    assert space.best_position.item() == pytest.approx(0.25)
    assert space.best_fitness.item() == pytest.approx(0.0625)
    assert space.best_position.square().sum().item() == pytest.approx(space.best_fitness.item())


@pytest.mark.parametrize("optimizer_cls", [GP, GSGP])
def test_repeated_transfers_preserve_pending_historical_best(optimizer_cls):
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[2.0],
        functions=[],
        device="cpu",
    )
    space.trees = [_terminal(0, [1.0]), _terminal(1, [2.0])]
    space.sync_positions()
    space.best_tree = _terminal("archive", [0.25])
    space.population.best_position = space.evaluate_tree(space.best_tree)
    space.population.best_fitness = torch.tensor(0.0625)

    space.to("cpu", dtype=torch.float64)
    assert space._best_tree_needs_evaluation
    space.to("cpu", dtype=torch.float32)
    assert space._best_tree_needs_evaluation

    optimizer = optimizer_cls(
        {
            "p_reproduction": 0.0,
            "p_crossover": 0.0,
            "p_mutation": 0.0,
        }
    )
    optimizer.rebind(space)
    optimizer.evaluate(space.population, Function(lambda x: x.square().sum()))

    assert not space._best_tree_needs_evaluation
    assert space.best_position.item() == pytest.approx(0.25)
    assert space.best_fitness.item() == pytest.approx(0.0625)
    assert space.evaluate_tree(space.best_tree).item() == pytest.approx(0.25)


def test_large_historical_score_does_not_lose_archive_during_downcast():
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[2.0],
        functions=[],
        device="cpu",
    ).to("cpu", dtype=torch.float64)
    space.trees = [
        _terminal(0, [1.0], dtype=torch.float64),
        _terminal(1, [2.0], dtype=torch.float64),
    ]
    space.sync_positions()
    archived_tree = _terminal("archive", [0.25], dtype=torch.float64)
    space.best_tree = archived_tree
    space.population.best_position = space.evaluate_tree(archived_tree)
    space.population.best_fitness = torch.tensor(1e100, dtype=torch.float64)

    space.to("cpu", dtype=torch.float32)

    assert space._best_tree_needs_evaluation
    assert space.best_tree is archived_tree
    assert space.best_position.item() == pytest.approx(0.25)

    optimizer = GP({"p_reproduction": 0.0, "p_crossover": 0.0, "p_mutation": 0.0})
    optimizer.rebind(space)
    optimizer.evaluate(space.population, Function(lambda x: x.square().sum()))

    assert space.best_tree is archived_tree
    assert space.best_fitness.item() == pytest.approx(0.0625)


def test_failed_archive_evaluation_remains_pending_for_retry():
    space = TreeSpace(
        n_agents=2,
        n_variables=1,
        lower_bound=[0.0],
        upper_bound=[2.0],
        functions=[],
        device="cpu",
    )
    space.trees = [_terminal(0, [1.0]), _terminal(1, [2.0])]
    space.sync_positions()
    space.best_tree = _terminal("archive", [0.25])
    space.population.best_position = space.evaluate_tree(space.best_tree)
    space.population.best_fitness = torch.tensor(0.0625)
    space.to("cpu", dtype=torch.float64)

    optimizer = GP({"p_reproduction": 0.0, "p_crossover": 0.0, "p_mutation": 0.0})
    optimizer.rebind(space)

    def fail(positions):
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        optimizer.evaluate(space.population, Function(fail, batch=True))
    assert space._best_tree_needs_evaluation

    optimizer.evaluate(space.population, Function(lambda x: x.square().sum()))
    assert not space._best_tree_needs_evaluation
    assert space.best_position.item() == pytest.approx(0.25)
    assert space.best_fitness.item() == pytest.approx(0.0625)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("optimizer_cls", [GP, GSGP])
def test_tree_terminals_and_semantic_operators_stay_on_cuda(optimizer_cls):
    torch.manual_seed(7)
    space = _make_space(n_agents=6).to("cuda", dtype=torch.float64)
    optimizer = optimizer_cls({"p_crossover": 0.5, "p_mutation": 0.5})
    model = Otorchmizer(space, optimizer, Function(lambda x: x.square().sum()))
    model.start(n_iterations=1)

    assert space.population.positions.is_cuda
    assert space.population.dtype == torch.float64
    assert all(terminal.is_cuda for terminal in space.terminals)
    assert all(
        node.value.is_cuda and node.value.dtype == torch.float64
        for tree in [*space.trees, space.best_tree]
        for node in tree.pre_order
        if node.value is not None
    )
