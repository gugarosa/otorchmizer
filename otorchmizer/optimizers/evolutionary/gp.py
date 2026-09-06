# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Tree-based Genetic Programming.

References:
    J. R. Koza. Genetic Programming: On the Programming of Computers by Means of Natural Selection.
    MIT Press (1992).

"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch

import otorchmizer.math.general as g
import otorchmizer.math.random as r
from otorchmizer.core.function import Function
from otorchmizer.core.node import Node
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.core.space import Space
from otorchmizer.spaces.tree import TreeSpace


def _replace_subtree(tree: Node, position: int, replacement: Node) -> Node:
    replacement.parent = None
    replacement.flag = True
    if position == 0:
        return replacement

    target = tree.pre_order[position]
    parent = target.parent
    if parent is None:
        raise ValueError("`position` should identify a linked subtree.")

    if parent.left is target:
        parent.left = replacement
        replacement.flag = True
    elif parent.right is target:
        parent.right = replacement
        replacement.flag = False
    else:
        raise ValueError("`target.parent` should reference the target node.")

    replacement.parent = parent
    target.parent = None
    return tree


class GP(Optimizer):
    """Evolve expression trees with tournament selection and subtree operators.

    Notes:
        Population positions are derived exclusively from TreeSpace trees. Lifecycle callbacks may inspect this
        state but must not mutate positions independently from the corresponding expressions.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        Notes:
            The canonical defaults are 0.25 reproduction, 0.1 mutation, 0.1 crossover, and no pruning.

        """

        self.p_reproduction = 0.25
        self.p_mutation = 0.1
        self.p_crossover = 0.1
        self.prunning_ratio = 0.0
        self._space: TreeSpace | None = None

        super().__init__(params)

    @property
    def p_reproduction(self) -> float:
        """Return the population fraction reproduced unchanged."""

        return self._p_reproduction

    @p_reproduction.setter
    def p_reproduction(self, p_reproduction: float) -> None:
        if isinstance(p_reproduction, bool) or not isinstance(p_reproduction, Real):
            raise TypeError("`p_reproduction` should be a float or integer.")
        if not 0 <= p_reproduction <= 1:
            raise ValueError("`p_reproduction` should be between 0 and 1.")
        self._p_reproduction = float(p_reproduction)

    @property
    def p_mutation(self) -> float:
        """Return the population fraction created through mutation."""

        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if isinstance(p_mutation, bool) or not isinstance(p_mutation, Real):
            raise TypeError("`p_mutation` should be a float or integer.")
        if not 0 <= p_mutation <= 1:
            raise ValueError("`p_mutation` should be between 0 and 1.")
        self._p_mutation = float(p_mutation)

    @property
    def p_crossover(self) -> float:
        """Return the population fraction created through crossover."""

        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, p_crossover: float) -> None:
        if isinstance(p_crossover, bool) or not isinstance(p_crossover, Real):
            raise TypeError("`p_crossover` should be a float or integer.")
        if not 0 <= p_crossover <= 1:
            raise ValueError("`p_crossover` should be between 0 and 1.")
        self._p_crossover = float(p_crossover)

    @property
    def prunning_ratio(self) -> float:
        """Return the fraction of trailing operator points excluded from sampling."""

        return self._prunning_ratio

    @prunning_ratio.setter
    def prunning_ratio(self, prunning_ratio: float) -> None:
        if isinstance(prunning_ratio, bool) or not isinstance(prunning_ratio, Real):
            raise TypeError("`prunning_ratio` should be a float or integer.")
        if not 0 <= prunning_ratio <= 1:
            raise ValueError("`prunning_ratio` should be between 0 and 1.")
        self._prunning_ratio = float(prunning_ratio)

    def bind(self, space: Space) -> None:
        """Bind the expression-tree state used by tensor-only evaluation.

        Args:
            space: TreeSpace whose trees correspond one-to-one with population rows.

        Raises:
            TypeError: The supplied space is not a TreeSpace.

        Notes:
            Otorchmizer invokes this lifecycle hook before compile(). It keeps the established
            evaluate(population, function) contract while making the structural genotype explicit.

        """

        if not isinstance(space, TreeSpace):
            raise TypeError("`space` should be a TreeSpace.")

        super().bind(space)
        self._space = space
        space.sync_positions()

    def validate_space(self, space: Space) -> None:
        """Validate tree ownership and genotype-to-phenotype synchronization.

        Args:
            space: TreeSpace expected to be bound to this optimizer.

        Raises:
            RuntimeError: The optimizer has not been bound.
            ValueError: A different space is supplied or its positions were externally changed.

        """

        if self._space is None:
            raise RuntimeError("`optimizer` should be bound to a TreeSpace before validation.")
        if space is not self._space:
            raise ValueError("`space` should be the TreeSpace bound to the optimizer.")

        self._space.validate_positions()

    def _prune_nodes(self, n_nodes: int) -> int:
        """Limit the prefix of preorder nodes eligible for a genetic operator.

        Args:
            n_nodes: Number of nodes in the tree.

        Returns:
            Maximum number of eligible preorder nodes.

        """

        pruned_nodes = int(n_nodes * (1 - self.prunning_ratio))
        return max(2, pruned_nodes)

    @staticmethod
    def _select(fitness: torch.Tensor, n_individuals: int) -> list[int]:
        if n_individuals <= 0:
            return []
        return g.tournament_selection(fitness, n_individuals).tolist()

    @staticmethod
    def _select_pairs(fitness: torch.Tensor, n_pairs: int) -> tuple[list[int], list[int]]:
        if n_pairs <= 0:
            return [], []

        n_agents = fitness.shape[0]
        if n_agents < 2:
            raise ValueError("`fitness` should contain at least two individuals for crossover.")

        fathers = g.tournament_selection(fitness, n_pairs)
        candidates = torch.randint(0, n_agents - 1, (n_pairs, 2), device=fitness.device)
        candidates += candidates >= fathers.unsqueeze(1)
        candidate_fitness = fitness[candidates]
        mothers = candidates[torch.arange(n_pairs, device=fitness.device), candidate_fitness.argmin(dim=1)]
        return fathers.tolist(), mothers.tolist()

    def _mutate(self, space: TreeSpace, tree: Node, max_nodes: int) -> Node:
        """Replace one randomly selected subtree with a newly grown branch.

        Args:
            space: TreeSpace used to grow the replacement branch.
            tree: Parent expression tree.
            max_nodes: Maximum preorder prefix eligible for replacement.

        Returns:
            Independent mutated tree.

        """

        offspring = tree.clone()
        limit = min(offspring.n_nodes, max_nodes)
        mutation_point = r.generate_integer_random_number(0, limit)
        branch = space.grow(space.min_depth, space.max_depth)
        return _replace_subtree(offspring, mutation_point, branch)

    def _cross(
        self,
        father: Node,
        mother: Node,
        max_father: int,
        max_mother: int,
    ) -> tuple[Node, Node]:
        """Exchange independently owned subtrees between two parents.

        Args:
            father: First parent tree.
            mother: Second parent tree.
            max_father: Maximum preorder prefix eligible in the first parent.
            max_mother: Maximum preorder prefix eligible in the second parent.

        Returns:
            Two independent offspring trees.

        """

        father_point = r.generate_integer_random_number(0, min(father.n_nodes, max_father))
        mother_point = r.generate_integer_random_number(0, min(mother.n_nodes, max_mother))

        father_branch = father.pre_order[father_point].clone()
        mother_branch = mother.pre_order[mother_point].clone()

        father_offspring = _replace_subtree(father.clone(), father_point, mother_branch)
        mother_offspring = _replace_subtree(mother.clone(), mother_point, father_branch)
        return father_offspring, mother_offspring

    def _evolve(
        self,
        space: TreeSpace,
        p_reproduction: float,
        p_crossover: float,
        p_mutation: float,
    ) -> None:
        population = space.population
        n_agents = population.n_agents
        parent_trees = space.trees
        offspring = [tree.clone() for tree in parent_trees]
        targets = population.fitness.argsort(descending=True).tolist()
        cursor = 0

        n_reproduction = min(int(n_agents * p_reproduction), len(targets) - cursor)
        selected = self._select(population.fitness, n_reproduction)
        for target, source in zip(targets[cursor : cursor + n_reproduction], selected):
            offspring[target] = parent_trees[source].clone()
        cursor += n_reproduction

        requested_crossover = int(n_agents * p_crossover)
        if requested_crossover > 0 and n_agents < 2:
            raise ValueError("`space.n_agents` should be at least 2 when crossover selects offspring.")
        if requested_crossover % 2:
            requested_crossover += 1
        n_crossover = min(requested_crossover, len(targets) - cursor)
        n_crossover -= n_crossover % 2
        fathers, mothers = self._select_pairs(population.fitness, n_crossover // 2)
        for pair, (father, mother) in enumerate(zip(fathers, mothers)):
            children = self._cross(
                parent_trees[father],
                parent_trees[mother],
                self._prune_nodes(parent_trees[father].n_nodes),
                self._prune_nodes(parent_trees[mother].n_nodes),
            )
            first_target = targets[cursor + pair * 2]
            second_target = targets[cursor + pair * 2 + 1]
            offspring[first_target], offspring[second_target] = children
        cursor += n_crossover

        n_mutation = min(int(n_agents * p_mutation), len(targets) - cursor)
        selected = self._select(population.fitness, n_mutation)
        for target, source in zip(targets[cursor : cursor + n_mutation], selected):
            parent = parent_trees[source]
            offspring[target] = self._mutate(space, parent, self._prune_nodes(parent.n_nodes))
        cursor += n_mutation

        if cursor == 0:
            return

        space.trees = offspring
        space.sync_positions()
        population.fitness.fill_(torch.inf)

    def _reproduction(self, space: TreeSpace) -> None:
        """Reproduce tournament winners into the worst population slots.

        Args:
            space: Bound tree search space.

        """

        self._evolve(space, self.p_reproduction, 0.0, 0.0)

    def _crossover(self, space: TreeSpace) -> None:
        """Create subtree-crossover offspring in the worst population slots.

        Args:
            space: Bound tree search space.

        """

        self._evolve(space, 0.0, self.p_crossover, 0.0)

    def _mutation(self, space: TreeSpace) -> None:
        """Create subtree-mutation offspring in the worst population slots.

        Args:
            space: Bound tree search space.

        """

        self._evolve(space, 0.0, 0.0, self.p_mutation)

    def evaluate(self, population: Population, function: Function) -> None:
        """Batch-evaluate tree phenotypes and update tensor and tree best state.

        Args:
            population: Population bound to the optimizer's TreeSpace.
            function: Objective function.

        Raises:
            RuntimeError: The optimizer has not been bound to a TreeSpace.
            ValueError: The supplied population is not owned by the bound TreeSpace.

        """

        if self._space is None:
            raise RuntimeError("`optimizer` should be bound to a TreeSpace before evaluation.")
        if population is not self._space.population:
            raise ValueError("`population` should belong to the bound TreeSpace.")

        self.validate_space(self._space)

        if self._space._best_tree_needs_evaluation:
            archived_position = self._space.evaluate_tree(self._space.best_tree)
            fitness = function(torch.cat((population.positions, archived_position.unsqueeze(0))))
            population.fitness = fitness[:-1]
            archived_fitness = fitness[-1]
            if torch.isfinite(archived_fitness):
                population.best_position = archived_position.clone()
                population.best_fitness = archived_fitness.clone()
            else:
                population.best_fitness.fill_(torch.inf)
            self._space._best_tree_needs_evaluation = False
        else:
            population.fitness = function(population.positions)

        best_index = population.fitness.argmin()
        if population.fitness[best_index] < population.best_fitness:
            population.best_fitness = population.fitness[best_index].clone()
            population.best_position = population.positions[best_index].clone()
            self._space.best_tree = self._space.trees[best_index.item()].clone()

    def update(self, ctx: UpdateContext) -> None:
        """Create the next generation with reproduction, crossover, and mutation.

        Args:
            ctx: Current optimization state and objective.

        Raises:
            RuntimeError: The optimizer has not been bound to a TreeSpace.
            ValueError: The update context contains a different space.

        """

        if self._space is None:
            raise RuntimeError("`optimizer` should be bound to a TreeSpace before updating.")
        if ctx.space is not self._space:
            raise ValueError("`ctx.space` should be the TreeSpace bound to the optimizer.")

        self.validate_space(self._space)
        self._evolve(
            self._space,
            self.p_reproduction,
            self.p_crossover,
            self.p_mutation,
        )
