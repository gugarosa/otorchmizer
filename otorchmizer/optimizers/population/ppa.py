# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Parasitism-Predation Algorithm.

References:
    A. S. Mohamed, A. A. Hadi, and A. W. Mohamed.
    Parasitism-Predation algorithm (PPA): A novel approach for feature selection.
    Ain Shams Engineering Journal (2020).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.distribution as d
import otorchmizer.math.general as g
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class PPA(Optimizer):
    """Apply crow nesting, cuckoo parasitism, and cat predation phases."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> PPA.")

        super().__init__(params)

        logger.info("Class overrided.")

    def compile(self, population) -> None:
        """Initialize cat velocities and validate population cardinality.

        Args:
            population: Population that defines state shape, device, and dtype.

        Raises:
            ValueError: The population contains fewer than two agents.

        """

        if population.n_agents < 2:
            raise e.ValueError("`population.n_agents` must be at least 2.")

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device, dtype=population.dtype)

    @staticmethod
    def _calculate_population(n_agents: int, iteration: int, n_iterations: int) -> tuple[int, int, int]:
        n_crows = round(n_agents * (2 / 3 - iteration * ((2 / 3 - 1 / 2) / n_iterations)))
        n_cats = round(n_agents * (0.01 + iteration * ((1 / 3 - 0.01) / n_iterations)))
        n_cuckoos = n_agents - n_crows - n_cats
        return n_crows, n_cats, n_cuckoos

    @staticmethod
    def _sample_other(indices: torch.Tensor, n_agents: int) -> torch.Tensor:
        sampled = torch.randint(
            0,
            n_agents - 1,
            indices.shape,
            device=indices.device,
        )
        return sampled + (sampled >= indices)

    def _nesting_phase(self, pop, n_crows: int) -> None:
        crow_idx = torch.arange(n_crows, device=pop.device)
        other = self._sample_other(crow_idx, pop.n_agents)
        levy = d.generate_levy_distribution(
            size=(n_crows, pop.n_variables, 1),
            device=pop.device,
            dtype=pop.dtype,
        )
        positions = pop.positions.clone()
        pop.positions[:n_crows] = 0.01 * levy * (positions[other] - positions[crow_idx])
        pop.positions[:n_crows] = pop.positions[:n_crows].clamp(
            min=pop.lb,
            max=pop.ub,
        )

    def _parasitism_phase(self, pop, n_crows: int, n_cuckoos: int, progress: float) -> None:
        if n_cuckoos == 0:
            return

        cuckoo_idx = torch.arange(
            n_crows,
            n_crows + n_cuckoos,
            device=pop.device,
        )
        winners = g.tournament_selection(pop.fitness[cuckoo_idx], n_cuckoos)
        winning_cuckoos = pop.positions[cuckoo_idx[winners]].clone()
        first = torch.randint(
            0,
            pop.n_agents,
            (n_cuckoos,),
            device=pop.device,
        )
        second = self._sample_other(first, pop.n_agents)
        scale = torch.rand(
            n_cuckoos,
            1,
            1,
            device=pop.device,
            dtype=pop.dtype,
        )
        positions = pop.positions.clone()
        gaussian_step = (positions[first] - positions[second]) * scale
        preserve = (
            torch.rand(
                n_cuckoos,
                pop.n_variables,
                1,
                device=pop.device,
                dtype=pop.dtype,
            )
            < 1 - progress
        ).to(pop.dtype)
        pop.positions[cuckoo_idx] = winning_cuckoos + gaussian_step * preserve
        pop.positions[cuckoo_idx] = pop.positions[cuckoo_idx].clamp(
            min=pop.lb,
            max=pop.ub,
        )

    def _predation_phase(self, pop, n_crows: int, n_cuckoos: int, progress: float) -> None:
        cat_start = n_crows + n_cuckoos
        if cat_start == pop.n_agents:
            return

        cat_idx = torch.arange(cat_start, pop.n_agents, device=pop.device)
        scale = torch.rand(
            cat_idx.shape[0],
            1,
            1,
            device=pop.device,
            dtype=pop.dtype,
        )
        self.velocity[cat_idx] += scale * (2 - progress) * (pop.best_position.unsqueeze(0) - pop.positions[cat_idx])
        pop.positions[cat_idx] = (pop.positions[cat_idx] + self.velocity[cat_idx]).clamp(
            min=pop.lb.unsqueeze(0),
            max=pop.ub.unsqueeze(0),
        )

    def update(self, ctx: UpdateContext) -> None:
        """Partition the population and run the three search phases.

        Args:
            ctx: Current optimization state.

        """

        pop = ctx.space.population
        n_iterations = max(ctx.n_iterations, 1)
        progress = ctx.iteration / n_iterations
        n_crows, _, n_cuckoos = self._calculate_population(
            pop.n_agents,
            ctx.iteration,
            n_iterations,
        )

        self._nesting_phase(pop, n_crows)
        self._parasitism_phase(pop, n_crows, n_cuckoos, progress)
        self._predation_phase(pop, n_crows, n_cuckoos, progress)
