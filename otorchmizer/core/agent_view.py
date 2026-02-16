"""AgentView — backward-compatible accessor into a Population row.

Does NOT own data — it references the Population tensors.
Use for debugging, inspection, and migration; never in hot loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from otorchmizer.core.population import Population


class AgentView:
    """Lightweight, non-owning view into a single agent within a Population.

    Provides attribute-style access to an agent's position and fitness
    without extracting data from the batched tensor storage.
    """

    def __init__(self, population: Population, index: int) -> None:
        """Initialization method.

        Args:
            population: The population this agent belongs to.
            index: The row index of this agent in the population.
        """

        self._pop = population
        self._idx = index

    @property
    def position(self) -> torch.Tensor:
        """Agent's position tensor, shape (n_variables, n_dimensions)."""

        return self._pop.positions[self._idx]

    @position.setter
    def position(self, value: torch.Tensor) -> None:
        self._pop.positions[self._idx] = value

    @property
    def fit(self) -> float:
        """Agent's fitness as a Python float."""

        return self._pop.fitness[self._idx].item()

    @fit.setter
    def fit(self, value: float) -> None:
        self._pop.fitness[self._idx] = value

    @property
    def lb(self) -> torch.Tensor:
        """Lower bounds from the population."""

        return self._pop.lb

    @property
    def ub(self) -> torch.Tensor:
        """Upper bounds from the population."""

        return self._pop.ub

    @property
    def mapping(self) -> list[str]:
        """Variable names from the population."""

        return self._pop.mapping

    @property
    def n_variables(self) -> int:
        return self._pop.n_variables

    @property
    def n_dimensions(self) -> int:
        return self._pop.n_dimensions

    def clip_by_bound(self) -> None:
        """Clips this agent's position to bounds."""

        self._pop.positions[self._idx] = self._pop.positions[self._idx].clamp(
            min=self._pop.lb, max=self._pop.ub
        )

    def __repr__(self) -> str:
        return f"AgentView(index={self._idx}, fit={self.fit:.6f})"
