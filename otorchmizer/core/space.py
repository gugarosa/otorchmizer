"""Base search space managing a Population of candidates."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.device import DeviceManager
from otorchmizer.core.population import Population
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class Space:
    """Base class for all search spaces.

    Manages a Population (batched tensor storage) and handles
    initialization, bound enforcement, and device placement.
    """

    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        lower_bound: Union[float, List, Tuple, torch.Tensor] = 0.0,
        upper_bound: Union[float, List, Tuple, torch.Tensor] = 1.0,
        mapping: Optional[List[str]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of candidate solutions.
            n_variables: Number of decision variables.
            n_dimensions: Dimensionality per variable.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: Human-readable variable names.
            device: Device for tensor storage ("auto", "cpu", "cuda:0", etc.).
        """

        self.device = DeviceManager(device).device

        lb = self._to_tensor(lower_bound, n_variables)
        ub = self._to_tensor(upper_bound, n_variables)

        self.population = Population(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            lower_bound=lb,
            upper_bound=ub,
            mapping=mapping,
            device=self.device,
        )

        self.built = False

    @staticmethod
    def _to_tensor(value: Union[float, List, Tuple, torch.Tensor],
                   n_variables: int) -> torch.Tensor:
        """Converts bound values to a tensor."""

        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, (int, float)):
            return torch.full((n_variables,), float(value))
        return torch.tensor(value, dtype=torch.float32)

    @property
    def built(self) -> bool:
        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        self._built = built

    @property
    def n_agents(self) -> int:
        return self.population.n_agents

    @property
    def n_variables(self) -> int:
        return self.population.n_variables

    @property
    def n_dimensions(self) -> int:
        return self.population.n_dimensions

    @property
    def best_position(self) -> torch.Tensor:
        return self.population.best_position

    @property
    def best_fitness(self) -> torch.Tensor:
        return self.population.best_fitness

    def build(self) -> None:
        """Builds the space by initializing the population."""

        self._initialize()
        self.built = True

        logger.debug(
            "Agents: %d | Size: (%d, %d) | Device: %s | Built: %s.",
            self.n_agents,
            self.n_variables,
            self.n_dimensions,
            self.device,
            self.built,
        )

    def _initialize(self) -> None:
        """Initializes agent positions. Override in subclasses."""

        self.population.initialize_uniform()

    def clip(self) -> None:
        """Clips all agents' positions to bounds."""

        self.population.clip()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_agents={self.n_agents}, "
            f"n_variables={self.n_variables}, device={self.device})"
        )
