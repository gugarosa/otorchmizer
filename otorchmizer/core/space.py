# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Base search space managing a Population of candidates."""

from __future__ import annotations

import torch

from otorchmizer.core.device import DeviceManager
from otorchmizer.core.population import Population


class Space:
    """Base class for all search spaces.

    Notes:
        Manages batched population storage, initialization, bound enforcement, and device placement.

    """

    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        lower_bound: float | list | tuple | torch.Tensor = 0.0,
        upper_bound: float | list | tuple | torch.Tensor = 1.0,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocate an unbuilt population with broadcastable bounds.

        Args:
            n_agents: Number of candidate solutions.
            n_variables: Number of decision variables.
            n_dimensions: Dimensionality per variable.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: Human-readable variable names.
            device: Device for tensor storage ("auto", "cpu", "cuda:0", etc.).
            dtype: Storage dtype, or None to use the PyTorch default without an intermediate conversion.

        """

        self.device = DeviceManager(device).device

        self.population = Population(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            device=self.device,
            dtype=dtype,
        )
        self.device = self.population.device
        self.built = False

    @property
    def built(self) -> bool:
        """Whether the space has completed population initialization."""

        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        self._built = built

    @property
    def n_agents(self) -> int:
        """Number of candidate solutions in the population."""

        return self.population.n_agents

    @property
    def n_variables(self) -> int:
        """Number of decision variables per candidate."""

        return self.population.n_variables

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions per decision variable."""

        return self.population.n_dimensions

    @property
    def best_position(self) -> torch.Tensor:
        """Tracked best position shaped (n_variables, n_dimensions)."""

        return self.population.best_position

    @property
    def best_fitness(self) -> torch.Tensor:
        """Tracked best fitness as a scalar tensor."""

        return self.population.best_fitness

    def build(self) -> None:
        """Initialize a fresh population, resetting its scores and best archive."""

        self._initialize()
        self.built = True

    def _initialize(self) -> None:
        self.population.initialize_uniform()

    def clip(self) -> None:
        """Clips all agents' positions to bounds."""

        self.population.clip()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_agents={self.n_agents}, n_variables={self.n_variables}, device={self.device})"
        )
