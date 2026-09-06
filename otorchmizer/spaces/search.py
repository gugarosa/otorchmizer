# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Standard continuous search space."""

from __future__ import annotations

import torch

from otorchmizer.core.space import Space


class SearchSpace(Space):
    """Standard search space for continuous optimization."""

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        lower_bound: float | list[float] | tuple[float, ...] | torch.Tensor,
        upper_bound: float | list[float] | tuple[float, ...] | torch.Tensor,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a continuous search space.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            lower_bound: Lower bound for each decision variable.
            upper_bound: Upper bound for each decision variable.
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.
            dtype: Storage dtype, or None to use the PyTorch default.

        Notes:
            Agent positions are initialized uniformly within the supplied bounds.

        """

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            device=device,
            dtype=dtype,
        )

        self.build()

    def _initialize(self) -> None:
        self.population.initialize_uniform()
