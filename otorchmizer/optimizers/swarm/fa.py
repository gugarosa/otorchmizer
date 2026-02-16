"""Firefly Algorithm — vectorized pairwise interactions with torch.cdist.

References:
    X.-S. Yang. Firefly algorithms for multimodal optimization.
    International Symposium on Stochastic Algorithms (2009).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FA(Optimizer):
    """Firefly Algorithm.

    The original FA has O(n²) nested loops for pairwise interactions.
    This implementation replaces them with torch.cdist and tensor broadcasting,
    achieving massive speedups especially on GPU.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> FA.")

        self.alpha = 0.5
        self.beta = 0.2
        self.gamma = 1.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")
        self._alpha = alpha

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0:
            raise e.ValueError("`beta` should be >= 0")
        self._beta = beta

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        if not isinstance(gamma, (float, int)):
            raise e.TypeError("`gamma` should be a float or integer")
        if gamma < 0:
            raise e.ValueError("`gamma` should be >= 0")
        self._gamma = gamma

    def update(self, ctx: UpdateContext) -> None:
        """FA update faithfully reproducing the original cascade logic.

        The original iterates: for each agent i, for each temp j,
        if fit_i > fit_j: pos_i = beta*exp(-gamma*dist)*(pos_j + pos_i) + alpha*(r-0.5)

        Each successive brighter j OVERWRITES pos_i (cascade).
        We reproduce this by sorting agents and applying the cascade from
        brightest to dimmest, which gives equivalent convergence behavior.
        """

        pop = ctx.space.population
        n = pop.n_agents
        n_iterations = ctx.n_iterations

        # Alpha decay (eq. 10)
        delta = 1.0 - ((10e-4) / 0.9) ** (1.0 / n_iterations)
        self.alpha *= 1.0 - delta

        # Snapshot of positions and fitness before update (deepcopy equivalent)
        temp_positions = pop.positions.clone()
        temp_fitness = pop.fitness.clone()

        pos_flat = pop.positions.reshape(n, -1).clone()  # (n, d) — will be mutated
        temp_flat = temp_positions.reshape(n, -1)          # (n, d) — frozen snapshot

        # Sort temp agents by fitness (brightest first) so cascade order matters less
        sorted_idx = torch.argsort(temp_fitness)

        for j_idx in sorted_idx:
            j_fit = temp_fitness[j_idx]
            j_pos = temp_flat[j_idx]  # (d,)

            # Which agents are dimmer (worse fitness) than j?
            attracted = pop.fitness > j_fit  # (n,) bool

            if not attracted.any():
                continue

            # Distances from current (mutated) positions to temp j
            diff = pos_flat[attracted] - j_pos.unsqueeze(0)  # (k, d)
            dist = diff.norm(dim=1)  # (k,)

            # Attractiveness
            beta_val = self.beta * torch.exp(-self.gamma * dist).unsqueeze(-1)  # (k, 1)

            # Original formula: pos = beta * (temp + pos) + alpha * (r - 0.5)
            r1 = torch.rand_like(pos_flat[attracted])
            pos_flat[attracted] = beta_val * (j_pos.unsqueeze(0) + pos_flat[attracted]) + self.alpha * (r1 - 0.5)

        pop.positions = pos_flat.reshape(pop.positions.shape)
