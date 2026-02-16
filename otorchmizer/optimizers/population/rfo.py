"""Red Fox Optimization.

References:
    D. Polap and M. Woźniak.
    Red fox optimization algorithm.
    Expert Systems with Applications (2021).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.general as g
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class RFO(Optimizer):
    """Red Fox Optimization.

    Relocation, noticing, and habitat replacement phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> RFO.")

        self.phi = 3.14159
        self.theta = 0.5
        self.p_replacement = 0.05

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def phi(self) -> float:
        return self._phi

    @phi.setter
    def phi(self, phi: float) -> None:
        if not isinstance(phi, (float, int)):
            raise e.TypeError("`phi` should be a float or integer")
        self._phi = phi

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, theta: float) -> None:
        if not isinstance(theta, (float, int)):
            raise e.TypeError("`theta` should be a float or integer")
        self._theta = theta

    @property
    def p_replacement(self) -> float:
        return self._p_replacement

    @p_replacement.setter
    def p_replacement(self, p_replacement: float) -> None:
        if not isinstance(p_replacement, (float, int)):
            raise e.TypeError("`p_replacement` should be a float or integer")
        if not 0 <= p_replacement <= 1:
            raise e.ValueError("`p_replacement` should be between 0 and 1")
        self._p_replacement = p_replacement

    def compile(self, population) -> None:
        self.n_replacement = max(int(self.p_replacement * population.n_agents), 1)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        alpha = torch.rand(1, device=device) * 0.2

        sorted_idx = torch.argsort(pop.fitness)
        best_pos = pop.positions[sorted_idx[0]]
        second_pos = pop.positions[sorted_idx[1]] if n > 1 else best_pos

        for i in range(n):
            # Relocation
            dist = g.euclidean_distance(pop.positions[i].reshape(-1), best_pos.reshape(-1))
            dist = torch.sqrt(dist + 1e-10)
            sign = torch.sign(best_pos - pop.positions[i])
            pop.positions[i] = pop.positions[i] + torch.rand(1, device=device) * dist * sign

            # Noticing
            if torch.rand(1, device=device).item() > 0.75:
                if self.phi != 0:
                    radius = alpha * torch.sin(torch.tensor(self.phi, device=device)) / self.phi
                else:
                    radius = torch.tensor(self.theta, device=device)

                noise = alpha * radius * torch.randn_like(pop.positions[i])
                pop.positions[i] = pop.positions[i] + noise

        pop.positions = pop.positions.clamp(min=lb, max=ub)

        # Habitat replacement for worst agents
        center = (best_pos + second_pos) / 2
        diameter = g.euclidean_distance(best_pos.reshape(-1), second_pos.reshape(-1))
        diameter = torch.sqrt(diameter + 1e-10)

        worst_idx = torch.argsort(pop.fitness, descending=True)[:self.n_replacement]
        for idx in worst_idx:
            k = torch.rand(1, device=device)
            if k.item() >= 0.45:
                pop.positions[idx] = torch.rand_like(pop.positions[idx]) + center + diameter / 2
            else:
                pop.positions[idx] = k * (best_pos + second_pos) / 2

        pop.positions = pop.positions.clamp(min=lb, max=ub)
