# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Univariate Marginal Distribution Algorithm.

Marginal frequencies are clipped using equation 47 and resampled using equation 53.
Sampling sets a bit when ``uniform_draw < probs``.
Bounds are validated at construction and before updates or direct probability calculation.

References:
    H. Mühlenbein. The equation for response to selection and its use for prediction.
    Evolutionary Computation (1997).

"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class UMDA(Optimizer):
    """Univariate Marginal Distribution Algorithm.

    Notes:
        Samples Boolean candidates from independently estimated marginal probabilities.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the UMDA optimizer.

        Args:
            params: Algorithm parameter overrides.

        Raises:
            TypeError: A selection or probability parameter is not numeric.
            ValueError: The selection proportion or probability interval is invalid.

        """

        self._p_selection = 0.75
        self._lower_bound = 0.05
        self._upper_bound = 0.95
        super().__init__(params)
        self._validate_parameters()

    def build(self, params: dict[str, Any] | None = None) -> None:
        """Apply parameter overrides with atomic probability-bound validation.

        Args:
            params: Algorithm parameter overrides.

        """

        values = dict(params or {})
        lower_bound = values.pop("lower_bound", self.lower_bound)
        upper_bound = values.pop("upper_bound", self.upper_bound)
        self._validate_bounds(lower_bound, upper_bound)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        super().build(values)
        if params:
            self.params.update({name: params[name] for name in ("lower_bound", "upper_bound") if name in params})

    @property
    def p_selection(self) -> float:
        """Return the selected population fraction."""

        return self._p_selection

    @p_selection.setter
    def p_selection(self, value: float) -> None:
        if not isinstance(value, Real):
            raise TypeError("`p_selection` must be a real number.")
        if not isfinite(value) or not 0 < value <= 1:
            raise ValueError("`p_selection` must be finite, greater than 0, and at most 1.")
        self._p_selection = value

    @property
    def lower_bound(self) -> float:
        """Return the minimum marginal probability."""

        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value: float) -> None:
        if not isinstance(value, Real):
            raise TypeError("`lower_bound` must be a real number.")
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("`lower_bound` must be finite and between 0 and 1.")
        if value > self.upper_bound:
            raise ValueError("`lower_bound` must be finite and between 0 and `upper_bound`.")
        self._lower_bound = value

    @property
    def upper_bound(self) -> float:
        """Return the maximum marginal probability."""

        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, value: float) -> None:
        if not isinstance(value, Real):
            raise TypeError("`upper_bound` must be a real number.")
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("`upper_bound` must be finite and between 0 and 1.")
        if value < self.lower_bound:
            raise ValueError("`upper_bound` must be finite and between `lower_bound` and 1.")
        self._upper_bound = value

    def _validate_parameters(self) -> None:
        self.p_selection = self.p_selection
        self._validate_bounds(self.lower_bound, self.upper_bound)

    @staticmethod
    def _validate_bounds(lower_bound: float, upper_bound: float) -> None:
        for name, value in (("lower_bound", lower_bound), ("upper_bound", upper_bound)):
            if not isinstance(value, Real):
                raise TypeError(f"`{name}` must be a real number.")
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be finite and between 0 and 1.")
        if lower_bound > upper_bound:
            raise ValueError("`lower_bound` must not exceed `upper_bound`.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one UMDA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        self._validate_parameters()
        pop = ctx.space.population
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        n_selected = max(int(n * self.p_selection), 1)

        sorted_idx = torch.argsort(pop.fitness)
        selected = pop.positions[sorted_idx[:n_selected]]

        # Calculate probabilities
        probs = selected.mean(dim=0)
        probs = probs.clamp(min=self.lower_bound, max=self.upper_bound)

        # Sample new positions
        r = torch.rand_like(pop.positions)
        pop.positions = (probs.unsqueeze(0) > r).to(dtype=pop.dtype)
        pop.positions = pop.positions.clamp(min=lb, max=ub)
