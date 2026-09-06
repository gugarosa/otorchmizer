# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Grid Search — exhaustive evaluation of grid points.

References:
    J. Bergstra and Y. Bengio.
    Random search for hyper-parameter optimization.
    Journal of Machine Learning Research (2012).
"""

from __future__ import annotations

from typing import Any

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GS(Optimizer):
    """Grid Search optimizer.

    Notes:
        Leaves positions unchanged because GridSpace precomputes every candidate point.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the GS optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> GS.")
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one GS step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pass
