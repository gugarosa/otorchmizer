"""Grid Search — exhaustive evaluation of grid points.

References:
    J. Bergstra and Y. Bengio.
    Random search for hyper-parameter optimization.
    Journal of Machine Learning Research (2012).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GS(Optimizer):
    """Grid Search optimizer.

    Does not update positions — evaluation happens on fixed grid points.
    Used with GridSpace which pre-computes all candidate positions.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> GS.")
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """No-op: grid positions are static."""
        pass
