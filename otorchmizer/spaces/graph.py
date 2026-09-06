# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Graph-based search space (experimental)."""

from __future__ import annotations

from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GraphSpace:
    """Experimental placeholder for graph-structured optimization spaces."""

    def __init__(self, n_blocks: int) -> None:
        """Initialize the graph-space placeholder.

        Args:
            n_blocks: Number of blocks represented by the placeholder.

        Notes:
            This class records graph-space metadata only. It does not yet implement graph construction or optimization.

        """

        logger.info("Creating class: GraphSpace.")

        self.n_blocks = n_blocks
        self.built = True

        logger.debug("Blocks: %d | Built: %s.", self.n_blocks, self.built)
        logger.info("Class created.")
