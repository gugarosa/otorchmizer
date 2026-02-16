"""Graph-based search space (experimental)."""

from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GraphSpace:
    """Placeholder for graph-structured optimization spaces.

    Manages blocks/cells in a DAG structure. To be fully
    implemented with graph-based optimizers.
    """

    def __init__(self, n_blocks: int) -> None:
        logger.info("Creating class: GraphSpace.")

        self.n_blocks = n_blocks
        self.built = True

        logger.debug("Blocks: %d | Built: %s.", self.n_blocks, self.built)
        logger.info("Class created.")
