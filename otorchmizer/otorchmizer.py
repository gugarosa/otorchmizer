"""Otorchmizer — main optimization entry point.

Orchestrates the full optimization loop:
evaluate → update → clip → record history → repeat
"""

from __future__ import annotations

import time
from typing import List, Optional

import dill
from tqdm import tqdm

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.space import Space
from otorchmizer.utils import logging
from otorchmizer.utils.callback import Callback, CallbackVessel
from otorchmizer.utils.history import History

logger = logging.get_logger(__name__)


class Otorchmizer:
    """Holds all information needed to perform an optimization task.

    Wires together a Space (population), Optimizer (algorithm),
    and Function (objective), then runs the optimization loop with
    callbacks, history tracking, and checkpoint support.
    """

    def __init__(
        self,
        space: Space,
        optimizer: Optimizer,
        function: Function,
        save_agents: bool = False,
    ) -> None:
        """Initialization method.

        Args:
            space: A built Space instance.
            optimizer: A built Optimizer instance.
            function: A built Function instance.
            save_agents: Whether to save all agent positions per iteration.
        """

        logger.info("Creating class: Otorchmizer.")

        if not space.built:
            raise e.BuildError("`space` should be built before using Otorchmizer")
        if not optimizer.built:
            raise e.BuildError("`optimizer` should be built before using Otorchmizer")
        if not function.built:
            raise e.BuildError("`function` should be built before using Otorchmizer")

        self.space = space
        self.optimizer = optimizer
        self.function = function

        self.optimizer.compile(space.population)

        self.history = History(save_agents=save_agents)

        self.iteration = 0
        self.total_iterations = 0
        self.n_iterations = 0

        logger.debug(
            "Space: %s | Optimizer: %s | Function: %s.",
            self.space, self.optimizer, self.function,
        )
        logger.info("Class created.")

    def _make_context(self) -> UpdateContext:
        """Creates an UpdateContext for the current iteration."""

        return UpdateContext(
            space=self.space,
            function=self.function,
            iteration=self.iteration,
            n_iterations=self.n_iterations,
            device=self.space.device,
        )

    def evaluate(self, callbacks: CallbackVessel) -> None:
        """Runs the evaluation pipeline with callbacks.

        Args:
            callbacks: Callback vessel for lifecycle hooks.
        """

        callbacks.on_evaluate_before(self.space.population, self.function)
        self.optimizer.evaluate(self.space.population, self.function)
        callbacks.on_evaluate_after(self.space.population, self.function)

    def update(self, callbacks: CallbackVessel) -> None:
        """Runs the update pipeline with callbacks and bound clipping.

        Args:
            callbacks: Callback vessel for lifecycle hooks.
        """

        ctx = self._make_context()

        callbacks.on_update_before(ctx)
        self.optimizer.update(ctx)
        callbacks.on_update_after(ctx)

        self.space.clip()

    def start(
        self,
        n_iterations: int = 1,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """Starts the optimization task.

        Args:
            n_iterations: Maximum number of iterations.
            callbacks: List of Callback instances.
        """

        logger.info("Starting optimization task.")

        self.n_iterations = n_iterations
        vessel = CallbackVessel(callbacks)

        start_time = time.time()

        vessel.on_task_begin(self)

        # Initial evaluation
        self.evaluate(vessel)

        with tqdm(total=n_iterations, ascii=True) as bar:
            for t in range(n_iterations):
                logger.to_file(f"Iteration {t + 1}/{n_iterations}")

                self.total_iterations += 1
                self.iteration = t

                vessel.on_iteration_begin(self.total_iterations, self)

                self.update(vessel)
                self.evaluate(vessel)

                best_fit = self.space.population.best_fitness.item()
                bar.set_postfix(fitness=best_fit)
                bar.update()

                self.history.dump(
                    best_agent=(
                        self.space.population.best_position,
                        self.space.population.best_fitness,
                    ),
                    positions=self.space.population.positions,
                    fitness=self.space.population.fitness,
                )

                vessel.on_iteration_end(self.total_iterations, self)

                logger.to_file(f"Fitness: {best_fit}")

        vessel.on_task_end(self)

        elapsed = time.time() - start_time
        self.history.dump(time=elapsed)

        logger.info("Optimization task ended.")
        logger.info("It took %s seconds.", elapsed)

    def save(self, file_path: str) -> None:
        """Saves the optimization model to a dill file.

        Args:
            file_path: Output file path.
        """

        with open(file_path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> Otorchmizer:
        """Loads an optimization model from a dill file.

        Args:
            file_path: Input file path.

        Returns:
            Loaded Otorchmizer instance.
        """

        with open(file_path, "rb") as f:
            return dill.load(f)
