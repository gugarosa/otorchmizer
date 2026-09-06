# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Run both genetic-programming variants on expression-tree phenotypes."""

import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function
from otorchmizer.optimizers.evolutionary import GP, GSGP
from otorchmizer.spaces import TreeSpace


def _sphere(position: torch.Tensor) -> torch.Tensor:
    return position.square().sum()


def main() -> None:
    """Optimize bounded tree phenotypes with GP and GSGP."""

    for optimizer_class in (GP, GSGP):
        space = TreeSpace(
            n_agents=20,
            n_variables=3,
            lower_bound=-2,
            upper_bound=2,
            n_terminals=4,
            functions=["SUM", "SUB", "MUL"],
            device="auto",
        )
        optimizer = optimizer_class()
        engine = Otorchmizer(space, optimizer, Function(_sphere))
        engine.start(n_iterations=3)
        print(f"{optimizer.algorithm}: {space.best_fitness.item():.6g}")


if __name__ == "__main__":
    main()
