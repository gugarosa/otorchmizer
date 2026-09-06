# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Run Lion Optimization with a viable population of prides and nomads."""

import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.population import LOA


def _sphere(position: torch.Tensor) -> torch.Tensor:
    return position.square().sum()


def main() -> None:
    """Optimize a sphere objective with the default lion demographics."""

    space = Space(n_agents=60, n_variables=4, lower_bound=-2, upper_bound=2, device="auto")
    space.build()
    optimizer = LOA()
    engine = Otorchmizer(space, optimizer, Function(_sphere))
    engine.start(n_iterations=3)
    print(f"{optimizer.algorithm}: {space.best_fitness.item():.6g}")


if __name__ == "__main__":
    main()
