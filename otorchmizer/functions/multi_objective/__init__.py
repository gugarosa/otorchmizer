# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Multi-objective function wrappers."""

from otorchmizer.functions.multi_objective.standard import MultiObjectiveFunction
from otorchmizer.functions.multi_objective.weighted import MultiObjectiveWeightedFunction

__all__ = ["MultiObjectiveFunction", "MultiObjectiveWeightedFunction"]
