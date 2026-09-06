# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Social-based optimizers inspired by human group behavior."""

from otorchmizer.optimizers.social.bso import BSO
from otorchmizer.optimizers.social.ci import CI
from otorchmizer.optimizers.social.isa import ISA
from otorchmizer.optimizers.social.mvpa import MVPA
from otorchmizer.optimizers.social.qsa import QSA
from otorchmizer.optimizers.social.ssd import SSD

__all__ = ["BSO", "CI", "ISA", "MVPA", "QSA", "SSD"]
