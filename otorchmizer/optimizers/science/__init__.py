# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Science-based optimizers inspired by physical and chemical phenomena."""

from otorchmizer.optimizers.science.aso import ASO
from otorchmizer.optimizers.science.bh import BH
from otorchmizer.optimizers.science.eo import EO
from otorchmizer.optimizers.science.gsa import GSA
from otorchmizer.optimizers.science.mvo import MVO
from otorchmizer.optimizers.science.sa import SA
from otorchmizer.optimizers.science.science_extra import (
    AIG,
    CDO,
    EFO,
    ESA,
    HGSO,
    LSA,
    MOA,
    SMA,
    TEO,
    TWO,
    WEO,
)
from otorchmizer.optimizers.science.wca import WCA
from otorchmizer.optimizers.science.wdo import WDO
from otorchmizer.optimizers.science.wwo import WWO

__all__ = [
    "AIG",
    "ASO",
    "BH",
    "CDO",
    "EFO",
    "EO",
    "ESA",
    "GSA",
    "HGSO",
    "LSA",
    "MOA",
    "MVO",
    "SA",
    "SMA",
    "TEO",
    "TWO",
    "WCA",
    "WDO",
    "WEO",
    "WWO",
]
