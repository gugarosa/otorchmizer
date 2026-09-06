# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Swarm-based optimizers inspired by collective animal behavior."""

from otorchmizer.optimizers.swarm.abc import ABC
from otorchmizer.optimizers.swarm.abo import ABO
from otorchmizer.optimizers.swarm.af import AF
from otorchmizer.optimizers.swarm.ba import BA
from otorchmizer.optimizers.swarm.boa import BOA
from otorchmizer.optimizers.swarm.bwo import BWO
from otorchmizer.optimizers.swarm.cs import CS
from otorchmizer.optimizers.swarm.csa import CSA
from otorchmizer.optimizers.swarm.eho import EHO
from otorchmizer.optimizers.swarm.fa import FA
from otorchmizer.optimizers.swarm.ffoa import FFOA
from otorchmizer.optimizers.swarm.fpa import FPA
from otorchmizer.optimizers.swarm.fso import FSO
from otorchmizer.optimizers.swarm.goa import GOA
from otorchmizer.optimizers.swarm.js import JS, NBJS
from otorchmizer.optimizers.swarm.kh import KH
from otorchmizer.optimizers.swarm.mfo import MFO
from otorchmizer.optimizers.swarm.mrfo import MRFO
from otorchmizer.optimizers.swarm.pio import PIO
from otorchmizer.optimizers.swarm.pso import AIWPSO, PSO, RPSO, SAVPSO, VPSO
from otorchmizer.optimizers.swarm.sbo import SBO
from otorchmizer.optimizers.swarm.sca import SCA
from otorchmizer.optimizers.swarm.sfo import SFO
from otorchmizer.optimizers.swarm.sos import SOS
from otorchmizer.optimizers.swarm.ssa import SSA
from otorchmizer.optimizers.swarm.sso import SSO
from otorchmizer.optimizers.swarm.stoa import STOA
from otorchmizer.optimizers.swarm.waoa import WAOA
from otorchmizer.optimizers.swarm.woa import WOA

__all__ = [
    "ABC",
    "ABO",
    "AF",
    "AIWPSO",
    "BA",
    "BOA",
    "BWO",
    "CS",
    "CSA",
    "EHO",
    "FA",
    "FFOA",
    "FPA",
    "FSO",
    "GOA",
    "JS",
    "KH",
    "MFO",
    "MRFO",
    "NBJS",
    "PIO",
    "PSO",
    "RPSO",
    "SAVPSO",
    "SBO",
    "SCA",
    "SFO",
    "SOS",
    "SSA",
    "SSO",
    "STOA",
    "VPSO",
    "WAOA",
    "WOA",
]
