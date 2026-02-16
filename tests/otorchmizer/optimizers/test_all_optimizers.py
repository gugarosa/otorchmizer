"""Tests for all migrated optimizers — verifies instantiation, compile, update, and convergence."""

import torch
import pytest
import logging

logging.disable(logging.CRITICAL)

from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.function import Function
from otorchmizer.core.space import Space


def sphere_fn(x):
    return (x ** 2).sum(dim=(-1, -2))


def make_ctx(space, fn, iteration=0, n_iterations=10):
    return UpdateContext(
        space=space, function=fn, iteration=iteration,
        n_iterations=n_iterations, device=torch.device("cpu"),
    )


def run_optimizer(cls, n_iter=20, n_agents=15, binary=False, params=None):
    """Run an optimizer for n_iter iterations and return final best fitness."""
    opt = cls(params)
    fn = Function(sphere_fn)
    lb = torch.zeros(5)
    ub = torch.ones(5) * 5
    space = Space(n_agents=n_agents, n_variables=5, n_dimensions=1,
                  lower_bound=lb, upper_bound=ub)
    if binary:
        space.population.initialize_binary()
    else:
        space.population.initialize_uniform()
    opt.compile(space.population)
    opt.evaluate(space.population, fn)

    for i in range(n_iter):
        ctx = make_ctx(space, fn, i, n_iter)
        opt.update(ctx)
        space.population.clip()
        opt.evaluate(space.population, fn)

    return space.population.best_fitness.item()


# ============================================================
# Swarm Optimizers
# ============================================================

from otorchmizer.optimizers.swarm import (
    ABC, ABO, AF, BA, BOA, BWO, CS, CSA, EHO, FFOA, FPA, FSO,
    GOA, JS, KH, MFO, MRFO, PIO, SBO, SCA, SFO, SOS, SSA, SSO,
    STOA, WAOA,
)


class TestSwarmOptimizers:
    @pytest.mark.parametrize("cls,name", [
        (ABC, "ABC"), (ABO, "ABO"), (AF, "AF"), (BA, "BA"),
        (BOA, "BOA"), (BWO, "BWO"), (CS, "CS"), (CSA, "CSA"),
        (EHO, "EHO"), (FFOA, "FFOA"), (FPA, "FPA"), (FSO, "FSO"),
        (GOA, "GOA"), (JS, "JS"), (KH, "KH"), (MFO, "MFO"),
        (MRFO, "MRFO"), (PIO, "PIO"), (SBO, "SBO"), (SCA, "SCA"),
        (SFO, "SFO"), (SOS, "SOS"), (SSA, "SSA"), (SSO, "SSO"),
        (STOA, "STOA"), (WAOA, "WAOA"),
    ])
    def test_runs_without_error(self, cls, name):
        """Each optimizer completes 20 iterations on sphere function."""
        best_fit = run_optimizer(cls)
        assert best_fit < 200, f"{name} fitness too high: {best_fit}"

    @pytest.mark.parametrize("cls", [CS, FPA, BA, BOA, SCA])
    def test_improves_from_initial(self, cls):
        """Selected optimizers should improve from random initialization."""
        torch.manual_seed(42)
        best_fit = run_optimizer(cls, n_iter=50)
        assert best_fit < 30, f"{cls.__name__} didn't improve enough: {best_fit}"


# ============================================================
# Evolutionary Optimizers
# ============================================================

from otorchmizer.optimizers.evolutionary import (
    BSA, DE, EP, ES, FOA, HS, IHS, GHS, SGHS, NGHS, GOGHS, IWO, RRA,
)


class TestEvolutionaryOptimizers:
    @pytest.mark.parametrize("cls,name", [
        (BSA, "BSA"), (DE, "DE"), (EP, "EP"), (ES, "ES"),
        (FOA, "FOA"), (HS, "HS"), (IHS, "IHS"), (GHS, "GHS"),
        (SGHS, "SGHS"), (NGHS, "NGHS"), (GOGHS, "GOGHS"),
        (IWO, "IWO"), (RRA, "RRA"),
    ])
    def test_runs_without_error(self, cls, name):
        best_fit = run_optimizer(cls)
        assert best_fit < 200, f"{name} fitness too high: {best_fit}"

    def test_de_convergence(self):
        torch.manual_seed(42)
        best_fit = run_optimizer(DE, n_iter=50)
        assert best_fit < 20, f"DE didn't converge: {best_fit}"


# ============================================================
# Misc Optimizers
# ============================================================

from otorchmizer.optimizers.misc import AOA, CEM, DOA


class TestMiscOptimizers:
    @pytest.mark.parametrize("cls,name", [
        (AOA, "AOA"), (CEM, "CEM"), (DOA, "DOA"),
    ])
    def test_runs_without_error(self, cls, name):
        best_fit = run_optimizer(cls)
        assert best_fit < 200, f"{name} fitness too high: {best_fit}"


# ============================================================
# Population Optimizers
# ============================================================

from otorchmizer.optimizers.population import (
    AEO, AO, COA, EPO, GCO, GWO, HHO, OSA, PPA, PVS, RFO,
)


class TestPopulationOptimizers:
    @pytest.mark.parametrize("cls,name", [
        (AEO, "AEO"), (AO, "AO"), (COA, "COA"), (EPO, "EPO"),
        (GCO, "GCO"), (GWO, "GWO"), (HHO, "HHO"), (OSA, "OSA"),
        (PPA, "PPA"), (PVS, "PVS"), (RFO, "RFO"),
    ])
    def test_runs_without_error(self, cls, name):
        best_fit = run_optimizer(cls)
        assert best_fit < 200, f"{name} fitness too high: {best_fit}"

    def test_gwo_convergence(self):
        torch.manual_seed(42)
        best_fit = run_optimizer(GWO, n_iter=50)
        assert best_fit < 20, f"GWO didn't converge: {best_fit}"


# ============================================================
# Science Optimizers
# ============================================================

from otorchmizer.optimizers.science import (
    AIG, ASO, BH, CDO, EFO, EO, ESA, GSA, HGSO, LSA, MOA, MVO,
    SA, SMA, TEO, TWO, WCA, WDO, WEO,
)


class TestScienceOptimizers:
    @pytest.mark.parametrize("cls,name", [
        (AIG, "AIG"), (ASO, "ASO"), (BH, "BH"), (CDO, "CDO"),
        (EFO, "EFO"), (EO, "EO"), (ESA, "ESA"), (GSA, "GSA"),
        (HGSO, "HGSO"), (LSA, "LSA"), (MOA, "MOA"), (MVO, "MVO"),
        (SA, "SA"), (SMA, "SMA"), (TEO, "TEO"), (TWO, "TWO"),
        (WCA, "WCA"), (WDO, "WDO"), (WEO, "WEO"),
    ])
    def test_runs_without_error(self, cls, name):
        best_fit = run_optimizer(cls)
        assert best_fit < 200, f"{name} fitness too high: {best_fit}"

    def test_sa_convergence(self):
        torch.manual_seed(42)
        best_fit = run_optimizer(SA, n_iter=100)
        assert best_fit < 30, f"SA didn't converge: {best_fit}"


# ============================================================
# Social Optimizers
# ============================================================

from otorchmizer.optimizers.social import BSO, CI, ISA, MVPA, QSA, SSD


class TestSocialOptimizers:
    @pytest.mark.parametrize("cls,name", [
        (BSO, "BSO"), (CI, "CI"), (ISA, "ISA"),
        (MVPA, "MVPA"), (QSA, "QSA"), (SSD, "SSD"),
    ])
    def test_runs_without_error(self, cls, name):
        best_fit = run_optimizer(cls)
        assert best_fit < 200, f"{name} fitness too high: {best_fit}"


# ============================================================
# Boolean Optimizers
# ============================================================

from otorchmizer.optimizers.boolean import BMRFO, BPSO, UMDA


class TestBooleanOptimizers:
    @pytest.mark.parametrize("cls,name", [
        (BMRFO, "BMRFO"), (BPSO, "BPSO"), (UMDA, "UMDA"),
    ])
    def test_runs_without_error(self, cls, name):
        best_fit = run_optimizer(cls, binary=True)
        assert best_fit >= 0, f"{name} returned negative fitness"
