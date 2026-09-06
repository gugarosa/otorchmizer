# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Lion Optimization Algorithm.

References:
    M. Yazdani and F. Jolai.
    Lion Optimization Algorithm (LOA): A nature-inspired metaheuristic algorithm.
    Journal of Computational Design and Engineering (2016).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class _LionBatch:
    """Store aligned tensor state for a cohort of lions.

    Attributes:
        positions: Current candidate positions.
        fitness: Fitness corresponding to the current positions.
        best_positions: Personal best positions.
        best_fitness: Fitness corresponding to the personal best positions.
        female: Whether each lion is female.
        pride: Pride index, or -1 for a nomad.
        group: Hunting group, where 0 denotes a non-hunter.
        success: Whether each lion improved in the previous iteration.
        improved: Whether each lion improved in the current iteration.

    """

    positions: torch.Tensor
    fitness: torch.Tensor
    best_positions: torch.Tensor
    best_fitness: torch.Tensor
    female: torch.Tensor
    pride: torch.Tensor
    group: torch.Tensor
    success: torch.Tensor
    improved: torch.Tensor

    @property
    def size(self) -> int:
        return self.positions.shape[0]

    def take(self, indices: torch.Tensor) -> _LionBatch:
        if indices.dtype == torch.bool:
            indices = indices.nonzero(as_tuple=True)[0]
        return _LionBatch(
            self.positions.index_select(0, indices),
            self.fitness.index_select(0, indices),
            self.best_positions.index_select(0, indices),
            self.best_fitness.index_select(0, indices),
            self.female.index_select(0, indices),
            self.pride.index_select(0, indices),
            self.group.index_select(0, indices),
            self.success.index_select(0, indices),
            self.improved.index_select(0, indices),
        )

    @classmethod
    def concatenate(cls, batches: list[_LionBatch]) -> _LionBatch:
        populated = [batch for batch in batches if batch.size]
        if not populated:
            raise e.ValueError("`batches` must contain at least one lion.")
        return cls(
            *(torch.cat([getattr(batch, field) for batch in populated], dim=0) for field in cls.__dataclass_fields__)
        )


class LOA(Optimizer):
    """Apply pride and nomad social behavior from the Lion Optimization Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> LOA.")

        self.N = 0.2
        self.P = 4
        self.S = 0.8
        self.R = 0.2
        self.I = 0.4
        self.Ma = 0.3
        self.Mu = 0.2

        super().__init__(params)

        logger.info("Class overrided.")

    @staticmethod
    def _validate_ratio(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise e.TypeError(f"`{name}` must be a float or integer.")
        if not 0 <= value <= 1:
            raise e.ValueError(f"`{name}` must be between 0 and 1.")
        return float(value)

    @property
    def N(self) -> float:
        """Return the nomad-lion ratio."""

        return self._N

    @N.setter
    def N(self, N: float) -> None:
        self._N = self._validate_ratio("N", N)

    @property
    def P(self) -> int:
        """Return the number of prides."""

        return self._P

    @P.setter
    def P(self, P: int) -> None:
        if isinstance(P, bool) or not isinstance(P, int):
            raise e.TypeError("`P` must be an integer.")
        if P <= 0:
            raise e.ValueError("`P` must be positive.")
        self._P = P

    @property
    def S(self) -> float:
        """Return the female ratio within each pride."""

        return self._S

    @S.setter
    def S(self, S: float) -> None:
        self._S = self._validate_ratio("S", S)

    @property
    def R(self) -> float:
        """Return the resident-male roaming ratio."""

        return self._R

    @R.setter
    def R(self, R: float) -> None:
        self._R = self._validate_ratio("R", R)

    @property
    def I(self) -> float:  # noqa: E743
        """Return the female immigration ratio."""

        return self._I

    @I.setter
    def I(self, value: float) -> None:  # noqa: E743
        self._I = self._validate_ratio("I", value)

    @property
    def Ma(self) -> float:
        """Return the mating probability."""

        return self._Ma

    @Ma.setter
    def Ma(self, Ma: float) -> None:
        self._Ma = self._validate_ratio("Ma", Ma)

    @property
    def Mu(self) -> float:
        """Return the per-gene mutation probability."""

        return self._Mu

    @Mu.setter
    def Mu(self, Mu: float) -> None:
        self._Mu = self._validate_ratio("Mu", Mu)

    @staticmethod
    def _gender_count(ratio: float, size: int) -> int:
        return min(max(round(ratio * size), 1), size - 1)

    def compile(self, population: Population) -> None:
        """Initialize persistent personal-best and demographic state.

        Args:
            population: Population that defines state shape, device, and dtype.

        Raises:
            ValueError: The requested population cannot provide nomad and pride lions of both sexes.

        """

        n_nomads = round(self.N * population.n_agents)
        n_residents = population.n_agents - n_nomads
        if n_nomads < 2:
            raise e.ValueError("`N * population.n_agents` must allocate at least 2 nomad lions.")
        if n_residents < 2 * self.P:
            raise e.ValueError("`(1 - N) * population.n_agents` must allocate at least 2 lions per pride.")

        device = population.device
        n = population.n_agents
        permutation = torch.randperm(n, device=device)
        nomad_indices = permutation[:n_nomads]
        resident_indices = permutation[n_nomads:]

        self.pride_sizes = torch.full((self.P,), n_residents // self.P, dtype=torch.long, device=device)
        self.pride_sizes[: n_residents % self.P] += 1
        self.pride_females = torch.empty(self.P, dtype=torch.long, device=device)

        self.pride = torch.full((n,), -1, dtype=torch.long, device=device)
        self.female = torch.zeros(n, dtype=torch.bool, device=device)
        cursor = 0
        for pride_index in range(self.P):
            pride_size = int(self.pride_sizes[pride_index])
            members = resident_indices[cursor : cursor + pride_size]
            self.pride[members] = pride_index

            n_females = self._gender_count(self.S, pride_size)
            self.pride_females[pride_index] = n_females
            gender_order = members[torch.randperm(pride_size, device=device)]
            self.female[gender_order[:n_females]] = True
            cursor += pride_size

        self.nomad_females = self._gender_count(1 - self.S, n_nomads)
        nomad_order = nomad_indices[torch.randperm(n_nomads, device=device)]
        self.female[nomad_order[: self.nomad_females]] = True

        self.n_nomads = n_nomads
        self.group = torch.zeros(n, dtype=torch.long, device=device)
        self.nomad = self.pride < 0
        self.local_position = population.positions.clone()
        self.local_fitness = population.fitness.clone()
        self.improved = torch.ones(n, dtype=torch.bool, device=device)
        self._compiled_n_agents = n

    def _check_compiled(self, population: Population) -> None:
        if not hasattr(self, "_compiled_n_agents"):
            raise e.BuildError("`LOA.compile` must be called before evaluation or update.")
        if self._compiled_n_agents != population.n_agents:
            raise e.SizeError("`population.n_agents` must match the population used by `LOA.compile`.")
        if (
            self.local_position.device != population.positions.device
            or self.local_position.dtype != population.positions.dtype
        ):
            raise e.ValueError("`population` device and dtype must match the population used by `LOA.compile`.")

    @staticmethod
    def _record_global(population: Population, positions: torch.Tensor, fitness: torch.Tensor) -> None:
        best_index = fitness.argmin()
        if fitness[best_index] < population.best_fitness:
            population.best_position = positions[best_index].clone()
            population.best_fitness = fitness[best_index].clone()

    def evaluate(self, population: Population, function: Function) -> None:
        """Evaluate lions and update their personal and global best positions.

        Args:
            population: Population to evaluate.
            function: Objective function applied to the population.

        """

        self._check_compiled(population)
        fitness = function(population.positions).to(device=population.device, dtype=population.dtype)
        improved = fitness < self.local_fitness
        if improved.any():
            self.local_position[improved] = population.positions[improved].clone()
            self.local_fitness[improved] = fitness[improved].clone()
            self.improved |= improved

        population.fitness = fitness
        self._record_global(population, population.positions, fitness)

    def _make_lions(self, population: Population) -> _LionBatch:
        return _LionBatch(
            positions=population.positions.clone(),
            fitness=population.fitness.clone(),
            best_positions=self.local_position.clone(),
            best_fitness=self.local_fitness.clone(),
            female=self.female.clone(),
            pride=self.pride.clone(),
            group=self.group.clone(),
            success=self.improved.clone(),
            improved=torch.zeros_like(self.improved),
        )

    def _evaluate_move(
        self,
        lions: _LionBatch,
        indices: torch.Tensor,
        candidates: torch.Tensor,
        population: Population,
        function: Function,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = indices.reshape(-1)
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0)
        candidates = candidates.clamp(min=population.lb, max=population.ub)
        if not torch.isfinite(candidates).all():
            raise e.ValueError("`candidate positions` must be finite.")

        fitness = function(candidates).to(device=population.device, dtype=population.dtype)
        lions.positions[indices] = candidates
        lions.fitness[indices] = fitness

        improved = fitness < lions.best_fitness[indices]
        if improved.any():
            improved_indices = indices[improved]
            lions.best_positions[improved_indices] = candidates[improved].clone()
            lions.best_fitness[improved_indices] = fitness[improved].clone()
            lions.improved[improved_indices] = True

        self._record_global(population, candidates, fitness)
        return fitness, improved

    @staticmethod
    def _empty_like(lions: _LionBatch) -> _LionBatch:
        indices = torch.empty(0, dtype=torch.long, device=lions.positions.device)
        return lions.take(indices)

    @staticmethod
    def _orthogonal_directions(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = delta.reshape(-1)
        distance = torch.linalg.vector_norm(flat)
        epsilon = torch.finfo(delta.dtype).eps
        if distance <= epsilon:
            zeros = torch.zeros_like(delta)
            return zeros, zeros, distance

        direction_flat = flat / distance
        if flat.numel() == 1:
            return direction_flat.reshape_as(delta), torch.zeros_like(delta), distance

        random_flat = torch.randn_like(flat)
        orthogonal_flat = random_flat - torch.dot(random_flat, direction_flat) * direction_flat
        orthogonal_norm = torch.linalg.vector_norm(orthogonal_flat)
        if orthogonal_norm <= epsilon:
            basis = torch.zeros_like(flat)
            basis[torch.argmin(direction_flat.abs())] = 1
            orthogonal_flat = basis - torch.dot(basis, direction_flat) * direction_flat
            orthogonal_norm = torch.linalg.vector_norm(orthogonal_flat)

        return (
            direction_flat.reshape_as(delta),
            (orthogonal_flat / orthogonal_norm).reshape_as(delta),
            distance,
        )

    @staticmethod
    def _hunting_candidate(position: torch.Tensor, prey: torch.Tensor, center: bool) -> torch.Tensor:
        opposite = 2 * prey - position
        endpoint = position if center else opposite
        low = torch.minimum(endpoint, prey)
        high = torch.maximum(endpoint, prey)
        return low + torch.rand_like(position) * (high - low)

    def _hunting(self, lions: _LionBatch, population: Population, function: Function) -> None:
        lions.group.zero_()
        for pride_index in range(self.P):
            members = (lions.pride == pride_index).nonzero(as_tuple=True)[0]
            females = members[lions.female[members]]
            if not females.numel():
                continue

            assignments = torch.randint(0, 4, (females.numel(),), device=population.device)
            if not (assignments > 0).any():
                selected = torch.randint(0, females.numel(), (1,), device=population.device)
                assignments[selected] = torch.randint(1, 4, (1,), device=population.device)
            lions.group[females] = assignments

            hunters = females[lions.group[females] > 0]
            prey = lions.positions[hunters].mean(dim=0)
            group_fitness = population.fitness.new_full((3,), -torch.inf)
            for group_index in range(1, 4):
                group_members = hunters[lions.group[hunters] == group_index]
                if group_members.numel():
                    group_fitness[group_index - 1] = lions.fitness[group_members].sum()
            center_group = group_fitness.argmax() + 1

            hunter_order = hunters[torch.randperm(hunters.numel(), device=population.device)]
            for hunter in hunter_order:
                previous_fitness = lions.fitness[hunter].clone()
                candidate = self._hunting_candidate(
                    lions.positions[hunter],
                    prey,
                    bool(lions.group[hunter] == center_group),
                )
                fitness, _ = self._evaluate_move(lions, hunter, candidate, population, function)
                if fitness[0] < previous_fitness:
                    denominator = previous_fitness.abs().clamp_min(torch.finfo(population.dtype).eps)
                    percentage = ((previous_fitness - fitness[0]) / denominator).clamp_min(0)
                    escape = torch.rand((), device=population.device, dtype=population.dtype)
                    prey = prey + escape * percentage * (prey - candidate)
                    prey = prey.clamp(min=population.lb, max=population.ub)

    def _safe_place_candidate(self, position: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        direction, orthogonal, distance = self._orthogonal_directions(target - position)
        if not torch.count_nonzero(direction):
            return position.clone()

        device = position.device
        dtype = position.dtype
        forward = 2 * distance * torch.rand((), device=device, dtype=dtype)
        lateral = 2 * torch.rand((), device=device, dtype=dtype) - 1
        theta = (torch.rand((), device=device, dtype=dtype) - 0.5) * (torch.pi / 3)
        return position + forward * direction + lateral * torch.tan(theta) * distance * orthogonal

    def _moving_safe_place(self, lions: _LionBatch, population: Population, function: Function) -> None:
        for pride_index in range(self.P):
            members = (lions.pride == pride_index).nonzero(as_tuple=True)[0]
            non_hunters = members[lions.female[members] & (lions.group[members] == 0)]
            if not non_hunters.numel():
                continue

            successes = int(lions.success[members].sum())
            tournament_size = min(members.numel(), max(2, (successes + 1) // 2))
            for lion in non_hunters:
                contestants = members[torch.randperm(members.numel(), device=population.device)[:tournament_size]]
                winner = contestants[lions.best_fitness[contestants].argmin()]
                candidate = self._safe_place_candidate(lions.positions[lion], lions.best_positions[winner])
                self._evaluate_move(lions, lion, candidate, population, function)

    def _roaming_candidate(self, position: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        direction, orthogonal, distance = self._orthogonal_directions(target - position)
        if not torch.count_nonzero(direction):
            return position.clone()

        device = position.device
        dtype = position.dtype
        step = 2 * distance * torch.rand((), device=device, dtype=dtype)
        if position.numel() == 1:
            return position + step * direction

        theta = (torch.rand((), device=device, dtype=dtype) - 0.5) * (torch.pi / 3)
        rotated_direction = torch.cos(theta) * direction + torch.sin(theta) * orthogonal
        return position + step * rotated_direction

    def _pride_roaming(self, lions: _LionBatch, population: Population, function: Function) -> None:
        for pride_index in range(self.P):
            members = (lions.pride == pride_index).nonzero(as_tuple=True)[0]
            males = members[~lions.female[members]]
            n_territories = min(members.numel(), round(self.R * members.numel()))
            for male in males:
                selected = members[torch.randperm(members.numel(), device=population.device)[:n_territories]]
                for territory in selected:
                    candidate = self._roaming_candidate(
                        lions.positions[male],
                        lions.best_positions[territory],
                    )
                    self._evaluate_move(lions, male, candidate, population, function)

                lions.positions[male] = lions.best_positions[male].clone()
                lions.fitness[male] = lions.best_fitness[male].clone()

    def _mating_operator(
        self,
        lions: _LionBatch,
        female_index: torch.Tensor,
        male_indices: torch.Tensor,
        pride_index: int,
        population: Population,
        function: Function,
    ) -> _LionBatch:
        female_position = lions.positions[female_index]
        male_position = lions.positions[male_indices].mean(dim=0)
        beta = 0.5 + 0.1 * torch.randn((), device=population.device, dtype=population.dtype)
        children = torch.stack(
            (
                beta * female_position + (1 - beta) * male_position,
                (1 - beta) * female_position + beta * male_position,
            )
        )

        mutation = torch.rand_like(children) < self.Mu
        random_positions = population.lb + torch.rand_like(children) * (population.ub - population.lb)
        children = torch.where(mutation, random_positions, children)
        children = children.clamp(min=population.lb, max=population.ub)
        child_fitness = function(children).to(device=population.device, dtype=population.dtype)
        self._record_global(population, children, child_fitness)

        female = torch.zeros(2, dtype=torch.bool, device=population.device)
        female[torch.randint(0, 2, (1,), device=population.device)] = True
        pride = torch.full((2,), pride_index, dtype=torch.long, device=population.device)
        return _LionBatch(
            positions=children,
            fitness=child_fitness,
            best_positions=children.clone(),
            best_fitness=child_fitness.clone(),
            female=female,
            pride=pride,
            group=torch.zeros(2, dtype=torch.long, device=population.device),
            success=torch.ones(2, dtype=torch.bool, device=population.device),
            improved=torch.ones(2, dtype=torch.bool, device=population.device),
        )

    def _pride_mating(self, lions: _LionBatch, population: Population, function: Function) -> _LionBatch:
        cubs = []
        for pride_index in range(self.P):
            members = (lions.pride == pride_index).nonzero(as_tuple=True)[0]
            females = members[lions.female[members]]
            males = members[~lions.female[members]]
            mating_females = females[
                torch.rand(females.numel(), device=population.device, dtype=population.dtype) < self.Ma
            ]
            for female in mating_females:
                n_males = torch.randint(1, males.numel() + 1, (1,), device=population.device)
                selected = males[torch.randperm(males.numel(), device=population.device)[:n_males]]
                cubs.append(self._mating_operator(lions, female, selected, pride_index, population, function))

        return _LionBatch.concatenate(cubs) if cubs else self._empty_like(lions)

    def _defense(self, lions: _LionBatch, cubs: _LionBatch) -> _LionBatch:
        if cubs.size:
            lions = _LionBatch.concatenate([lions, cubs])

        for pride_index in range(self.P):
            resident_males = ((lions.pride == pride_index) & ~lions.female).nonzero(as_tuple=True)[0]
            target_males = int(self.pride_sizes[pride_index] - self.pride_females[pride_index])
            if resident_males.numel() < target_males:
                raise e.ValueError(f"`pride {pride_index}` must contain at least {target_males} male lions.")

            ranking = resident_males[torch.argsort(lions.fitness[resident_males])]
            expelled = ranking[target_males:]
            lions.pride[expelled] = -1
            lions.group[expelled] = 0

        return lions

    def _nomad_roaming(self, lions: _LionBatch, population: Population, function: Function) -> None:
        nomads = (lions.pride < 0).nonzero(as_tuple=True)[0]
        best_fitness = lions.fitness[nomads].min()
        denominator = best_fitness.abs().clamp_min(torch.finfo(population.dtype).eps)

        for nomad in nomads:
            relative_gap = ((lions.fitness[nomad] - best_fitness) / denominator).clamp(0, 0.5)
            probability = 0.1 + relative_gap
            random_position = population.lb + torch.rand_like(lions.positions[nomad]) * (population.ub - population.lb)
            move = torch.rand_like(lions.positions[nomad]) <= probability
            candidate = torch.where(move, random_position, lions.positions[nomad])
            self._evaluate_move(lions, nomad, candidate, population, function)

    def _nomad_mating(self, lions: _LionBatch, population: Population, function: Function) -> _LionBatch:
        nomads = (lions.pride < 0).nonzero(as_tuple=True)[0]
        females = nomads[lions.female[nomads]]
        males = nomads[~lions.female[nomads]]
        mating_females = females[
            torch.rand(females.numel(), device=population.device, dtype=population.dtype) < self.Ma
        ]

        cubs = []
        for female in mating_females:
            selected = males[torch.randint(0, males.numel(), (1,), device=population.device)]
            cubs.append(self._mating_operator(lions, female, selected, -1, population, function))

        return _LionBatch.concatenate([lions, *cubs]) if cubs else lions

    def _nomad_attack(self, lions: _LionBatch, population: Population) -> None:
        nomad_males = ((lions.pride < 0) & ~lions.female).nonzero(as_tuple=True)[0]
        for nomad in nomad_males:
            if lions.pride[nomad] >= 0:
                continue

            attacks = torch.rand(self.P, device=population.device, dtype=population.dtype) < 0.5
            for pride_index in range(self.P):
                if not attacks[pride_index]:
                    continue

                resident_males = ((lions.pride == pride_index) & ~lions.female).nonzero(as_tuple=True)[0]
                weakest = resident_males[lions.fitness[resident_males].argmax()]
                if lions.fitness[nomad] < lions.fitness[weakest]:
                    lions.pride[nomad] = pride_index
                    lions.pride[weakest] = -1
                    lions.group[nomad] = 0
                    lions.group[weakest] = 0
                    break

    def _migration(self, lions: _LionBatch, population: Population) -> None:
        for pride_index in range(self.P):
            females = ((lions.pride == pride_index) & lions.female).nonzero(as_tuple=True)[0]
            target = int(self.pride_females[pride_index])
            surplus = max(females.numel() - target, 0)
            n_migrating = min(females.numel(), surplus + round(self.I * target))
            migrating = females[torch.randperm(females.numel(), device=population.device)[:n_migrating]]
            lions.pride[migrating] = -1
            lions.group[migrating] = 0

        deficits = torch.empty(self.P, dtype=torch.long, device=population.device)
        for pride_index in range(self.P):
            current = ((lions.pride == pride_index) & lions.female).sum()
            deficits[pride_index] = self.pride_females[pride_index] - current

        total_deficit = int(deficits.sum())
        if not total_deficit:
            return

        nomad_females = ((lions.pride < 0) & lions.female).nonzero(as_tuple=True)[0]
        if nomad_females.numel() < total_deficit:
            raise e.ValueError("`nomad females` must fill every pride migration vacancy.")
        best = nomad_females[torch.argsort(lions.fitness[nomad_females])[:total_deficit]]
        best = best[torch.randperm(best.numel(), device=population.device)]

        cursor = 0
        for pride_index in range(self.P):
            count = int(deficits[pride_index])
            selected = best[cursor : cursor + count]
            lions.pride[selected] = pride_index
            lions.group[selected] = 0
            cursor += count

    def _population_control(self, lions: _LionBatch, population: Population) -> _LionBatch:
        keep = lions.pride >= 0
        nomad_females = ((lions.pride < 0) & lions.female).nonzero(as_tuple=True)[0]
        nomad_males = ((lions.pride < 0) & ~lions.female).nonzero(as_tuple=True)[0]
        target_females = self.nomad_females
        target_males = self.n_nomads - target_females
        if nomad_females.numel() < target_females or nomad_males.numel() < target_males:
            raise e.ValueError("`nomads` must contain enough lions of each sex for population equilibrium.")

        female_keep = nomad_females[torch.argsort(lions.fitness[nomad_females])[:target_females]]
        male_keep = nomad_males[torch.argsort(lions.fitness[nomad_males])[:target_males]]
        keep[female_keep] = True
        keep[male_keep] = True
        controlled = lions.take(keep)

        if controlled.size != population.n_agents:
            raise e.SizeError(f"`controlled population` must contain {population.n_agents} lions.")
        return controlled

    def _validate_population(self, lions: _LionBatch, population: Population) -> None:
        if not torch.isfinite(lions.positions).all():
            raise e.ValueError("`population.positions` must be finite after an LOA update.")
        if (lions.positions < population.lb).any() or (lions.positions > population.ub).any():
            raise e.ValueError("`population.positions` must remain within bounds after an LOA update.")

        for pride_index in range(self.P):
            members = lions.pride == pride_index
            if int(members.sum()) != int(self.pride_sizes[pride_index]):
                raise e.SizeError(f"`pride {pride_index}` must retain its compiled size.")
            if int((members & lions.female).sum()) != int(self.pride_females[pride_index]):
                raise e.SizeError(f"`pride {pride_index}` must retain its compiled female count.")

        nomads = lions.pride < 0
        if int(nomads.sum()) != self.n_nomads:
            raise e.SizeError("`nomads` must retain their compiled population size.")
        if int((nomads & lions.female).sum()) != self.nomad_females:
            raise e.SizeError("`nomads` must retain their compiled female count.")

    def _synchronize(self, lions: _LionBatch, population: Population) -> None:
        self._validate_population(lions, population)
        population.positions = lions.positions
        population.fitness = lions.fitness
        self.local_position = lions.best_positions
        self.local_fitness = lions.best_fitness
        self.female = lions.female
        self.pride = lions.pride
        self.group = lions.group
        self.nomad = lions.pride < 0
        self.improved = lions.improved
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance hunting, movement, mating, defense, migration, and equilibrium.

        Args:
            ctx: Current population and objective function.

        """

        population = ctx.space.population
        self._check_compiled(population)
        if not torch.isfinite(population.fitness).all():
            self.evaluate(population, ctx.function)
        lions = self._make_lions(population)

        self._hunting(lions, population, ctx.function)
        self._moving_safe_place(lions, population, ctx.function)
        self._pride_roaming(lions, population, ctx.function)
        pride_cubs = self._pride_mating(lions, population, ctx.function)
        lions = self._defense(lions, pride_cubs)

        self._nomad_roaming(lions, population, ctx.function)
        lions = self._nomad_mating(lions, population, ctx.function)
        self._nomad_attack(lions, population)

        self._migration(lions, population)
        lions = self._population_control(lions, population)

        self._synchronize(lions, population)
