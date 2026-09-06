# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Compare representative Otorchmizer and Opytimizer runs in isolated interpreters.

The coordinator never imports either implementation. Each requested source tree is
loaded by its own interpreter through an explicit version adapter, using a paired
initial population and a shared iteration budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
ADAPTERS = ("target", "current", "legacy")
FAMILIES = ("boolean", "evolutionary", "misc", "population", "science", "social", "swarm")
OPTIMIZER_MODULES = {
    "PSO": "swarm.pso",
    "WOA": "swarm.woa",
    "GA": "evolutionary.ga",
    "HC": "misc.hc",
    "FA": "swarm.fa",
}
OPTIMIZER_PARAMETERS = {
    "PSO": ("w", "c1", "c2"),
    "WOA": ("b",),
    "GA": ("p_selection", "p_mutation", "p_crossover"),
    "HC": ("r_mean", "r_var"),
    "FA": ("alpha", "beta", "gamma"),
}
OBJECTIVE_BOUNDS = {
    "sphere": (-5.12, 5.12),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-2.048, 2.048),
    "ackley": (-5.0, 5.0),
}
DEFAULT_OPTIMIZERS = ("PSO", "WOA", "GA")
DEFAULT_OBJECTIVES = ("sphere", "rastrigin", "rosenbrock")


@dataclass(frozen=True)
class _Runner:
    """Describe one isolated implementation process.

    Attributes:
        name: Explicit adapter name.
        python: Interpreter used for the worker process.
        source: Source checkout placed alone on PYTHONPATH.
        device: Requested execution device.
        dtype: Requested floating-point dtype.

    """

    name: str
    python: Path
    source: Path
    device: str
    dtype: str


class _NumpyObjectiveCounter:
    def __init__(self, numpy_module: Any, objective: str) -> None:
        self.np = numpy_module
        self.objective = objective
        self.evaluations = 0
        self.best_fitness: float | None = None
        self.best_position: Any = None

    def __call__(self, position: Any) -> float:
        value = float(_numpy_objective(self.np, self.objective, position))
        self.evaluations += 1
        if self.best_fitness is None or value < self.best_fitness:
            self.best_fitness = value
            self.best_position = self.np.asarray(position).copy()
        return value


class _TorchObjectiveCounter:
    def __init__(self, torch_module: Any, function: Any) -> None:
        self.torch = torch_module
        self.function = function
        self.evaluations = 0
        self.best_fitness: Any = None
        self.best_position: Any = None

    def __call__(self, positions: Any) -> Any:
        values = self.function(positions)
        self.evaluations += int(positions.shape[0])
        index = values.argmin()
        candidate_fitness = values[index].detach().clone()
        candidate_position = positions[index].detach().clone()
        if self.best_fitness is None:
            self.best_fitness = candidate_fitness
            self.best_position = candidate_position
        else:
            improved = candidate_fitness < self.best_fitness
            self.best_fitness = self.torch.where(improved, candidate_fitness, self.best_fitness)
            self.best_position = self.torch.where(improved, candidate_position, self.best_position)
        return values


def _parse_assignments(values: list[str], option: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"`{option}` entries must use NAME=VALUE.")
        name, assigned = value.split("=", 1)
        if name not in ADAPTERS:
            raise ValueError(f"`{option}` name must be one of {', '.join(ADAPTERS)}.")
        if not assigned:
            raise ValueError(f"`{option}` value for `{name}` must not be empty.")
        if name in assignments:
            raise ValueError(f"`{option}` must not repeat `{name}`.")
        assignments[name] = assigned
    return assignments


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled implementation comparisons in isolated Python processes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_runners(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--runner",
            action="append",
            required=True,
            metavar="NAME=PYTHON",
            help="Interpreter for target, current, or legacy; repeat for each implementation.",
        )
        command.add_argument(
            "--source",
            action="append",
            required=True,
            metavar="NAME=CHECKOUT",
            help="Matching source checkout used as the worker's isolated PYTHONPATH.",
        )
        command.add_argument(
            "--device",
            action="append",
            default=[],
            metavar="NAME=DEVICE",
            help="Execution device; only target accepts non-CPU values.",
        )
        command.add_argument(
            "--dtype",
            action="append",
            default=[],
            metavar="NAME=DTYPE",
            help="float32 or float64 for an implementation; defaults to float64.",
        )
        command.add_argument("--timeout", type=_positive_int, default=300, help="Worker timeout in seconds.")
        command.add_argument("--output", default="-", help="JSON output path, or - for stdout.")

    inventory = subparsers.add_parser("inventory", help="Inventory every public optimizer family export.")
    add_runners(inventory)

    compare = subparsers.add_parser("compare", help="Run representative controlled numeric comparisons.")
    add_runners(compare)
    compare.add_argument(
        "--optimizer",
        action="append",
        choices=tuple(OPTIMIZER_MODULES),
        help="Optimizer; repeat as needed. Defaults to PSO, WOA, and GA.",
    )
    compare.add_argument(
        "--objective",
        action="append",
        choices=tuple(OBJECTIVE_BOUNDS),
        help="Objective; repeat as needed. Defaults to sphere, rastrigin, and rosenbrock.",
    )
    compare.add_argument("--agents", type=_positive_int, default=20, help="Population size.")
    compare.add_argument("--variables", type=_positive_int, default=5, help="Number of scalar variables.")
    compare.add_argument("--iterations", type=_positive_int, default=20, help="Iteration budget of at least two.")
    compare.add_argument("--repeats", type=_positive_int, default=3, help="Independent paired repeats.")
    compare.add_argument("--seed", type=int, default=1234, help="Base initialization and algorithm seed.")
    compare.add_argument(
        "--wall-time",
        action="store_true",
        help="Record raw worker run time without calculating speedup claims.",
    )

    return parser


def _build_runners(args: argparse.Namespace) -> dict[str, _Runner]:
    interpreters = _parse_assignments(args.runner, "--runner")
    sources = _parse_assignments(args.source, "--source")
    devices = _parse_assignments(args.device, "--device")
    dtypes = _parse_assignments(args.dtype, "--dtype")

    if set(interpreters) != set(sources):
        raise ValueError("`--runner` and `--source` must define the same implementation names.")
    if not set(devices) <= set(interpreters):
        raise ValueError("`--device` must reference a configured runner.")
    if not set(dtypes) <= set(interpreters):
        raise ValueError("`--dtype` must reference a configured runner.")

    runners = {}
    for name, interpreter in interpreters.items():
        python = Path(interpreter).expanduser().resolve()
        source = Path(sources[name]).expanduser().resolve()
        device = devices.get(name, "cpu")
        dtype = dtypes.get(name, "float64")

        if not python.is_file():
            raise ValueError(f"`--runner {name}` must point to an existing interpreter.")
        if not source.is_dir():
            raise ValueError(f"`--source {name}` must point to an existing checkout.")
        if dtype not in {"float32", "float64"}:
            raise ValueError(f"`--dtype {name}` must be float32 or float64.")
        if name != "target" and device != "cpu":
            raise ValueError(f"`--device {name}` must be cpu for a NumPy reference.")

        runners[name] = _Runner(name, python, source, device, dtype)
    return runners


def _git_output(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"`git {' '.join(arguments)}` failed for `{source}`: {result.stderr.strip()}")
    return result.stdout.strip()


def _implementation_metadata(adapter: str, source_root: str) -> dict[str, Any]:
    package_name = "otorchmizer" if adapter == "target" else "opytimizer"
    package = importlib.import_module(package_name)
    source = Path(source_root).resolve()
    module_path = Path(package.__file__).resolve()
    try:
        module_path.relative_to(source)
    except ValueError as error:
        message = f"`{package_name}` loaded from `{module_path}`, outside requested source `{source}`."
        raise RuntimeError(message) from error

    try:
        distribution_version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None

    metadata = {
        "adapter": adapter,
        "package": package_name,
        "module_version": getattr(package, "__version__", None),
        "distribution_version": distribution_version,
        "version_consistent": distribution_version in {None, getattr(package, "__version__", None)},
        "module_path": str(module_path),
        "source_root": str(source),
        "git_commit": _git_output(source, "rev-parse", "HEAD"),
        "git_describe": _git_output(source, "describe", "--tags", "--always", "--dirty"),
        "git_dirty": bool(_git_output(source, "status", "--porcelain")),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "worker_pid": os.getpid(),
    }

    if adapter == "target":
        torch = importlib.import_module("torch")
        metadata["backend"] = {
            "name": "pytorch",
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    else:
        np = importlib.import_module("numpy")
        metadata["backend"] = {"name": "numpy", "version": np.__version__, "device": "cpu"}
    return metadata


def _optimizer_inventory(adapter: str) -> dict[str, Any]:
    package_name = "otorchmizer" if adapter == "target" else "opytimizer"
    optimizer_module = importlib.import_module(f"{package_name}.core.optimizer")
    optimizer_base = optimizer_module.Optimizer
    families: dict[str, list[str]] = {}
    modules: dict[str, str] = {}

    for family in FAMILIES:
        module = importlib.import_module(f"{package_name}.optimizers.{family}")
        if adapter == "target":
            names = list(module.__all__)
        else:
            names = [
                name
                for name, value in vars(module).items()
                if not name.startswith("_")
                and inspect.isclass(value)
                and issubclass(value, optimizer_base)
                and value is not optimizer_base
            ]

        verified = []
        for name in sorted(names):
            value = getattr(module, name)
            if not inspect.isclass(value) or not issubclass(value, optimizer_base):
                raise RuntimeError(f"`{package_name}.optimizers.{family}.{name}` is not an optimizer class.")
            verified.append(name)
            modules[name] = value.__module__
        families[family] = verified

    flattened = [name for names in families.values() for name in names]
    duplicates = sorted({name for name in flattened if flattened.count(name) > 1})
    return {
        "export_method": "__all__" if adapter == "target" else "public optimizer subclasses imported by family",
        "families": families,
        "class_modules": dict(sorted(modules.items())),
        "total_exports": len(flattened),
        "total_unique_exports": len(set(flattened)),
        "duplicates": duplicates,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"`value` is not JSON scalar compatible: {type(value).__name__}.")


def _optimizer_parameters(optimizer: Any, optimizer_name: str) -> dict[str, Any]:
    return {name: _json_scalar(getattr(optimizer, name)) for name in OPTIMIZER_PARAMETERS[optimizer_name]}


def _torch_objective(torch: Any, name: str, position: Any) -> Any:
    values = position.reshape(-1)
    if name == "sphere":
        return values.square().sum()
    if name == "rastrigin":
        return 10 * values.numel() + (values.square() - 10 * torch.cos(2 * torch.pi * values)).sum()
    if name == "rosenbrock":
        return (100 * (values[1:] - values[:-1].square()).square() + (1 - values[:-1]).square()).sum()
    if name == "ackley":
        square_mean = values.square().mean()
        cosine_mean = torch.cos(2 * torch.pi * values).mean()
        return -20 * torch.exp(-0.2 * torch.sqrt(square_mean)) - torch.exp(cosine_mean) + 20 + torch.e
    raise ValueError(f"`objective` is unsupported: {name}.")


def _numpy_objective(np: Any, name: str, position: Any) -> Any:
    values = np.asarray(position).reshape(-1)
    if name == "sphere":
        return np.square(values).sum()
    if name == "rastrigin":
        return 10 * values.size + (np.square(values) - 10 * np.cos(2 * np.pi * values)).sum()
    if name == "rosenbrock":
        return (100 * np.square(values[1:] - np.square(values[:-1])) + np.square(1 - values[:-1])).sum()
    if name == "ackley":
        square_mean = np.square(values).mean()
        cosine_mean = np.cos(2 * np.pi * values).mean()
        return -20 * np.exp(-0.2 * np.sqrt(square_mean)) - np.exp(cosine_mean) + 20 + np.e
    raise ValueError(f"`objective` is unsupported: {name}.")


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shape(values: Any) -> list[int]:
    return list(values.shape)


def _close_tolerance(dtype: str) -> tuple[float, float]:
    return (1e-5, 1e-6) if dtype == "float32" else (1e-9, 1e-10)


def _summarize_result(
    *,
    positions: list[Any],
    position_shape: list[int],
    stored_fitness: list[float],
    current_fitness: list[float],
    best_position: list[Any],
    best_fitness: float,
    best_recomputed: float,
    observed_best_fitness: float,
    lower_bound: float,
    upper_bound: float,
    actual_dtype: str,
    actual_device: str,
    stored_fitness_semantics: str,
    personal_best_state_consistent: bool | None,
    objective_evaluations: int,
    wall_time_seconds: float | None,
) -> dict[str, Any]:
    relative, absolute = _close_tolerance(actual_dtype)
    flat_positions = [float(value) for agent in positions for variable in agent for value in variable]
    flat_best = [float(value) for variable in best_position for value in variable]
    population_lengths_consistent = (
        len(position_shape) == 3
        and position_shape[0] == len(positions)
        and len(positions) == len(stored_fitness)
        and len(stored_fitness) == len(current_fitness)
    )
    stored_matches = population_lengths_consistent and all(
        math.isclose(float(stored), float(current), rel_tol=relative, abs_tol=absolute)
        for stored, current in zip(stored_fitness, current_fitness)
    )
    archive_consistent = math.isclose(
        best_fitness,
        best_recomputed,
        rel_tol=relative,
        abs_tol=absolute,
    )
    archive_matches_observed = math.isclose(
        best_fitness,
        observed_best_fitness,
        rel_tol=relative,
        abs_tol=absolute,
    )
    current_best = min(current_fitness)
    archive_not_worse = best_fitness <= current_best + absolute + relative * abs(current_best)

    invariants = {
        "population_lengths_consistent": population_lengths_consistent,
        "positions_finite": all(math.isfinite(value) for value in flat_positions),
        "stored_fitness_finite": all(math.isfinite(float(value)) for value in stored_fitness),
        "objective_values_finite": all(math.isfinite(float(value)) for value in current_fitness),
        "positions_within_bounds": all(lower_bound <= value <= upper_bound for value in flat_positions),
        "best_position_finite": all(math.isfinite(value) for value in flat_best),
        "best_position_within_bounds": all(lower_bound <= value <= upper_bound for value in flat_best),
        "best_position_fitness_consistent": archive_consistent,
        "archive_matches_best_observed_evaluation": archive_matches_observed,
        "archive_not_worse_than_current_population": archive_not_worse,
        "stored_fitness_matches_current_positions": stored_matches,
        "personal_best_state_consistent": personal_best_state_consistent,
    }
    required = [
        invariants["population_lengths_consistent"],
        invariants["positions_finite"],
        invariants["stored_fitness_finite"],
        invariants["objective_values_finite"],
        invariants["positions_within_bounds"],
        invariants["best_position_finite"],
        invariants["best_position_within_bounds"],
        invariants["best_position_fitness_consistent"],
        invariants["archive_matches_best_observed_evaluation"],
        invariants["archive_not_worse_than_current_population"],
    ]
    if stored_fitness_semantics == "current_position":
        required.append(stored_matches)
    elif stored_fitness_semantics != "personal_best":
        raise ValueError(f"`stored_fitness_semantics` is unsupported: {stored_fitness_semantics}.")
    if personal_best_state_consistent is not None:
        required.append(personal_best_state_consistent)
    invariants["all_required"] = all(required)

    return {
        "status": "ok" if invariants["all_required"] else "invariant_failure",
        "measurements": {
            "objective_evaluations": objective_evaluations,
            "wall_time_seconds": wall_time_seconds,
            "wall_time_scope": (
                "initial evaluation plus update, clipping, and evaluation iterations; setup and imports excluded"
                if wall_time_seconds is not None
                else None
            ),
        },
        "result": {
            "positions_shape": position_shape,
            "best_position": best_position,
            "best_fitness": best_fitness,
            "best_position_recomputed_fitness": best_recomputed,
            "best_observed_evaluation_fitness": observed_best_fitness,
            "current_population_best_fitness": current_best,
            "stored_fitness_semantics": stored_fitness_semantics,
        },
        "execution": {
            "actual_dtype": actual_dtype,
            "actual_device": actual_device,
            "actual_bounds": [lower_bound, upper_bound],
        },
        "invariants": invariants,
    }


def _run_target(request: dict[str, Any]) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    function_module = importlib.import_module("otorchmizer.core.function")
    optimizer_module = importlib.import_module("otorchmizer.core.optimizer")
    space_module = importlib.import_module("otorchmizer.core.space")
    module_name = OPTIMIZER_MODULES[request["optimizer"]]
    optimizer_class = getattr(importlib.import_module(f"otorchmizer.optimizers.{module_name}"), request["optimizer"])

    dtype = getattr(torch, request["dtype"])
    device = torch.device(request["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`device` requested CUDA, but CUDA is unavailable.")

    lower, upper = request["bounds"]
    space = space_module.Space(
        n_agents=request["n_agents"],
        n_variables=request["n_variables"],
        n_dimensions=1,
        lower_bound=lower,
        upper_bound=upper,
        device=device,
        dtype=dtype,
    )
    space.build()
    initial = torch.tensor(request["initial_positions"], device=space.device, dtype=dtype).unsqueeze(-1)
    space.population.initialize_static(initial)

    optimizer = optimizer_class()
    optimizer.bind(space)
    optimizer.compile(space.population)
    base_function = function_module.Function(lambda position: _torch_objective(torch, request["objective"], position))
    function = _TorchObjectiveCounter(torch, base_function)
    _seed_torch(torch, request["seed"])

    if request["measure_wall_time"] and space.device.type == "cuda":
        torch.cuda.synchronize(space.device)
    started = time.perf_counter() if request["measure_wall_time"] else None

    optimizer.validate_space(space)
    optimizer.evaluate(space.population, function)
    for iteration in range(request["iterations"]):
        context = optimizer_module.UpdateContext(
            space=space,
            function=function,
            iteration=iteration,
            n_iterations=request["iterations"],
            device=space.device,
        )
        optimizer.validate_space(space)
        optimizer(context)
        optimizer.validate_space(space)
        space.clip()
        optimizer.validate_space(space)
        optimizer.evaluate(space.population, function)
        optimizer.validate_space(space)

    if request["measure_wall_time"] and space.device.type == "cuda":
        torch.cuda.synchronize(space.device)
    elapsed = time.perf_counter() - started if started is not None else None

    population = space.population
    current_fitness = base_function(population.positions)
    best_recomputed = base_function(population.best_position.unsqueeze(0))[0]
    personal_consistent = None
    if hasattr(optimizer, "local_position") and hasattr(optimizer, "local_fitness"):
        personal_values = base_function(optimizer.local_position)
        relative, absolute = _close_tolerance(request["dtype"])
        personal_consistent = bool(
            torch.allclose(personal_values, optimizer.local_fitness, rtol=relative, atol=absolute)
        )

    summary = _summarize_result(
        positions=population.positions.detach().cpu().tolist(),
        position_shape=_shape(population.positions),
        stored_fitness=[float(value) for value in population.fitness.detach().cpu().tolist()],
        current_fitness=[float(value) for value in current_fitness.detach().cpu().tolist()],
        best_position=population.best_position.detach().cpu().tolist(),
        best_fitness=float(population.best_fitness.detach().cpu()),
        best_recomputed=float(best_recomputed.detach().cpu()),
        observed_best_fitness=float(function.best_fitness.detach().cpu()),
        lower_bound=float(population.lb.min().detach().cpu()),
        upper_bound=float(population.ub.max().detach().cpu()),
        actual_dtype=str(population.positions.dtype).removeprefix("torch."),
        actual_device=str(population.positions.device),
        stored_fitness_semantics="current_position",
        personal_best_state_consistent=personal_consistent,
        objective_evaluations=function.evaluations,
        wall_time_seconds=elapsed,
    )
    summary["optimizer_parameters"] = _optimizer_parameters(optimizer, request["optimizer"])
    if population.positions.device.type == "cuda":
        summary["execution"]["device_name"] = torch.cuda.get_device_name(population.positions.device)
    return summary


def _prepare_opytimizer_space(np: Any, search_space: Any, request: dict[str, Any]) -> Any:
    dtype = getattr(np, request["dtype"])
    lower, upper = request["bounds"]
    space = search_space(
        n_agents=request["n_agents"],
        n_variables=request["n_variables"],
        lower_bound=np.full(request["n_variables"], lower, dtype=dtype),
        upper_bound=np.full(request["n_variables"], upper, dtype=dtype),
    )
    initial = np.asarray(request["initial_positions"], dtype=dtype).reshape(
        request["n_agents"],
        request["n_variables"],
        1,
    )
    for agent, position in zip(space.agents, initial):
        agent.position = position.copy()
        agent.fit = float("inf")
    space.best_agent.position = initial[0].copy()
    space.best_agent.fit = float("inf")
    return space


def _run_opytimizer(request: dict[str, Any], adapter: str) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    opytimizer_class = importlib.import_module("opytimizer.opytimizer").Opytimizer
    search_space = importlib.import_module("opytimizer.spaces.search").SearchSpace
    module_name = OPTIMIZER_MODULES[request["optimizer"]]
    optimizer_class = getattr(importlib.import_module(f"opytimizer.optimizers.{module_name}"), request["optimizer"])

    space = _prepare_opytimizer_space(np, search_space, request)
    optimizer = optimizer_class()
    counter = _NumpyObjectiveCounter(np, request["objective"])
    if adapter == "current":
        function = counter
        callbacks = None
    elif adapter == "legacy":
        function_class = importlib.import_module("opytimizer.core.function").Function
        callback_vessel = importlib.import_module("opytimizer.utils.callback").CallbackVessel
        function = function_class(counter)
        callbacks = callback_vessel([])
    else:
        raise ValueError(f"`adapter` is unsupported for Opytimizer: {adapter}.")

    model = opytimizer_class(space, optimizer, function)
    model.n_iterations = request["iterations"]
    np.random.seed(request["seed"])

    started = time.perf_counter() if request["measure_wall_time"] else None
    if adapter == "current":
        model.evaluate()
        for iteration in range(request["iterations"]):
            model.total_iterations += 1
            model.iteration = iteration
            model.update()
            model.evaluate()
    else:
        model.evaluate(callbacks)
        for iteration in range(request["iterations"]):
            model.total_iterations += 1
            model.iteration = iteration
            model.update(callbacks)
            model.evaluate(callbacks)
    elapsed = time.perf_counter() - started if started is not None else None

    return _summarize_opytimizer(request, np, optimizer, space, counter, elapsed)


def _summarize_opytimizer(
    request: dict[str, Any],
    np: Any,
    optimizer: Any,
    space: Any,
    function: _NumpyObjectiveCounter,
    elapsed: float | None,
) -> dict[str, Any]:
    positions = np.stack([agent.position for agent in space.agents])
    stored_fitness = np.asarray([agent.fit for agent in space.agents])
    current_fitness = np.asarray([_numpy_objective(np, request["objective"], position) for position in positions])
    best_position = np.asarray(space.best_agent.position)
    best_recomputed = float(_numpy_objective(np, request["objective"], best_position))
    personal_consistent = None
    if request["optimizer"] == "PSO":
        personal_values = np.asarray(
            [_numpy_objective(np, request["objective"], position) for position in optimizer.local_position]
        )
        relative, absolute = _close_tolerance(str(positions.dtype))
        personal_consistent = bool(np.allclose(personal_values, stored_fitness, rtol=relative, atol=absolute))

    summary = _summarize_result(
        positions=positions.tolist(),
        position_shape=_shape(positions),
        stored_fitness=[float(value) for value in stored_fitness.tolist()],
        current_fitness=[float(value) for value in current_fitness.tolist()],
        best_position=best_position.tolist(),
        best_fitness=float(space.best_agent.fit),
        best_recomputed=best_recomputed,
        observed_best_fitness=float(function.best_fitness),
        lower_bound=float(np.min(space.agents[0].lb)),
        upper_bound=float(np.max(space.agents[0].ub)),
        actual_dtype=str(positions.dtype),
        actual_device="cpu",
        stored_fitness_semantics="personal_best" if request["optimizer"] == "PSO" else "current_position",
        personal_best_state_consistent=personal_consistent,
        objective_evaluations=function.evaluations,
        wall_time_seconds=elapsed,
    )
    summary["optimizer_parameters"] = _optimizer_parameters(optimizer, request["optimizer"])
    return summary


def _perform_worker(adapter: str, request: dict[str, Any]) -> dict[str, Any]:
    if request.get("adapter") != adapter:
        raise ValueError("`request.adapter` must match the worker adapter.")
    metadata = _implementation_metadata(adapter, request["source_root"])
    action = request.get("action")
    if action == "inventory":
        return {
            "status": "ok",
            "implementation": metadata,
            "inventory": _optimizer_inventory(adapter),
        }
    if action != "compare":
        raise ValueError("`request.action` must be inventory or compare.")

    if adapter == "target":
        result = _run_target(request)
    elif adapter in {"current", "legacy"}:
        result = _run_opytimizer(request, adapter)
    else:
        raise ValueError(f"`adapter` is unsupported: {adapter}.")
    result["implementation"] = metadata
    result["case"] = {
        key: request[key]
        for key in (
            "case_id",
            "optimizer",
            "objective",
            "n_agents",
            "n_variables",
            "n_dimensions",
            "iterations",
            "repeat",
            "seed",
            "bounds",
            "initial_population_sha256",
            "dtype",
            "device",
        )
    }
    return result


def _worker_main(adapter: str) -> int:
    request = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        result = _perform_worker(adapter, request)
    json.dump(result, sys.stdout, allow_nan=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _tail_output(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def _invoke_worker(runner: _Runner, request: dict[str, Any], timeout: int) -> dict[str, Any]:
    script = Path(__file__).resolve()
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(runner.source)
    command = [str(runner.python), str(script), "_worker", "--adapter", runner.name]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, allow_nan=False),
            capture_output=True,
            text=True,
            cwd=runner.source,
            env=environment,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "error",
            "implementation_name": runner.name,
            "error": {
                "kind": "timeout",
                "message": f"Worker exceeded {timeout} seconds.",
                "stderr": _tail_output(error.stderr, 20000),
            },
        }

    if completed.returncode:
        return {
            "status": "error",
            "implementation_name": runner.name,
            "error": {
                "kind": "worker_failure",
                "returncode": completed.returncode,
                "stderr": _tail_output(completed.stderr, 20000),
                "stdout": _tail_output(completed.stdout, 2000),
            },
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "status": "error",
            "implementation_name": runner.name,
            "error": {
                "kind": "invalid_worker_output",
                "message": str(error),
                "stderr": _tail_output(completed.stderr, 20000),
                "stdout": _tail_output(completed.stdout, 2000),
            },
        }


def _inventory_comparison(inventories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = {
        implementation: {name for family_names in result["inventory"]["families"].values() for name in family_names}
        for implementation, result in inventories.items()
        if result.get("status") == "ok"
    }
    if not names:
        return {"shared_by_all": [], "missing_from_implementation": {}}

    union = set().union(*names.values())
    shared = set.intersection(*names.values())
    return {
        "shared_by_all": sorted(shared),
        "missing_from_implementation": {
            implementation: sorted(union - exported) for implementation, exported in names.items()
        },
    }


def _initial_population(
    lower_bound: float,
    upper_bound: float,
    n_agents: int,
    n_variables: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    generator = random.Random(seed)
    positions = [[generator.uniform(lower_bound, upper_bound) for _ in range(n_variables)] for _ in range(n_agents)]
    encoded = json.dumps(positions, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return digest, {
        "method": "python random.Random(seed).uniform(lower_bound, upper_bound)",
        "seed": seed,
        "bounds": [lower_bound, upper_bound],
        "shape": [n_agents, n_variables, 1],
        "sha256": digest,
        "positions": positions,
    }


def _write_report(report: dict[str, Any], output: str) -> None:
    serialized = json.dumps(report, indent=2, allow_nan=False, sort_keys=True)
    if output == "-":
        print(serialized)
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {path}", file=sys.stderr)


def _base_report(command: str, runners: dict[str, _Runner]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "coordinator": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "process_id": os.getpid(),
        },
        "configured_implementations": {
            name: {
                "python": str(runner.python),
                "source": str(runner.source),
                "requested_device": runner.device,
                "requested_dtype": runner.dtype,
            }
            for name, runner in runners.items()
        },
        "scope": {
            "export_inventory_is_complete": True,
            "execution_is_representative_not_exhaustive": True,
            "trajectory_parity_claimed": False,
            "application_wide_speedup_claimed": False,
            "memory_compared": False,
        },
    }


def _collect_inventories(
    args: argparse.Namespace,
    runners: dict[str, _Runner],
) -> tuple[dict[str, dict[str, Any]], bool]:
    inventories = {}
    for name, runner in runners.items():
        request = {"action": "inventory", "adapter": name, "source_root": str(runner.source)}
        inventories[name] = _invoke_worker(runner, request, args.timeout)
    return inventories, any(result.get("status") != "ok" for result in inventories.values())


def _run_inventory_command(args: argparse.Namespace, runners: dict[str, _Runner]) -> tuple[dict[str, Any], bool]:
    report = _base_report("inventory", runners)
    inventories, had_errors = _collect_inventories(args, runners)
    report["inventories"] = inventories
    report["inventory_comparison"] = _inventory_comparison(inventories)
    return report, had_errors


def _run_compare_command(args: argparse.Namespace, runners: dict[str, _Runner]) -> tuple[dict[str, Any], bool]:
    optimizers = args.optimizer or list(DEFAULT_OPTIMIZERS)
    objectives = args.objective or list(DEFAULT_OBJECTIVES)
    if args.iterations < 2:
        raise ValueError("`--iterations` must be at least 2 for the representative optimizer set.")
    if args.agents < 4:
        raise ValueError("`--agents` must be at least 4 for GA selection.")
    if "rosenbrock" in objectives and args.variables < 2:
        raise ValueError("`--variables` must be at least 2 for rosenbrock.")

    report = _base_report("compare", runners)
    inventories, had_errors = _collect_inventories(args, runners)
    report["inventories"] = inventories
    report["inventory_comparison"] = _inventory_comparison(inventories)
    report["configuration"] = {
        "optimizers": optimizers,
        "objectives": objectives,
        "n_agents": args.agents,
        "n_variables": args.variables,
        "n_dimensions": 1,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "wall_time_measured": args.wall_time,
    }
    report["scope"]["execution_scope"] = "Selected numeric optimizers on bounded continuous scalar-variable spaces."
    report["scope"]["timing_note"] = (
        "Raw isolated-worker wall times are recorded only when requested; the tool computes no speedup ratio."
    )

    initial_populations: dict[str, dict[str, Any]] = {}
    runs = []
    pairing_records = []
    for objective in objectives:
        lower, upper = OBJECTIVE_BOUNDS[objective]
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            digest, initialization = _initial_population(lower, upper, args.agents, args.variables, seed)
            initial_populations.setdefault(digest, initialization)

            for optimizer in optimizers:
                case_id = f"{objective}:{optimizer}:repeat-{repeat}"
                case_runs = []
                for name, runner in runners.items():
                    request = {
                        "action": "compare",
                        "adapter": name,
                        "source_root": str(runner.source),
                        "case_id": case_id,
                        "optimizer": optimizer,
                        "objective": objective,
                        "n_agents": args.agents,
                        "n_variables": args.variables,
                        "n_dimensions": 1,
                        "iterations": args.iterations,
                        "repeat": repeat,
                        "seed": seed,
                        "bounds": [lower, upper],
                        "initial_population_sha256": digest,
                        "initial_positions": initialization["positions"],
                        "dtype": runner.dtype,
                        "device": runner.device,
                        "measure_wall_time": args.wall_time,
                    }
                    result = _invoke_worker(runner, request, args.timeout)
                    result["implementation_name"] = name
                    result["case_id"] = case_id
                    runs.append(result)
                    case_runs.append(result)
                    had_errors |= result.get("status") != "ok"

                successful = [result for result in case_runs if result.get("status") == "ok"]
                counts = {
                    result["implementation_name"]: result["measurements"]["objective_evaluations"]
                    for result in successful
                }
                pairing_records.append(
                    {
                        "case_id": case_id,
                        "controlled_initial_population": len(
                            {result["case"]["initial_population_sha256"] for result in successful}
                        )
                        <= 1,
                        "equal_iteration_budget": True,
                        "objective_evaluations": counts,
                        "equal_objective_evaluation_count": len(set(counts.values())) == 1 if len(counts) > 1 else None,
                        "successful_implementations": [result["implementation_name"] for result in successful],
                    }
                )

    report["initial_populations"] = initial_populations
    report["runs"] = runs
    report["pairings"] = pairing_records
    return report, had_errors


def main(argv: list[str] | None = None) -> int:
    """Run the comparison coordinator or one hidden isolated worker.

    Args:
        argv: Optional argument list excluding the executable name.

    Returns:
        Process status code.

    """

    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["_worker"]:
        worker_parser = argparse.ArgumentParser(prog="compare_implementations.py _worker", add_help=False)
        worker_parser.add_argument("_worker")
        worker_parser.add_argument("--adapter", required=True, choices=ADAPTERS)
        worker_args = worker_parser.parse_args(arguments)
        return _worker_main(worker_args.adapter)

    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        runners = _build_runners(args)
        if args.command == "inventory":
            report, had_errors = _run_inventory_command(args, runners)
        else:
            report, had_errors = _run_compare_command(args, runners)
    except ValueError as error:
        parser.error(str(error))

    _write_report(report, args.output)
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
