# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for isolated implementation comparison tooling."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import compare_implementations as comparison

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compare_implementations.py"


def _target_arguments(
    output: Path,
    optimizers: tuple[str, ...] = ("PSO",),
    objectives: tuple[str, ...] = ("rosenbrock",),
    agents: int = 8,
    variables: int = 3,
    iterations: int = 3,
    seed: int = 41,
) -> list[str]:
    arguments = [
        sys.executable,
        str(TOOL),
        "compare",
        "--runner",
        f"target={sys.executable}",
        "--source",
        f"target={ROOT}",
        "--agents",
        str(agents),
        "--variables",
        str(variables),
        "--iterations",
        str(iterations),
        "--repeats",
        "1",
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    for optimizer in optimizers:
        arguments.extend(("--optimizer", optimizer))
    for objective in objectives:
        arguments.extend(("--objective", objective))
    return arguments


def test_compare_cli_records_inventory_budget_and_invariants(tmp_path):
    output = tmp_path / "comparison.json"

    completed = subprocess.run(_target_arguments(output), capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.0"
    assert report["scope"]["export_inventory_is_complete"] is True
    assert report["scope"]["execution_is_representative_not_exhaustive"] is True
    assert report["scope"]["memory_compared"] is False

    inventory = report["inventories"]["target"]
    assert inventory["status"] == "ok"
    assert inventory["inventory"]["total_exports"] == 97
    assert inventory["inventory"]["total_unique_exports"] == 97
    assert "LOA" in inventory["inventory"]["families"]["population"]
    assert "PSO" in inventory["inventory"]["families"]["swarm"]

    run = report["runs"][0]
    assert run["status"] == "ok"
    assert run["implementation"]["worker_pid"] != report["coordinator"]["process_id"]
    assert Path(run["implementation"]["module_path"]).is_relative_to(ROOT)
    assert run["case"]["initial_population_sha256"] in report["initial_populations"]
    assert run["measurements"]["objective_evaluations"] == 32
    assert run["measurements"]["wall_time_seconds"] is None
    assert run["result"]["positions_shape"] == [8, 3, 1]
    assert run["result"]["stored_fitness_semantics"] == "current_position"
    assert len(run["execution"]["actual_bounds"]) == 2
    assert (
        run["execution"]["actual_bounds"]
        == report["initial_populations"][run["case"]["initial_population_sha256"]]["bounds"]
    )
    assert run["invariants"]["all_required"] is True
    assert run["invariants"]["best_position_fitness_consistent"] is True
    assert run["invariants"]["archive_matches_best_observed_evaluation"] is True

    pairing = report["pairings"][0]
    assert pairing["controlled_initial_population"] is True
    assert pairing["equal_iteration_budget"] is True
    assert pairing["objective_evaluations"] == {"target": 32}
    assert pairing["equal_objective_evaluation_count"] is None


def test_same_seed_and_initial_population_produce_deterministic_result(tmp_path):
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    first = subprocess.run(_target_arguments(first_output), capture_output=True, text=True, check=False)
    second = subprocess.run(_target_arguments(second_output), capture_output=True, text=True, check=False)

    assert first.returncode == second.returncode == 0
    first_report = json.loads(first_output.read_text(encoding="utf-8"))
    second_report = json.loads(second_output.read_text(encoding="utf-8"))
    first_run = first_report["runs"][0]
    second_run = second_report["runs"][0]

    assert first_report["initial_populations"] == second_report["initial_populations"]
    assert first_run["case"] == second_run["case"]
    assert first_run["result"] == second_run["result"]
    assert first_run["invariants"] == second_run["invariants"]
    assert first_run["optimizer_parameters"] == second_run["optimizer_parameters"]


def test_same_position_input_record_is_not_labeled_with_an_objective(tmp_path):
    output = tmp_path / "two-objectives.json"
    arguments = _target_arguments(
        output,
        objectives=("sphere", "rastrigin"),
        agents=4,
        variables=2,
        iterations=2,
        seed=17,
    )

    completed = subprocess.run(arguments, capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["initial_populations"]) == 1
    initialization = next(iter(report["initial_populations"].values()))
    assert "objective" not in initialization
    assert "objective_values" not in initialization
    assert "best_fitness" not in initialization
    assert {run["case"]["objective"] for run in report["runs"]} == {"sphere", "rastrigin"}
    assert len({run["case"]["initial_population_sha256"] for run in report["runs"]}) == 1


def test_hc_fa_and_ackley_execute_through_the_comparison_cli(tmp_path):
    output = tmp_path / "additional-selections.json"
    arguments = _target_arguments(
        output,
        optimizers=("HC", "FA"),
        objectives=("ackley",),
        agents=4,
        variables=2,
        iterations=2,
        seed=23,
    )

    completed = subprocess.run(arguments, capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert {(run["case"]["optimizer"], run["case"]["objective"]) for run in report["runs"]} == {
        ("HC", "ackley"),
        ("FA", "ackley"),
    }
    assert all(run["status"] == "ok" for run in report["runs"])
    assert all(run["measurements"]["objective_evaluations"] == 12 for run in report["runs"])


@pytest.mark.parametrize(
    ("stored_fitness", "current_fitness", "semantics", "expected_length_check", "expected_status"),
    [
        ([1.0, 2.0], [1.0, 3.0], "current_position", True, "invariant_failure"),
        ([1.0], [1.0, 3.0], "current_position", False, "invariant_failure"),
        ([1.0, 2.0], [1.0, 3.0], "personal_best", True, "ok"),
    ],
)
def test_fitness_correspondence_respects_declared_semantics(
    stored_fitness,
    current_fitness,
    semantics,
    expected_length_check,
    expected_status,
):
    summary = comparison._summarize_result(
        positions=[[[0.0]], [[1.0]]],
        position_shape=[2, 1, 1],
        stored_fitness=stored_fitness,
        current_fitness=current_fitness,
        best_position=[[0.0]],
        best_fitness=1.0,
        best_recomputed=1.0,
        observed_best_fitness=1.0,
        lower_bound=-1.0,
        upper_bound=1.0,
        actual_dtype="float64",
        actual_device="cpu",
        stored_fitness_semantics=semantics,
        personal_best_state_consistent=True,
        objective_evaluations=2,
        wall_time_seconds=None,
    )

    assert summary["invariants"]["population_lengths_consistent"] is expected_length_check
    assert summary["invariants"]["stored_fitness_matches_current_positions"] is False
    assert summary["status"] == expected_status


def test_invariant_failure_makes_comparison_exit_nonzero(monkeypatch, tmp_path):
    output = tmp_path / "invariant-failure.json"

    def invoke(_runner, request, _timeout):
        if request["action"] == "inventory":
            return {
                "status": "ok",
                "inventory": {"families": {family: [] for family in comparison.FAMILIES}},
            }
        return {"status": "invariant_failure", "invariants": {"all_required": False}}

    monkeypatch.setattr(comparison, "_invoke_worker", invoke)
    status = comparison.main(_target_arguments(output)[2:])

    assert status == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runs"][0]["status"] == "invariant_failure"


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["inventory", "--runner", "target", "--source", f"target={ROOT}"], "NAME=VALUE"),
        (
            [
                "inventory",
                "--runner",
                f"current={sys.executable}",
                "--source",
                f"current={ROOT}",
                "--device",
                "current=cuda",
            ],
            "must be cpu",
        ),
        (
            [
                "compare",
                "--runner",
                f"target={sys.executable}",
                "--source",
                f"target={ROOT}",
                "--iterations",
                "1",
            ],
            "at least 2",
        ),
    ],
)
def test_cli_rejects_invalid_configuration(arguments, message):
    completed = subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_worker_source_mismatch_is_a_structured_failure(tmp_path):
    empty_source = tmp_path / "empty-source"
    empty_source.mkdir()
    output = tmp_path / "failure.json"
    arguments = [
        sys.executable,
        str(TOOL),
        "inventory",
        "--runner",
        f"target={sys.executable}",
        "--source",
        f"target={empty_source}",
        "--output",
        str(output),
    ]

    completed = subprocess.run(arguments, capture_output=True, text=True, check=False)

    assert completed.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    failure = report["inventories"]["target"]
    assert failure["status"] == "error"
    assert failure["error"]["kind"] == "worker_failure"
    assert "outside requested source" in failure["error"]["stderr"]
    assert report["inventory_comparison"]["shared_by_all"] == []


def test_hidden_worker_rejects_adapter_mismatch():
    request = {
        "action": "inventory",
        "adapter": "current",
        "source_root": str(ROOT),
    }

    completed = subprocess.run(
        [sys.executable, str(TOOL), "_worker", "--adapter", "target"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1"},
    )

    assert completed.returncode != 0
    assert "`request.adapter` must match" in completed.stderr
