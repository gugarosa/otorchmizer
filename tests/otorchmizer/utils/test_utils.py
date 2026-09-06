# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Tests for callback, history, logging, and constant utilities."""

import logging

import pytest
import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.population import Population
from otorchmizer.utils import logging as log_module
from otorchmizer.utils.callback import (
    Callback,
    CheckpointCallback,
    DiscreteSearchCallback,
)
from otorchmizer.utils.history import History


class TestCallback:
    def test_base_callback_hooks_exist(self):
        cb = Callback()
        # All hooks should be callable no-ops
        cb.on_task_begin(None)
        cb.on_task_end(None)
        cb.on_iteration_begin(0, None)
        cb.on_iteration_end(0, None)
        cb.on_evaluate_before(None, None)
        cb.on_evaluate_after(None, None)
        cb.on_update_before(None)
        cb.on_update_after(None)


class TestCheckpointCallback:
    def test_creation(self):
        cb = CheckpointCallback(file_path="test.pkl", frequency=10)
        assert cb.file_path == "test.pkl"
        assert cb.frequency == 10

    def test_zero_frequency_no_save(self):
        cb = CheckpointCallback(frequency=0)
        # Should not raise even if opt_model is None (frequency=0 skips save)
        cb.on_iteration_end(0, None)


class TestDiscreteSearchCallback:
    def test_creation(self):
        cb = DiscreteSearchCallback(allowed_values=[[1, 2, 3]])
        assert cb.allowed_values == [[1, 2, 3]]

    def test_snap_to_nearest(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([10.0])
        pop = Population(3, 1, 1, lb, ub)
        pop.positions = torch.tensor([[[2.3]], [[7.8]], [[4.6]]])

        cb = DiscreteSearchCallback(allowed_values=[[0.0, 5.0, 10.0]])
        cb.on_evaluate_before(pop, None)

        snapped = pop.positions.squeeze().tolist()
        assert snapped[0] == 0.0  # 2.3 → 0.0
        assert snapped[1] == 10.0  # 7.8 → 10.0
        assert snapped[2] == 5.0  # 4.6 → 5.0


class TestHistory:
    def test_creation(self):
        h = History()
        assert h.save_agents is False

    def test_save_agents_flag(self):
        h = History(save_agents=True)
        assert h.save_agents is True

    def test_invalid_save_agents(self):
        with pytest.raises(TypeError):
            History(save_agents="yes")

    def test_dump_creates_attribute(self):
        h = History()
        h.dump(best_fitness=1.5)
        assert hasattr(h, "best_fitness")
        assert h.best_fitness == [1.5]

    def test_dump_appends(self):
        h = History()
        h.dump(best_fitness=1.0)
        h.dump(best_fitness=0.5)
        assert h.best_fitness == [1.0, 0.5]

    def test_dump_tensor_converts_to_python(self):
        h = History()
        h.dump(value=torch.tensor(42.0))
        assert isinstance(h.value[0], float)

    def test_dump_best_agent(self):
        h = History()
        pos = torch.tensor([1.0, 2.0])
        fit = torch.tensor(3.0)
        h.dump(best_agent=(pos, fit))
        assert hasattr(h, "best_agent")
        stored_pos, stored_fit = h.best_agent[0]
        assert stored_pos == [1.0, 2.0]
        assert stored_fit == 3.0

    def test_dump_positions_skipped_without_flag(self):
        h = History(save_agents=False)
        h.dump(positions=torch.rand(5, 2, 1))
        assert not hasattr(h, "positions")

    def test_dump_positions_saved_with_flag(self):
        h = History(save_agents=True)
        h.dump(positions=torch.rand(5, 2, 1))
        assert hasattr(h, "positions")

    def test_dump_fitness(self):
        h = History()
        h.dump(fitness=torch.tensor([1.0, 2.0, 3.0]))
        assert h.fitness[0] == [1.0, 2.0, 3.0]

    def test_get_convergence(self):
        h = History()
        for i in range(5):
            h.dump(loss=float(5 - i))
        result = h.get_convergence("loss")
        assert len(result) == 5
        assert result[0] == 5.0

    def test_get_convergence_best_agent(self):
        h = History()
        for i in range(3):
            h.dump(best_agent=(torch.tensor([float(i)]), torch.tensor(float(i * 2))))
        positions, fitnesses = h.get_convergence("best_agent")
        assert len(positions) == 3
        assert len(fitnesses) == 3


class TestLogging:
    def test_get_logger(self):
        logger = log_module.get_logger("test_logger")
        assert logger is logging.getLogger("test_logger")

    def test_logger_preserves_application_configuration(self):
        logger = logging.getLogger("test_handler_logger")
        logger.setLevel(logging.WARNING)
        logger.propagate = True
        handlers = logger.handlers.copy()
        logger_class = logging.getLoggerClass()

        assert log_module.get_logger(logger.name) is logger
        assert logger.handlers == handlers
        assert logger.level == logging.WARNING
        assert logger.propagate
        assert logging.getLoggerClass() is logger_class


class TestConstants:
    def test_epsilon(self):
        assert c.EPSILON == 1e-32

    def test_float_max_finite(self):
        assert c.FLOAT_MAX < float("inf")
        assert c.FLOAT_MAX > 0

    def test_light_speed(self):
        assert c.LIGHT_SPEED == 3e5

    def test_function_n_args(self):
        assert c.FUNCTION_N_ARGS["SUM"] == 2
        assert c.FUNCTION_N_ARGS["EXP"] == 1
        assert len(c.FUNCTION_N_ARGS) == 10
