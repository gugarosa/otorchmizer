"""Tests for utils — callback, history, logging, exception, constant."""

import os
import tempfile

import pytest
import torch

from otorchmizer.utils.callback import (
    Callback,
    CallbackVessel,
    CheckpointCallback,
    DiscreteSearchCallback,
)
from otorchmizer.utils.history import History
from otorchmizer.utils import logging as log_module
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e


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


class TestCallbackVessel:
    def test_creation_empty(self):
        vessel = CallbackVessel()
        assert vessel.callbacks == []

    def test_creation_with_callbacks(self):
        cb1 = Callback()
        cb2 = Callback()
        vessel = CallbackVessel([cb1, cb2])
        assert len(vessel.callbacks) == 2

    def test_invalid_callbacks_type(self):
        with pytest.raises(e.TypeError):
            CallbackVessel("not_a_list")

    def test_dispatch_events(self):
        events = []

        class TrackingCallback(Callback):
            def on_task_begin(self, opt_model):
                events.append("begin")

            def on_task_end(self, opt_model):
                events.append("end")

            def on_iteration_begin(self, iteration, opt_model):
                events.append(f"iter_{iteration}")

        vessel = CallbackVessel([TrackingCallback()])
        vessel.on_task_begin(None)
        vessel.on_iteration_begin(0, None)
        vessel.on_task_end(None)
        assert events == ["begin", "iter_0", "end"]

    def test_multiple_callbacks_dispatched(self):
        counts = {"a": 0, "b": 0}

        class CbA(Callback):
            def on_task_begin(self, opt_model):
                counts["a"] += 1

        class CbB(Callback):
            def on_task_begin(self, opt_model):
                counts["b"] += 1

        vessel = CallbackVessel([CbA(), CbB()])
        vessel.on_task_begin(None)
        assert counts["a"] == 1
        assert counts["b"] == 1


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
        from otorchmizer.core.population import Population

        lb = torch.tensor([0.0])
        ub = torch.tensor([10.0])
        pop = Population(3, 1, 1, lb, ub)
        pop.positions = torch.tensor([[[2.3]], [[7.8]], [[4.6]]])

        cb = DiscreteSearchCallback(allowed_values=[[0.0, 5.0, 10.0]])
        cb.on_evaluate_before(pop, None)

        snapped = pop.positions.squeeze().tolist()
        assert snapped[0] == 0.0   # 2.3 → 0.0
        assert snapped[1] == 10.0  # 7.8 → 10.0
        assert snapped[2] == 5.0   # 4.6 → 5.0


class TestHistory:
    def test_creation(self):
        h = History()
        assert h.save_agents is False

    def test_save_agents_flag(self):
        h = History(save_agents=True)
        assert h.save_agents is True

    def test_invalid_save_agents(self):
        with pytest.raises(e.TypeError):
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
        assert isinstance(logger, log_module.Logger)

    def test_logger_has_handlers(self):
        logger = log_module.get_logger("test_handler_logger")
        assert len(logger.handlers) >= 1

    def test_formatter(self):
        assert log_module.FORMATTER is not None

    def test_to_file(self):
        logger = log_module.get_logger("test_to_file")
        # Should not raise
        logger.to_file("Test message to file")


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


class TestExceptions:
    def test_error(self):
        with pytest.raises(e.Error):
            raise e.Error("TestError", "test message")

    def test_argument_error(self):
        with pytest.raises(e.ArgumentError):
            raise e.ArgumentError("wrong args")

    def test_build_error(self):
        with pytest.raises(e.BuildError):
            raise e.BuildError("not built")

    def test_size_error(self):
        with pytest.raises(e.SizeError):
            raise e.SizeError("wrong size")

    def test_type_error(self):
        with pytest.raises(e.TypeError):
            raise e.TypeError("wrong type")

    def test_value_error(self):
        with pytest.raises(e.ValueError):
            raise e.ValueError("wrong value")

    def test_error_message_format(self):
        try:
            raise e.TypeError("bad type")
        except e.TypeError as ex:
            assert "TypeError" in str(ex)
            assert "bad type" in str(ex)
