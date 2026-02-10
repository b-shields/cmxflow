"""Tests for parallel execution utilities."""

import time
from typing import Any

import pytest

from cmxflow.block import Block
from cmxflow.parameter import Integer
from cmxflow.utils.parallel import ParallelConfig, make_parallel, parallel


class SquareBlock(Block):
    """Simple block that squares numbers."""

    def forward(self, x: int) -> int:
        """Square the input."""
        return x * x


class DoubleBlock(Block):
    """Block that doubles numbers with a parameter."""

    def __init__(self) -> None:
        """Initialize the block."""
        super().__init__()
        self.mutable(Integer("multiplier", default=2, low=1, high=10))

    def forward(self, x: int) -> int:
        """Double the input using the multiplier parameter."""
        return x * int(self.params["multiplier"].get())


class FilterBlock(Block):
    """Block that filters odd numbers."""

    def check_input(self, arg: Any) -> bool:
        """Only process even numbers."""
        return isinstance(arg, int) and arg % 2 == 0

    def forward(self, x: int) -> int:
        """Return the input unchanged."""
        return x


class ErrorBlock(Block):
    """Block that raises errors for specific inputs."""

    def forward(self, x: int) -> int:
        """Raise error if x is negative."""
        if x < 0:
            raise ValueError(f"Negative value: {x}")
        return x


# Module-level decorated classes (picklable for multiprocessing)
@parallel(max_workers=2)
class ParallelSquare(SquareBlock):
    """Parallel version of SquareBlock."""

    pass


@parallel(max_workers=2)
class ParallelDouble(DoubleBlock):
    """Parallel version of DoubleBlock."""

    pass


@parallel(max_workers=2, ordered=True)
class ParallelSquareOrdered(SquareBlock):
    """Parallel version of SquareBlock with ordered output."""

    pass


class TestParallelConfig:
    """Tests for ParallelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ParallelConfig()
        assert config.max_workers is None
        assert config.chunk_size == 1
        assert config.ordered is True
        assert config.error_handling == "skip"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ParallelConfig(
            max_workers=4,
            chunk_size=10,
            ordered=False,
            error_handling="raise",
        )
        assert config.max_workers == 4
        assert config.chunk_size == 10
        assert config.ordered is False
        assert config.error_handling == "raise"

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = ParallelConfig()
        with pytest.raises(AttributeError):
            config.max_workers = 8  # type: ignore[misc]


class TestParallelDecorator:
    """Tests for the @parallel decorator."""

    def test_basic_decorator(self) -> None:
        """Test basic decorator usage."""
        block = ParallelSquare()
        result = list(block(iter([1, 2, 3, 4, 5])))
        assert sorted(result) == [1, 4, 9, 16, 25]

    def test_decorator_with_parameters(self) -> None:
        """Test decorator preserves block parameters."""
        block = ParallelDouble()
        block.params["multiplier"].set(3)
        result = list(block(iter([1, 2, 3])))
        assert sorted(result) == [3, 6, 9]

    def test_decorator_ordered(self) -> None:
        """Test ordered output."""
        block = ParallelSquareOrdered()
        result = list(block(iter([1, 2, 3, 4, 5])))
        assert result == [1, 4, 9, 16, 25]


class TestMakeParallel:
    """Tests for make_parallel function."""

    def test_basic_make_parallel(self) -> None:
        """Test basic make_parallel usage."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        result = list(parallel_block(iter([1, 2, 3, 4, 5])))
        assert sorted(result) == [1, 4, 9, 16, 25]

    def test_make_parallel_preserves_params(self) -> None:
        """Test make_parallel preserves block parameters."""
        block = DoubleBlock()
        block.params["multiplier"].set(5)
        parallel_block = make_parallel(block, max_workers=2)
        result = list(parallel_block(iter([1, 2, 3])))
        assert sorted(result) == [5, 10, 15]

    def test_make_parallel_ordered(self) -> None:
        """Test ordered output."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2, ordered=True)
        result = list(parallel_block(iter([1, 2, 3, 4, 5])))
        assert result == [1, 4, 9, 16, 25]


class TestInputFiltering:
    """Tests for input/output filtering in parallel execution."""

    def test_check_input_filtering(self) -> None:
        """Test that check_input filters items."""
        block = FilterBlock()
        parallel_block = make_parallel(block, max_workers=2)
        result = list(parallel_block(iter([1, 2, 3, 4, 5, 6])))
        assert sorted(result) == [2, 4, 6]


class TestErrorHandling:
    """Tests for error handling in parallel execution."""

    def test_error_skip(self) -> None:
        """Test skip error handling mode."""
        block = ErrorBlock()
        parallel_block = make_parallel(block, max_workers=2, error_handling="skip")
        result = list(parallel_block(iter([-1, 1, -2, 2, 3])))
        assert sorted(result) == [1, 2, 3]

    def test_error_raise(self) -> None:
        """Test raise error handling mode."""
        block = ErrorBlock()
        parallel_block = make_parallel(block, max_workers=2, error_handling="raise")
        with pytest.raises(ValueError, match="Negative value"):
            list(parallel_block(iter([-1, 1, 2])))

    def test_error_log(self) -> None:
        """Test log error handling mode."""
        block = ErrorBlock()
        parallel_block = make_parallel(block, max_workers=2, error_handling="log")
        result = list(parallel_block(iter([-1, 1, -2, 2, 3])))
        assert sorted(result) == [1, 2, 3]


class TestUnorderedExecution:
    """Tests for unordered parallel execution."""

    def test_unordered(self) -> None:
        """Test unordered execution."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2, ordered=False)
        result = list(parallel_block(iter([1, 2, 3, 4, 5])))
        assert sorted(result) == [1, 4, 9, 16, 25]


class TestEmptyInput:
    """Tests for empty input handling."""

    def test_empty_iterator(self) -> None:
        """Test handling empty input."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        result = list(parallel_block(iter([])))
        assert result == []


class TestChunkSize:
    """Tests for chunk_size parameter."""

    def test_chunk_size(self) -> None:
        """Test chunk size."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2, chunk_size=5)
        result = list(parallel_block(iter(range(20))))
        assert sorted(result) == [i * i for i in range(20)]


class TestContextManager:
    """Tests for ParallelBlock context manager."""

    def test_context_manager_basic(self) -> None:
        """Test correct results using context manager."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        with parallel_block:
            result = list(parallel_block(iter([1, 2, 3, 4, 5])))
        assert sorted(result) == [1, 4, 9, 16, 25]

    def test_context_manager_multiple_calls(self) -> None:
        """Test pool reused across calls in one with block."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        with parallel_block:
            executor_id = id(parallel_block._executor)
            result1 = list(parallel_block(iter([1, 2, 3])))
            assert id(parallel_block._executor) == executor_id
            result2 = list(parallel_block(iter([4, 5, 6])))
            assert id(parallel_block._executor) == executor_id
        assert sorted(result1) == [1, 4, 9]
        assert sorted(result2) == [16, 25, 36]

    def test_context_manager_executor_cleanup(self) -> None:
        """Test _executor is None after exit."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        with parallel_block:
            assert parallel_block._executor is not None
        assert parallel_block._executor is None

    def test_standalone_no_executor_leak(self) -> None:
        """Test no _executor after standalone call."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        list(parallel_block(iter([1, 2, 3])))
        assert parallel_block._executor is None

    def test_context_manager_exception_cleanup(self) -> None:
        """Test cleanup on exception inside context manager."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        with pytest.raises(ValueError, match="test error"):
            with parallel_block:
                assert parallel_block._executor is not None
                raise ValueError("test error")
        assert parallel_block._executor is None

    def test_context_manager_not_reentrant(self) -> None:
        """Test nested with raises RuntimeError."""
        block = SquareBlock()
        parallel_block = make_parallel(block, max_workers=2)
        with parallel_block:
            with pytest.raises(RuntimeError, match="not reentrant"):
                parallel_block.__enter__()


class TestUnorderedDrainLoop:
    """Tests for unordered drain loop processing all items."""

    def test_all_items_processed(self) -> None:
        """Validates all items processed with enough items to exercise drain."""
        block = SquareBlock()
        # Use enough items to exceed batch_size (max_workers * 2 = 4)
        items = list(range(20))
        parallel_block = make_parallel(block, max_workers=2, ordered=False)
        result = list(parallel_block(iter(items)))
        assert sorted(result) == [i * i for i in items]


class SlowBlock(Block):
    """Block that sleeps for negative inputs to simulate a hang."""

    def forward(self, x: int) -> int:
        """Sleep if x is negative, otherwise return x squared."""
        if x < 0:
            time.sleep(abs(x))
        return x * x


class TestWorkerTimeout:
    """Tests for worker_timeout behavior."""

    def test_timeout_skips_slow_items_ordered(self) -> None:
        """Test that slow items are skipped in ordered mode."""
        block = SlowBlock()
        pb = make_parallel(block, max_workers=2, ordered=True)
        # 2s timeout comfortably exceeds pool startup but catches the
        # -30 item (which would sleep 30s)
        pb._config = ParallelConfig(max_workers=2, ordered=True, worker_timeout=1.5)
        result = list(pb(iter([1, -30, 2])))
        assert result == [1, 4]

    def test_timeout_skips_slow_items_unordered(self) -> None:
        """Test that slow items are skipped in unordered mode."""
        block = SlowBlock()
        pb = make_parallel(block, max_workers=2, ordered=False)
        pb._config = ParallelConfig(max_workers=2, ordered=False, worker_timeout=1.5)
        result = list(pb(iter([1, -30, 2])))
        assert sorted(result) == [1, 4]

    def test_timeout_raises_when_configured(self) -> None:
        """Test that TimeoutError is raised with error_handling='raise'."""
        block = SlowBlock()
        pb = make_parallel(block, max_workers=2, error_handling="raise")
        pb._config = ParallelConfig(
            max_workers=2,
            ordered=True,
            error_handling="raise",
            worker_timeout=1.5,
        )
        with pytest.raises(TimeoutError):
            list(pb(iter([1, -30, 2])))

    def test_no_timeout_when_disabled(self) -> None:
        """Test that worker_timeout=0 disables the timeout."""
        block = SlowBlock()
        pb = make_parallel(block, max_workers=2)
        pb._config = ParallelConfig(max_workers=2, ordered=True, worker_timeout=0)
        result = list(pb(iter([1, 2])))
        assert result == [1, 4]

    def test_default_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CMXFLOW_WORKER_TIMEOUT env var is respected."""
        from cmxflow.utils.parallel import _default_worker_timeout

        monkeypatch.setenv("CMXFLOW_WORKER_TIMEOUT", "45.5")
        assert _default_worker_timeout() == 45.5

    def test_invalid_env_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invalid env var falls back to default."""
        from cmxflow.utils.parallel import _default_worker_timeout

        monkeypatch.setenv("CMXFLOW_WORKER_TIMEOUT", "not_a_number")
        assert _default_worker_timeout() == 30.0
