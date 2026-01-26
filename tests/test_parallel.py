"""Tests for parallel execution utilities."""

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
