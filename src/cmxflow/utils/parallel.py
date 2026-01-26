"""Parallel execution utilities for Block operations."""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, TypeVar

if TYPE_CHECKING:
    from cmxflow.block import Block

from cmxflow.cmxmol import Mol

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Block")


class _SkipSentinel:
    """Sentinel value indicating an item should be skipped."""

    pass


_SKIP = _SkipSentinel()

# Worker process state (module-level for pickling)
_worker_block: Any = None
_worker_error_handling: str = "skip"


def _init_worker(factory: Callable[[], Any], error_handling: str) -> None:
    """Initialize worker process with a block instance.

    Args:
        factory: Callable that creates a configured block instance.
        error_handling: How to handle errors ("raise", "skip", "log").
    """
    global _worker_block, _worker_error_handling
    _worker_block = factory()
    _worker_error_handling = error_handling


def _worker_forward(item: Any) -> Any:
    """Process a single item in a worker process with error handling.

    Args:
        item: Input item to process.

    Returns:
        Processed item or _SKIP sentinel if item should be filtered.

    Raises:
        Exception: Re-raises if error_handling is "raise".
    """
    global _worker_block, _worker_error_handling

    try:
        if not _worker_block.check_input(item):
            return _SKIP
        result = _worker_block.forward(item)
        if not _worker_block.check_output(result):
            return _SKIP

        return result
    except Exception as e:
        if _worker_error_handling == "raise":
            raise
        if _worker_error_handling == "log":
            logger.warning(f"Error processing item: {e}")
        return _SKIP


class _BlockFactory:
    """Picklable factory for recreating block instances in worker processes.

    This class captures the block's class and state so it can be pickled
    and sent to worker processes, where it recreates configured block instances.

    Attributes:
        _cls: The block class to instantiate.
        _params: Captured parameter values.
        _input_files: Captured input file paths.
        _input_text: Captured input text values.
    """

    def __init__(self, block: Block) -> None:
        """Capture block class and current state.

        Args:
            block: Block instance to capture state from.
        """
        self._cls = block.__class__
        self._params = {k: v.get() for k, v in block.params.items()}
        self._input_files: dict[str, Path | str] = dict(block.input_files)
        self._input_text: dict[str, str] = dict(block.input_text)

    def __call__(self) -> Block:
        """Create a configured block instance.

        Returns:
            New block instance with captured state applied.
        """
        b = self._cls()
        for k, v in self._params.items():
            if k in b.params:
                b.params[k].set(v)
        b.input_files.update(self._input_files)
        b.input_text.update(self._input_text)
        return b


def _create_block_factory(block: Block) -> _BlockFactory:
    """Create a picklable factory for a block.

    Args:
        block: Block instance to capture state from.

    Returns:
        Picklable factory that creates configured block instances.
    """
    return _BlockFactory(block)


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes:
        max_workers: Maximum number of worker processes.
            Defaults to CPU count.
        chunk_size: Number of items per worker task for ordered mode.
        ordered: Whether to preserve input order in output.
        error_handling: How to handle errors ("raise", "skip", "log").
    """

    max_workers: int | None = None
    chunk_size: int = 1
    ordered: bool = True
    error_handling: Literal["raise", "skip", "log"] = "skip"


def _parallel_call_ordered(
    items: Iterator[Any],
    executor: ProcessPoolExecutor,
    config: ParallelConfig,
) -> Iterator[Any]:
    """Execute block in parallel with ordered output.

    Args:
        items: Input iterator.
        executor: ProcessPoolExecutor instance.
        config: Parallel configuration.

    Yields:
        Processed items in input order, skipping filtered items.
    """
    results = executor.map(_worker_forward, items, chunksize=config.chunk_size)
    for result in results:
        if isinstance(result, _SkipSentinel):
            continue
        if isinstance(result, Mol):
            result.restore_properties()
        yield result


def _parallel_call_unordered(
    items: Iterator[Any],
    executor: ProcessPoolExecutor,
    config: ParallelConfig,
) -> Iterator[Any]:
    """Execute block in parallel with unordered output.

    Args:
        items: Input iterator.
        executor: ProcessPoolExecutor instance.
        config: Parallel configuration.

    Yields:
        Processed items as they complete, skipping filtered items.
    """
    max_workers = config.max_workers or os.cpu_count() or 4

    # Submit in batches to avoid buffering entire iterator
    pending: set[Future[Any]] = set()
    batch_size = max_workers * 2

    for item in items:
        if len(pending) >= batch_size:
            # Wait for at least one to complete
            done, pending = _wait_for_any(pending)
            for future in done:
                result = future.result()
                if isinstance(result, _SkipSentinel):
                    continue
                if isinstance(result, Mol):
                    result.restore_properties()
                yield result

        pending.add(executor.submit(_worker_forward, item))

    # Process remaining futures
    for future in as_completed(pending):
        result = future.result()
        if not isinstance(result, _SkipSentinel):
            yield result


def _wait_for_any(
    futures: set[Future[Any]],
) -> tuple[set[Future[Any]], set[Future[Any]]]:
    """Wait for at least one future to complete.

    Args:
        futures: Set of pending futures.

    Returns:
        Tuple of (completed futures, pending futures).
    """
    done: set[Future[Any]] = set()
    for future in as_completed(futures):
        done.add(future)
        break
    pending = futures - done
    return done, pending


class ParallelBlock:
    """Wrapper that executes a block's forward method in parallel.

    This wrapper is necessary because Python looks up special methods like
    __call__ on the class, not the instance. Assigning __call__ to an instance
    doesn't work for method dispatch.

    Attributes:
        _block: The wrapped block instance.
        _config: Parallel execution configuration.
    """

    def __init__(self, block: Block, config: ParallelConfig) -> None:
        """Initialize the parallel wrapper.

        Args:
            block: Block instance to wrap.
            config: Parallel execution configuration.
        """
        self._block = block
        self._block.name = f"Parallel{self._block.name}"
        self._config = config

    def __call__(self, items: Iterator[Any]) -> Iterator[Any]:
        """Execute the wrapped block in parallel.

        Args:
            items: Input iterator of items to process.

        Yields:
            Processed items from the block's forward method.
        """
        factory = _create_block_factory(self._block)
        executor = ProcessPoolExecutor(
            max_workers=self._config.max_workers,
            initializer=_init_worker,
            initargs=(factory, self._config.error_handling),
        )

        try:
            if self._config.ordered:
                yield from _parallel_call_ordered(items, executor, self._config)
            else:
                yield from _parallel_call_unordered(items, executor, self._config)
        finally:
            executor.shutdown(wait=True)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped block.

        Args:
            name: Attribute name to look up.

        Returns:
            The attribute value from the wrapped block.
        """
        return getattr(self._block, name)

    def __repr__(self) -> str:
        """Return a string representation of the parallel wrapper.

        Returns:
            String representation showing the wrapped block.
        """
        return self._block.__repr__()


def parallel(
    max_workers: int | None = None,
    chunk_size: int = 1,
    ordered: bool = True,
    error_handling: Literal["raise", "skip", "log"] = "skip",
) -> Callable[[type[T]], type[T]]:
    """Class decorator to parallelize Block.__call__.

    Wraps a Block subclass to execute its forward method in parallel
    across multiple processes using ProcessPoolExecutor.

    Args:
        max_workers: Maximum number of worker processes.
            Defaults to os.cpu_count().
        chunk_size: Number of items per worker task. Higher values reduce
            IPC overhead but may cause uneven work distribution.
        ordered: If True (default), preserve input order in output.
            If False, yield results as they complete.
        error_handling: How to handle errors in worker processes:
            - "raise": Re-raise exceptions (terminates iteration)
            - "skip": Silently skip failed items (default)
            - "log": Log warnings and skip failed items

    Returns:
        Class decorator that creates a subclass with parallel __call__.

    Example:
        @parallel(max_workers=4)
        class ParallelAlign(MoleculeAlignBlock):
            pass

        block = ParallelAlign()
        block.input_files["query"] = "refs.sdf"
        results = list(block(molecule_iterator))
    """
    config = ParallelConfig(
        max_workers=max_workers,
        chunk_size=chunk_size,
        ordered=ordered,
        error_handling=error_handling,
    )

    def decorator(cls: type[T]) -> type[T]:
        def parallel_call(self: Any, items: Iterator[Any]) -> Iterator[Any]:
            """Execute the block's forward method in parallel."""
            factory = _create_block_factory(self)
            executor = ProcessPoolExecutor(
                max_workers=config.max_workers,
                initializer=_init_worker,
                initargs=(factory, config.error_handling),
            )

            try:
                if config.ordered:
                    yield from _parallel_call_ordered(items, executor, config)
                else:
                    yield from _parallel_call_unordered(items, executor, config)
            finally:
                executor.shutdown(wait=True)

        # Create a new class that inherits from the original
        # This ensures the class is picklable (unlike a wrapper function)
        new_cls = type(cls.__name__, (cls,), {"__call__": parallel_call})
        new_cls.__module__ = cls.__module__
        new_cls.__qualname__ = cls.__qualname__
        new_cls.__doc__ = cls.__doc__

        return new_cls  # type: ignore[return-value]

    return decorator


def make_parallel(
    block: Block,
    max_workers: int | None = None,
    chunk_size: int = 1,
    ordered: bool = True,
    error_handling: Literal["raise", "skip", "log"] = "skip",
) -> ParallelBlock:
    """Create a parallel-enabled wrapper around an existing block instance.

    Wraps the block to execute its forward method in parallel across
    multiple processes using ProcessPoolExecutor.

    Args:
        block: Block instance to wrap.
        max_workers: Maximum number of worker processes.
            Defaults to os.cpu_count().
        chunk_size: Number of items per worker task.
        ordered: If True (default), preserve input order in output.
        error_handling: How to handle errors ("raise", "skip", "log").

    Returns:
        ParallelBlock wrapper with parallel execution enabled.

    Example:
        block = MoleculeAlignBlock()
        block.input_files["query"] = "refs.sdf"
        parallel_block = make_parallel(block, max_workers=4)
        results = list(parallel_block(molecule_iterator))
    """
    config = ParallelConfig(
        max_workers=max_workers,
        chunk_size=chunk_size,
        ordered=ordered,
        error_handling=error_handling,
    )

    return ParallelBlock(block, config)
