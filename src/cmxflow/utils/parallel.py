"""Parallel execution utilities for Block operations."""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, TypeVar

import threadpoolctl

if TYPE_CHECKING:
    from cmxflow.block import Block

from cmxflow.cmxmol import Mol

logger = logging.getLogger(__name__)

# Held for the worker's lifetime so the runtime thread limit is not undone by
# garbage collection (the limiter restores prior counts when finalized).
_worker_threadpool_limiter: Any = None

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
    global _worker_block, _worker_error_handling, _worker_threadpool_limiter
    _worker_block = factory()
    _worker_error_handling = error_handling
    # Pin BLAS/OpenMP to one thread per worker. This block parallelizes at the
    # process level (one item per worker), so nested numerical threads only
    # oversubscribe cores -- e.g. 12 workers x 12 BLAS threads = 144 spin-waiting
    # threads on 12 cores, which erased the parallel speedup (~7.6x slower).
    # Env vars can't fix this (OpenBLAS latches its pool size at import, before
    # this runs); threadpoolctl resizes the live pool, so it must run after the
    # factory has imported the numerical stack.
    _worker_threadpool_limiter = threadpoolctl.threadpool_limits(limits=1)


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
        self._input_files: dict[str, Path] = dict(block.input_files)
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


def _get_mp_context(
    start_method: str | None,
) -> multiprocessing.context.BaseContext | None:
    """Get a multiprocessing context for the given start method.

    Args:
        start_method: Start method name (e.g. "forkserver", "spawn", "fork"),
            or None to use the platform default.

    Returns:
        Multiprocessing context, or None for platform default.
    """
    if start_method is None:
        return None
    return multiprocessing.get_context(start_method)


def _default_start_method() -> str | None:
    """Return the default start method for the current platform.

    Returns:
        "forkserver" on Unix-like systems, None (platform default) on Windows.
    """
    if sys.platform == "win32":
        return None
    return "forkserver"


_DEFAULT_WORKER_TIMEOUT: float = 120.0


def _default_worker_timeout() -> float:
    """Return the default worker timeout from env or built-in default.

    The ``CMXFLOW_WORKER_TIMEOUT`` environment variable overrides the
    built-in default of 120 seconds. Set to ``0`` to disable the timeout.

    Returns:
        Timeout in seconds, or 0.0 to disable.
    """
    raw = os.environ.get("CMXFLOW_WORKER_TIMEOUT")
    if raw is None:
        return _DEFAULT_WORKER_TIMEOUT
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid CMXFLOW_WORKER_TIMEOUT=%r, using default %.0fs",
            raw,
            _DEFAULT_WORKER_TIMEOUT,
        )
        return _DEFAULT_WORKER_TIMEOUT


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes:
        max_workers: Maximum number of worker processes.
            Defaults to CPU count.
        chunk_size: Number of items per worker task for ordered mode.
        ordered: Whether to preserve input order in output.
        error_handling: How to handle errors ("raise", "skip", "log").
        start_method: Multiprocessing start method. Defaults to "forkserver"
            on Unix, None (platform default) on Windows.
        worker_timeout: Seconds to wait for a single worker result before
            treating it as an error. Defaults to 120s (overridable via
            ``CMXFLOW_WORKER_TIMEOUT`` env var). Set to 0 to disable.
    """

    max_workers: int | None = None
    chunk_size: int = 1
    ordered: bool = True
    error_handling: Literal["raise", "skip", "log"] = "skip"
    start_method: str | None = field(default_factory=_default_start_method)
    worker_timeout: float = field(default_factory=_default_worker_timeout)


def _get_timeout(config: ParallelConfig) -> float | None:
    """Convert worker_timeout config to a value for future.result().

    Args:
        config: Parallel configuration.

    Returns:
        Timeout in seconds, or None to wait indefinitely.
    """
    if config.worker_timeout <= 0:
        return None
    return config.worker_timeout


def _handle_future_result(
    future: Future[Any],
    config: ParallelConfig,
    timeout: float | None,
) -> Any:
    """Get a future's result, applying timeout and error handling.

    Args:
        future: The future to resolve.
        config: Parallel configuration for error handling.
        timeout: Seconds to wait, or None for no limit.

    Returns:
        The future result, or _SKIP if the item timed out and
        error_handling is "skip" or "log".

    Raises:
        TimeoutError: If timeout exceeded and error_handling is "raise".
    """
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        if config.error_handling == "raise":
            raise
        if config.error_handling == "log":
            logger.warning("Worker timed out after %.1fs", config.worker_timeout)
        return _SKIP


def _yield_result(result: Any) -> Any:
    """Restore properties on Mol objects before yielding.

    Args:
        result: Processed result from a worker.

    Returns:
        The result, with properties restored if it's a Mol.
    """
    if isinstance(result, Mol):
        result.restore_properties()
    return result


def _parallel_call_ordered(
    items: Iterator[Any],
    executor: ProcessPoolExecutor,
    config: ParallelConfig,
) -> Iterator[Any]:
    """Execute block in parallel with ordered output using a bounded window.

    Submits futures lazily, maintaining at most ``max_workers * 2`` in-flight
    futures to bound memory usage to O(max_workers) instead of O(n_items).

    Args:
        items: Input iterator.
        executor: ProcessPoolExecutor instance.
        config: Parallel configuration.

    Yields:
        Processed items in input order, skipping filtered items.
    """
    max_workers = config.max_workers or os.cpu_count() or 4
    window_size = max_workers * 2
    window: deque[Future[Any]] = deque()
    timeout = _get_timeout(config)

    for item in items:
        if len(window) >= window_size:
            result = _handle_future_result(window.popleft(), config, timeout)
            if not isinstance(result, _SkipSentinel):
                yield _yield_result(result)
        window.append(executor.submit(_worker_forward, item))

    # Drain remaining futures front-to-back (preserves order)
    while window:
        result = _handle_future_result(window.popleft(), config, timeout)
        if not isinstance(result, _SkipSentinel):
            yield _yield_result(result)


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
    timeout = _get_timeout(config)

    # Submit in batches to avoid buffering entire iterator
    pending: set[Future[Any]] = set()
    batch_size = max_workers * 2

    for item in items:
        if len(pending) >= batch_size:
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                # All workers hung — cancel and stop submitting
                for f in pending:
                    f.cancel()
                pending = set()
                if config.error_handling == "raise":
                    raise TimeoutError(
                        f"Workers timed out after {config.worker_timeout}s"
                    )
                if config.error_handling == "log":
                    logger.warning(
                        "Workers timed out after %.1fs", config.worker_timeout
                    )
                break
            for future in done:
                result = future.result()
                if isinstance(result, _SkipSentinel):
                    continue
                yield _yield_result(result)

        pending.add(executor.submit(_worker_forward, item))

    # Drain remaining futures
    while pending:
        done, not_done = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        if not done:
            for f in not_done:
                f.cancel()
            if config.error_handling == "raise":
                raise TimeoutError(f"Workers timed out after {config.worker_timeout}s")
            if config.error_handling == "log":
                logger.warning("Workers timed out after %.1fs", config.worker_timeout)
            break
        pending = not_done
        for future in done:
            result = future.result()
            if not isinstance(result, _SkipSentinel):
                yield _yield_result(result)


def _run_parallel(
    block: Block,
    config: ParallelConfig,
    items: Iterator[Any],
    executor: ProcessPoolExecutor | None = None,
) -> Iterator[Any]:
    """Execute a block in parallel, optionally using a provided executor.

    When ``executor`` is provided (context manager mode), delegates to the
    ordered/unordered helper without managing the executor lifecycle.

    When ``executor`` is None, creates a new executor with the configured
    start method and shuts it down after iteration completes.

    Args:
        block: Block whose forward method to parallelize.
        config: Parallel execution configuration.
        items: Input iterator of items to process.
        executor: Optional pre-existing executor to reuse.

    Yields:
        Processed items from the block's forward method.
    """
    factory = _create_block_factory(block)

    if executor is not None:
        # Reinitialize workers with current block state
        executor.submit(_init_worker, factory, config.error_handling).result()
        if config.ordered:
            yield from _parallel_call_ordered(items, executor, config)
        else:
            yield from _parallel_call_unordered(items, executor, config)
        return

    mp_context = _get_mp_context(config.start_method)
    owned_executor = ProcessPoolExecutor(
        max_workers=config.max_workers,
        initializer=_init_worker,
        initargs=(factory, config.error_handling),
        mp_context=mp_context,
    )

    try:
        if config.ordered:
            yield from _parallel_call_ordered(items, owned_executor, config)
        else:
            yield from _parallel_call_unordered(items, owned_executor, config)
    finally:
        owned_executor.shutdown(wait=False, cancel_futures=True)


class ParallelBlock:
    """Wrapper that executes a block's forward method in parallel.

    Supports use as a context manager to reuse the process pool across
    multiple calls::

        pb = make_parallel(block, max_workers=4)
        with pb:
            result1 = list(pb(iter1))
            result2 = list(pb(iter2))

    Attributes:
        _block: The wrapped block instance.
        _config: Parallel execution configuration.
        _executor: Executor when used as a context manager, else None.
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
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> ParallelBlock:
        """Enter the context manager, creating a reusable process pool.

        Returns:
            Self with an active executor.

        Raises:
            RuntimeError: If already inside a ``with`` block (non-reentrant).
        """
        if self._executor is not None:
            raise RuntimeError("ParallelBlock context manager is not reentrant")
        mp_context = _get_mp_context(self._config.start_method)
        factory = _create_block_factory(self._block)
        self._executor = ProcessPoolExecutor(
            max_workers=self._config.max_workers,
            initializer=_init_worker,
            initargs=(factory, self._config.error_handling),
            mp_context=mp_context,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context manager, shutting down the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __call__(self, items: Iterator[Any]) -> Iterator[Any]:
        """Execute the wrapped block in parallel.

        Args:
            items: Input iterator of items to process.

        Yields:
            Processed items from the block's forward method.
        """
        yield from _run_parallel(self._block, self._config, items, self._executor)

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
        ```python
        @parallel(max_workers=4)
        class ParallelAlign(MoleculeAlignBlock):
            pass

        block = ParallelAlign()
        block.input_files["query"] = "refs.sdf"
        results = list(block(molecule_iterator))
        ```
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
            yield from _run_parallel(self, config, items)

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
        ```python
        block = MoleculeAlignBlock()
        block.input_files["query"] = "refs.sdf"
        parallel_block = make_parallel(block, max_workers=4)
        results = list(parallel_block(molecule_iterator))
        ```
    """
    config = ParallelConfig(
        max_workers=max_workers,
        chunk_size=chunk_size,
        ordered=ordered,
        error_handling=error_handling,
    )

    return ParallelBlock(block, config)
