"""Base class for molecule operation blocks."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from cmxflow.parameter import (
    Categorical,
    Continuous,
    Integer,
)
from cmxflow.utils import text


class BlockBase(ABC):
    """Base class with shared functionality for all block types.

    Attributes:
        name: Human-readable name for the block.
        params: Dictionary of mutable parameters.
        input_files: List of required input file names.
    """

    def __init__(
        self,
        name: str | None = None,
        input_files: list[str] | None = None,
        input_text: list[str] | None = None,
    ) -> None:
        """Initialize the block.

        Args:
            name: Optional name for the block. Defaults to class name.
            input_files: Optional files that will be surfaced as required
              input at run time.
            input_text: Optional text that will be surfaces as required
              input at run time.
        """
        self.name = name or self.__class__.__name__
        if input_files is None:
            input_files = []
        if input_text is None:
            input_text = []
        self.input_files: dict[str, Path] = {key: Path(".") for key in input_files}
        self.input_text: dict[str, str] = {key: "" for key in input_text}
        self.params: dict[str, Continuous | Categorical | Integer] = {}

    def set_inputs(self, **config: str) -> None:
        """Set inputs if matching required files or text.

        Args:
            **config: Keyword arguments mapping input names to values.
                For file inputs, values should be file paths.
                For text inputs, values should be strings.
                For params, values should match the parameter type.
        """
        for key, value in config.items():
            if key in self.input_files:
                path = Path(value)
                if path.is_file():
                    self.input_files[key] = path
            elif key in self.input_text:
                self.input_text[key] = value
            elif key in self.params:
                self.params[key].set(value)

    def get_params(self) -> dict[str, Any]:
        """Get all mutable parameters for this block.

        Returns:
            Dictionary mapping parameter names to Parameter objects.
        """
        return self.params

    def get_param(self, key: str) -> Any:
        """Get the current value of a mutable parameter.

        Args:
            key: Name of the parameter to retrieve.

        Returns:
            The current value of the parameter.

        Raises:
            KeyError: If the parameter name is not registered.
        """
        if key not in self.params:
            raise KeyError(f"{key} is not a valid parameter.")
        return self.params[key].get()

    def mutable(self, *parameters: Continuous | Integer | Categorical) -> None:
        """Register parameters as mutable for optimization.

        Args:
            *parameters: Parameter objects to register as mutable.
        """
        for parameter in parameters:
            self.params[parameter.name] = parameter

    def __repr__(self) -> str:
        block = text.generate_framed_block(self.name, self.params)
        _inputs = {}
        if self.input_files:
            _input_files: dict[str, str | Path] = dict(self.input_files)
            for key, value in _input_files.items():
                if str(value) == ".":
                    _input_files[key] = "[FILE]"
            _inputs = _input_files
        if self.input_text:
            _input_text = dict(self.input_text)
            for key, value in _input_text.items():
                if not value:
                    _input_text[key] = "[TEXT]"
            _inputs = {**_inputs, **_input_text}
        if _inputs:
            inputs = text.generate_framed_block("RequiredInput", _inputs)
            block = text.left_merge_framed_block(block, inputs)
        return block

    @abstractmethod
    def __call__(self, *arg: Any) -> Any: ...

    def reset_cache(self) -> None:
        """Reset any cached state for a new optimization iteration.

        Called at the start of each optimization trial to ensure blocks
        don't retain stale cached data. Override in subclasses that
        maintain internal caches.
        """
        pass


class Block(BlockBase):
    """Block that transforms items from an iterator.

    Subclasses must implement `forward` to define the transformation.
    """

    @abstractmethod
    def forward(self, arg: Any) -> Any:
        """Define a single unit of work.

        Args:
            arg: Input item to transform.

        Returns:
            Transformed item.
        """
        ...

    def __call__(self, iter: Iterator[Any]) -> Iterator[Any]:
        """Execute the block on an iterator of items.

        Args:
            iter: Iterator of input items to process.

        Yields:
            Transformed items that pass input and output checks.
        """
        for arg in iter:
            if not self.check_input(arg):
                continue
            out = self.forward(arg)
            if self.check_output(out):
                yield out

    def check_input(self, arg: Any) -> bool:
        """Validate an input item before processing.

        Override this method to filter out invalid inputs.

        Args:
            arg: Input item to validate.

        Returns:
            True if the item should be processed, False to skip.
        """
        return True

    def check_output(self, arg: Any) -> bool:
        """Validate an output item before yielding.

        Override this method to filter out invalid outputs.

        Args:
            arg: Output item to validate.

        Returns:
            True if the item should be yielded, False to discard.
        """
        return True


class SourceBlock(BlockBase):
    """Block that produces items from a source file."""

    def __init__(self, reader: Callable[[Path], Iterator[Any]]) -> None:
        """Initialize a source block with a reader function.

        Args:
            reader: Callable that takes a Path and yields items.
        """
        self.reader = reader
        super().__init__()

    def forward(self, path: Path) -> Iterator[Any]:
        """Read items from the source file.

        Args:
            path: Path to the source file.

        Yields:
            Items read from the source file.
        """
        for item in self.reader(path):
            yield item

    def __call__(self, path: Path) -> Iterator[Any]:
        """Execute the source block.

        Args:
            path: Path to the source file.

        Returns:
            Iterator of items from the source file.
        """
        return self.forward(path)

    def __repr__(self) -> str:
        return text.generate_framed_block(self.name, {"input": "[FILE]"})


class SinkBlock(BlockBase):
    """Block that terminates a workflow by writing to a file."""

    def __init__(self, writer: Callable[[Iterator[Any], Path], None]) -> None:
        """Initialize a sink block with a writer function.

        Args:
            writer: Callable that takes an iterator and Path to write items.
        """
        self.writer = writer
        super().__init__()

    def forward(self, iter: Iterator[Any], path: Path) -> None:
        """Write items to the destination file.

        Args:
            iter: Iterator of items to write.
            path: Path to the destination file.
        """
        self.writer(iter, path)

    def __call__(self, iter: Iterator[Any], path: Path) -> None:
        """Execute the sink block.

        Args:
            iter: Iterator of items to write.
            path: Path to the destination file.
        """
        self.forward(iter, path)

    def __repr__(self) -> str:
        return text.generate_framed_block(self.name, {"output": "[FILE]"})


class ScoreBlock(BlockBase):
    """Block with state-dependent behavior for optimization vs normal execution.

    During optimization (when uid is provided), computes a score via objective().
    During normal execution, passes items through via forward().

    Subclasses must implement objective() for optimization scoring.
    Optionally override forward() to transform items during normal execution.
    """

    def __init__(
        self,
        name: str | None = None,
        input_files: list[str] | None = None,
        input_text: list[str] | None = None,
    ) -> None:
        """Initialize the score block.

        Args:
            name: Optional name for the block.
            input_files: Optional files required at run time.
            input_text: Optional text required at run time.
        """
        super().__init__(name=name, input_files=input_files, input_text=input_text)
        self._cache: dict[Any, Any] = {}
        self._uid: None | tuple[str, ...] = None

    def _set_score_properties(self, *args: Any) -> None:
        pass

    @abstractmethod
    def objective(self, iter: Iterator[Any]) -> float:
        """Compute the optimization objective score.

        Called only during optimization. Must be implemented by subclasses.

        Args:
            iter: Iterator of items to score.

        Returns:
            Score value (higher is better for maximize, lower for minimize).
        """
        ...

    def forward(self, item: Any) -> Any:
        """Transform a single item during normal (non-optimization) execution.

        Default implementation passes items through unchanged.
        Override to filter or transform items.

        Args:
            item: Input item.

        Returns:
            Transformed item, or None to filter it out.
        """
        return item

    def __call__(
        self, iter: Iterator[Any], uid: tuple[str, ...] | None = None
    ) -> float | Iterator[Any]:
        """Execute the score block.

        Args:
            iter: Iterator of input items.
            uid: Unique identifier for caching (present during optimization).

        Returns:
            If uid is provided (optimization mode): float score.
            If uid is None (normal mode): Iterator of transformed items.
        """
        if uid is not None:
            self._uid = uid
            return self.objective(iter)
        else:
            return self._forward_iter(iter)

    def _forward_iter(self, iter: Iterator[Any]) -> Iterator[Any]:
        """Apply forward() to each item in the iterator.

        Args:
            iter: Iterator of input items.

        Yields:
            Transformed items (items where forward returns None are filtered).
        """
        for item in iter:
            result = self.forward(item)
            if result is not None:
                yield result
