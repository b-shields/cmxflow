"""Base class for molecule operation blocks."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from cmxflow.parameter import Parameter
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
        input_files: dict[str, Path] | None = None,
    ) -> None:
        """Initialize the block.

        Args:
            name: Optional name for the block. Defaults to class name.
            input_files: Optional files that must be read at instantiation.
        """
        self.name = name or self.__class__.__name__
        self.input_files: dict[str, Path] | None = input_files
        self.params: dict[str, Parameter] = {}

    def get_params(self) -> dict[str, Any]:
        return self.params

    def mutable(self, *parameters: Parameter) -> None:
        for parameter in parameters:
            self.params[parameter.name] = parameter

    def __repr__(self) -> str:
        block = text.generate_framed_block(self.name, self.params)
        if self.input_files is not None:
            inputs = text.generate_framed_block("FileInput", self.input_files)
            block = text.left_merge_framed_block(block, inputs)
        return block

    @abstractmethod
    def __call__(self, *arg: Any) -> Any: ...


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
        for arg in iter:
            yield self.forward(arg)


class SourceBlock(BlockBase):
    """Block that produces items from a source file."""

    def __init__(self, reader: Callable[[Path], Iterator[Any]]) -> None:
        self.reader = reader
        super().__init__()

    def forward(self, path: Path) -> Iterator[Any]:
        for item in self.reader(path):
            yield item

    def __call__(self, path: Path) -> Iterator[Any]:
        return self.forward(path)


class SinkBlock(BlockBase):
    """Block that consumes items and writes to a destination."""

    def __init__(self, writer: Callable[[Iterator[Any], Path], None]) -> None:
        self.writer = writer
        super().__init__()

    def forward(self, iter: Iterator[Any], path: Path) -> None:
        self.writer(iter, path)

    def __call__(self, iter: Iterator[Any], path: Path) -> None:
        self.forward(iter, path)
