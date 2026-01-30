"""Base class for molecule operation blocks."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from cmxflow.parameter import Categorical, Integer, Parameter
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
        self.input_files: dict[str, Path | str] = {
            key: Path(".") for key in input_files
        }
        self.input_text: dict[str, str] = {key: "" for key in input_text}
        self.params: dict[str, Parameter] = {}

    def get_params(self) -> dict[str, Any]:
        """Get all mutable parameters for this block.

        Returns:
            Dictionary mapping parameter names to Parameter objects.
        """
        return self.params

    def mutable(self, *parameters: Parameter) -> None:
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
            _input_files = dict(self.input_files)
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
        """Method called in each optimization iteration."""
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
    """Block that scores workflow outputs against a target.

    Pools iterator results into a DataFrame, computes scores, and evaluates
    against a target using a metric function. Caches results by unique ID.
    """

    def __init__(
        self,
        pooler: Callable[[Iterator[Any]], pd.DataFrame],
        metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        """Initialize with pooler, scorer, and metric functions."""
        super().__init__(input_text=["target"])
        self.pooler = pooler
        self.metric = metric
        self._cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._scaler: None | MinMaxScaler = None
        self._score_components: list[str] | None = None
        self.mutable(
            Categorical("compose", "geometric", ["arithmetic", "geometric"]),
            Integer("combinations", 1, 1, 5),
        )

    def __call__(
        self, iter: Iterator[Any], uid: tuple[str, ...]
    ) -> tuple[float, list[str] | None]:
        """Score iterator results against target, using cache if available."""
        target = self.input_text.get("target")
        if not target:
            raise ValueError("ScoreBlock requires a 'target' as input")

        # Collect cached result if possible
        if uid in self._cache:
            del iter
            df = self._cache[uid]
        else:
            df = self.pooler(iter)
            self._scaler = MinMaxScaler()
            cols = df.drop(target, axis=1).columns.values
            df[cols] = self._scaler.fit_transform(df[cols])
            self._cache[uid] = df

        # Compute scores for different combinations
        comb = self.params["combinations"].get()
        compose = self.params["compose"].get()
        possible = combinations(
            df.drop(target, axis=1).columns.values, min(df.shape[1], comb)
        )
        best_score = -np.inf
        best_cols: list[str] | None = None
        for cols in possible:
            cols = list(cols)
            if compose == "arithmetic":
                score = df[cols].mean(axis=1).to_numpy()
            elif compose == "geometric":
                score = df[cols].prod(axis=1).to_numpy() ** (1 / len(cols))
            else:
                raise ValueError(f"{compose} is not a valid compose setting")
            metric = self.metric(score, df[target].to_numpy())
            if metric > best_score:
                best_score = metric
                best_cols = cols

        self._score_components = list(best_cols) if best_cols is not None else None

        return best_score, self._score_components
