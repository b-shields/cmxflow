"""Tests for the Optuna-based Optimizer class."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from cmxflow.block import Block, ScoreBlock, SinkBlock, SourceBlock
from cmxflow.opt.optuna import Optimizer
from cmxflow.parameter import Categorical, Continuous, Integer
from cmxflow.workflow import Workflow


def dummy_reader(_path: Path) -> Iterator[dict[str, Any]]:
    """Dummy reader that yields test data."""
    for i in range(10):
        yield {"id": i, "value": i * 0.1}


def dummy_writer(iter: Iterator[Any], _path: Path) -> None:
    """Dummy writer that consumes the iterator."""
    for _ in iter:
        pass


class TransformBlock(Block):
    """Block that applies a parameterized transformation."""

    def __init__(self) -> None:
        """Initialize with optimizable parameters."""
        super().__init__()
        self.mutable(
            Continuous("scale", default=1.0, low=0.1, high=10.0),
            Integer("offset", default=0, low=-5, high=5),
        )

    def forward(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply transformation to the item."""
        scale = self.params["scale"].get()
        offset = self.params["offset"].get()
        return {
            "id": item["id"],
            "value": item["value"] * scale + offset,
            "target": item["id"] % 2,
        }


class CategoricalBlock(Block):
    """Block with a categorical parameter."""

    def __init__(self) -> None:
        """Initialize with a categorical parameter."""
        super().__init__()
        self.mutable(
            Categorical("mode", default="linear", choices=["linear", "quadratic"]),
        )

    def forward(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply mode-dependent transformation."""
        mode = self.params["mode"].get()
        value = item["value"]
        if mode == "quadratic":
            value = value**2
        return {**item, "value": value, "target": item["id"] % 2}


class NoParamBlock(Block):
    """Block without any parameters."""

    def forward(self, item: dict[str, Any]) -> dict[str, Any]:
        """Pass through unchanged."""
        return item


class DummyScoreBlock(ScoreBlock):
    """Concrete ScoreBlock for testing that computes correlation."""

    def objective(self, iter: Iterator[Any]) -> float:
        """Compute correlation between value and target columns."""
        data = list(iter)
        if not data:
            return 0.0
        df = pd.DataFrame(data)
        if "value" not in df.columns or "target" not in df.columns:
            return 0.0
        pred = df["value"].to_numpy()
        target = df["target"].to_numpy()
        if len(pred) < 2:
            return 0.0
        corr = np.corrcoef(pred, target)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0


class NoParamScoreBlock(ScoreBlock):
    """ScoreBlock without any parameters for testing."""

    def objective(self, iter: Iterator[Any]) -> float:
        """Return zero score."""
        list(iter)  # Consume iterator
        return 0.0


class TestOptimizerInit:
    """Tests for Optimizer initialization."""

    def test_valid_workflow(self) -> None:
        """Test initialization with a valid workflow."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        assert optimizer.workflow is workflow
        assert optimizer.input_path == Path("test.csv")

    def test_workflow_not_ending_with_score_block(self) -> None:
        """Test that workflow must end with ScoreBlock."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            SinkBlock(dummy_writer),
        )

        with pytest.raises(ValueError, match="must end with a ScoreBlock"):
            Optimizer(workflow, input_path="test.csv")

    def test_workflow_without_params(self) -> None:
        """Test that workflow must have optimizable parameters."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            NoParamBlock(),
            NoParamScoreBlock(),
        )

        with pytest.raises(ValueError, match="no optimizable parameters"):
            Optimizer(workflow, input_path="test.csv")

    def test_path_conversion(self) -> None:
        """Test that string paths are converted to Path objects."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        assert isinstance(optimizer.input_path, Path)


class TestOptimizerOptimize:
    """Tests for the optimize method."""

    def test_optimize_runs(self) -> None:
        """Test that optimization runs successfully."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        study = optimizer.optimize(n_trials=3, show_progress_bar=False)

        assert study is not None
        assert len(study.trials) == 3

    def test_optimize_direction(self) -> None:
        """Test optimization direction setting."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        study = optimizer.optimize(
            n_trials=2, direction="minimize", show_progress_bar=False
        )

        assert study.direction.name == "MINIMIZE"

    def test_categorical_params(self) -> None:
        """Test optimization with categorical parameters."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            CategoricalBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        study = optimizer.optimize(n_trials=3, show_progress_bar=False)

        assert "mode" in study.best_params
        assert study.best_params["mode"] in ["linear", "quadratic"]


class TestOptimizerProperties:
    """Tests for Optimizer properties."""

    def test_best_params_before_optimize(self) -> None:
        """Test that best_params raises before optimization."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")

        with pytest.raises(RuntimeError, match="Call optimize"):
            _ = optimizer.best_params

    def test_best_score_before_optimize(self) -> None:
        """Test that best_score raises before optimization."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")

        with pytest.raises(RuntimeError, match="Call optimize"):
            _ = optimizer.best_score

    def test_study_before_optimize(self) -> None:
        """Test that study raises before optimization."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")

        with pytest.raises(RuntimeError, match="Call optimize"):
            _ = optimizer.study

    def test_best_params_after_optimize(self) -> None:
        """Test best_params returns values after optimization."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        optimizer.optimize(n_trials=2, show_progress_bar=False)

        params = optimizer.best_params
        assert "scale" in params
        assert "offset" in params

    def test_best_score_after_optimize(self) -> None:
        """Test best_score returns value after optimization."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        optimizer.optimize(n_trials=2, show_progress_bar=False)

        score = optimizer.best_score
        assert isinstance(score, float)


class TestSetBestParams:
    """Tests for set_best_params method."""

    def test_set_best_params_before_optimize(self) -> None:
        """Test that set_best_params raises before optimization."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")

        with pytest.raises(RuntimeError, match="Call optimize"):
            optimizer.set_best_params()

    def test_set_best_params_updates_workflow(self) -> None:
        """Test that set_best_params updates workflow parameters."""
        workflow = Workflow()
        transform_block = TransformBlock()
        workflow.add(
            SourceBlock(dummy_reader),
            transform_block,
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")
        optimizer.optimize(n_trials=3, show_progress_bar=False)

        best_params = optimizer.best_params
        optimizer.set_best_params()

        assert transform_block.params["scale"].get() == best_params["scale"]
        assert transform_block.params["offset"].get() == best_params["offset"]


class TestSuggestParam:
    """Tests for parameter suggestion methods."""

    def test_unsupported_parameter_type(self) -> None:
        """Test that unsupported parameter types raise TypeError."""
        workflow = Workflow()
        workflow.add(
            SourceBlock(dummy_reader),
            TransformBlock(),
            DummyScoreBlock(),
        )

        optimizer = Optimizer(workflow, input_path="test.csv")

        # Create a mock parameter with unsupported type
        mock_param = MagicMock()
        mock_param.name = "test"
        mock_trial = MagicMock()

        with pytest.raises(TypeError, match="Unsupported parameter type"):
            optimizer._suggest_param(mock_trial, mock_param)
