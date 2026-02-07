"""Tests for WorkflowRegistry."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cmxflow.block import Block, SinkBlock, SourceBlock
from cmxflow.utils.serial import WorkflowRegistry
from cmxflow.workflow import Workflow


def simple_reader(path: Path) -> Iterator[int]:
    """Simple reader that yields integers."""
    for i in range(5):
        yield i


def simple_writer(iter: Iterator[Any], path: Path) -> None:
    """Simple writer that consumes the iterator."""
    list(iter)


class SquareBlock(Block):
    """Simple block that squares numbers."""

    def forward(self, x: int) -> int:
        """Square the input."""
        return x * x


def _make_workflow(name: str = "TestWorkflow") -> Workflow:
    """Create a simple valid workflow for testing."""
    wf = Workflow(name)
    wf.add(SourceBlock(simple_reader))
    wf.add(SquareBlock())
    wf.add(SinkBlock(simple_writer))
    return wf


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry."""

    def test_register_and_list(self, tmp_path: Path) -> None:
        """Test registering a workflow and listing it."""
        registry = WorkflowRegistry(tmp_path / "registry")
        wf = _make_workflow()

        registry.register("my_workflow", wf)

        df = registry.list()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == ["name", "date", "representation"]
        assert df.iloc[0]["name"] == "my_workflow"
        assert "SquareBlock" in df.iloc[0]["representation"]

    def test_register_duplicate_raises(self, tmp_path: Path) -> None:
        """Test that registering a duplicate name raises ValueError."""
        registry = WorkflowRegistry(tmp_path / "registry")
        wf = _make_workflow()

        registry.register("dup", wf)
        with pytest.raises(ValueError, match="already exists"):
            registry.register("dup", wf)

    def test_register_overwrite(self, tmp_path: Path) -> None:
        """Test overwriting an existing registration."""
        registry = WorkflowRegistry(tmp_path / "registry")
        wf1 = _make_workflow("First")
        wf2 = _make_workflow("Second")

        registry.register("overwrite_me", wf1)
        registry.register("overwrite_me", wf2, overwrite=True)

        df = registry.list()
        assert len(df) == 1
        loaded = registry.load("overwrite_me")
        assert loaded.name == "Second"

    def test_load(self, tmp_path: Path) -> None:
        """Test loading a registered workflow by name."""
        registry = WorkflowRegistry(tmp_path / "registry")
        wf = _make_workflow("LoadTest")

        registry.register("loadable", wf)
        loaded = registry.load("loadable")

        assert loaded.name == "LoadTest"
        assert len(loaded.blocks) == 3

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading a nonexistent workflow raises KeyError."""
        registry = WorkflowRegistry(tmp_path / "registry")

        with pytest.raises(KeyError, match="No workflow registered"):
            registry.load("nonexistent")

    def test_list_empty(self, tmp_path: Path) -> None:
        """Test listing an empty registry returns empty DataFrame."""
        registry = WorkflowRegistry(tmp_path / "registry")
        df = registry.list()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["name", "date", "representation"]

    def test_multiple_workflows(self, tmp_path: Path) -> None:
        """Test registering and listing multiple workflows."""
        registry = WorkflowRegistry(tmp_path / "registry")

        for i in range(3):
            wf = _make_workflow(f"Workflow_{i}")
            registry.register(f"wf_{i}", wf)

        df = registry.list()
        assert len(df) == 3
        assert set(df["name"]) == {"wf_0", "wf_1", "wf_2"}

    def test_remove(self, tmp_path: Path) -> None:
        """Test removing a registered workflow."""
        registry = WorkflowRegistry(tmp_path / "registry")
        wf = _make_workflow()

        registry.register("removable", wf)
        assert len(registry.list()) == 1

        registry.remove("removable")
        assert len(registry.list()) == 0
        assert not (tmp_path / "registry" / "removable.pkl.gz").exists()

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        """Test removing a nonexistent workflow raises KeyError."""
        registry = WorkflowRegistry(tmp_path / "registry")

        with pytest.raises(KeyError, match="No workflow registered"):
            registry.remove("ghost")
