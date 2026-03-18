"""Tests for workflow save and load functionality."""

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cmxflow.block import Block, SinkBlock, SourceBlock
from cmxflow.utils.serial import load_workflow, save_workflow
from cmxflow.workflow import Workflow, WorkflowValidationError


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


class TestSaveWorkflow:
    """Tests for save_workflow function."""

    def test_save_valid_workflow(self, tmp_path: Path) -> None:
        """Test saving a valid workflow."""
        workflow = Workflow("TestWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        save_path = tmp_path / "workflow.pkl"
        save_workflow(workflow, save_path)

        assert save_path.exists()

    def test_save_valid_workflow_string_path(self, tmp_path: Path) -> None:
        """Test saving a valid workflow with string path."""
        workflow = Workflow("TestWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        save_path = str(tmp_path / "workflow.pkl")
        save_workflow(workflow, save_path)

        assert Path(save_path).exists()

    def test_save_empty_workflow_raises_error(self, tmp_path: Path) -> None:
        """Test saving an empty workflow raises WorkflowValidationError."""
        workflow = Workflow("EmptyWorkflow")
        save_path = tmp_path / "workflow.pkl"

        with pytest.raises(WorkflowValidationError, match="Cannot save invalid"):
            save_workflow(workflow, save_path)

    def test_save_workflow_without_source_raises_error(self, tmp_path: Path) -> None:
        """Test saving workflow without SourceBlock raises error."""
        workflow = Workflow("NoSourceWorkflow")
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))
        save_path = tmp_path / "workflow.pkl"

        with pytest.raises(WorkflowValidationError):
            save_workflow(workflow, save_path)

    def test_save_workflow_without_sink_raises_error(self, tmp_path: Path) -> None:
        """Test saving workflow without SinkBlock raises error."""
        workflow = Workflow("NoSinkWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        save_path = tmp_path / "workflow.pkl"

        with pytest.raises(WorkflowValidationError):
            save_workflow(workflow, save_path)


class TestLoadWorkflow:
    """Tests for load_workflow function."""

    def test_load_valid_workflow(self, tmp_path: Path) -> None:
        """Test loading a valid workflow."""
        workflow = Workflow("TestWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        save_path = tmp_path / "workflow.pkl"
        save_workflow(workflow, save_path)

        loaded = load_workflow(save_path)

        assert loaded.name == "TestWorkflow"
        assert len(loaded.blocks) == 3

    def test_load_valid_workflow_string_path(self, tmp_path: Path) -> None:
        """Test loading a valid workflow with string path."""
        workflow = Workflow("TestWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        save_path = tmp_path / "workflow.pkl"
        save_workflow(workflow, save_path)

        loaded = load_workflow(str(save_path))

        assert loaded.name == "TestWorkflow"
        assert len(loaded.blocks) == 3

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test loading from nonexistent file raises FileNotFoundError."""
        save_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            load_workflow(save_path)

    def test_load_preserves_workflow_state(self, tmp_path: Path) -> None:
        """Test that loaded workflow preserves original state."""
        workflow = Workflow("PreserveStateWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        save_path = tmp_path / "workflow.pkl"
        save_workflow(workflow, save_path)

        loaded = load_workflow(save_path)

        assert loaded.name == workflow.name
        assert len(loaded.blocks) == len(workflow.blocks)
        for i, block in enumerate(loaded.blocks):
            assert isinstance(workflow.blocks[i], type(block))


class TestRoundTrip:
    """Tests for save/load round-trip behavior."""

    def test_roundtrip_preserves_block_types(self, tmp_path: Path) -> None:
        """Test round-trip preserves block types."""
        workflow = Workflow("RoundTripWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        save_path = tmp_path / "workflow.pkl.gz"
        save_workflow(workflow, save_path)
        loaded = load_workflow(save_path)

        assert isinstance(loaded.blocks[0], SourceBlock)
        assert isinstance(loaded.blocks[1], SquareBlock)
        assert isinstance(loaded.blocks[2], SinkBlock)

    def test_saved_file_is_gzip_compressed(self, tmp_path: Path) -> None:
        """Test that saved files are actually gzip-compressed."""
        workflow = Workflow("GzipWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        save_path = tmp_path / "workflow.pkl.gz"
        save_workflow(workflow, save_path)

        # Verify the file starts with gzip magic bytes
        with open(save_path, "rb") as f:
            magic = f.read(2)
        assert magic == b"\x1f\x8b", "File is not gzip-compressed"

    def test_load_legacy_uncompressed_pkl(self, tmp_path: Path) -> None:
        """Test that load_workflow can read legacy uncompressed pickle files."""
        workflow = Workflow("LegacyWorkflow")
        workflow.add(SourceBlock(simple_reader))
        workflow.add(SquareBlock())
        workflow.add(SinkBlock(simple_writer))

        # Save as plain pickle (legacy format)
        save_path = tmp_path / "legacy.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(workflow, f)

        loaded = load_workflow(save_path)
        assert loaded.name == "LegacyWorkflow"
        assert len(loaded.blocks) == 3
