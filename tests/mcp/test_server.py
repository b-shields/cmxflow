"""Tests for the cmxflow MCP server."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cmxflow.mcp.server import (
    _build_workflow_impl,
    _optimize_workflow_impl,
    _run_workflow_impl,
)
from cmxflow.mcp.state import (
    get_available_blocks,
    get_block_descriptions,
    get_global_state,
    reset_global_state,
)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset global state before each test for isolation."""
    reset_global_state()


class TestState:
    """Tests for state management utilities."""

    def test_get_available_blocks(self) -> None:
        """Test that get_available_blocks returns expected block types."""
        blocks = get_available_blocks()

        assert "MoleculeSourceBlock" in blocks
        assert "MoleculeSinkBlock" in blocks
        assert "ConformerGenerationBlock" in blocks
        assert "RDKitBlock" in blocks
        assert "EnrichmentScoreBlock" in blocks

    def test_get_block_descriptions(self) -> None:
        """Test that get_block_descriptions returns descriptions for all blocks."""
        descriptions = get_block_descriptions()
        blocks = get_available_blocks()

        # All blocks should have descriptions
        for block_name in blocks:
            assert block_name in descriptions
            assert isinstance(descriptions[block_name], str)
            assert len(descriptions[block_name]) > 0


class TestBuildWorkflow:
    """Tests for the build_workflow tool."""

    def test_create_workflow(self) -> None:
        """Test creating a new workflow."""
        result = _build_workflow_impl(action="create")

        assert result["status"] == "success"
        assert "MoleculeSourceBlock" in result["message"]
        assert "workflow" in result

    def test_show_empty(self) -> None:
        """Test showing workflow when none exists."""
        result = _build_workflow_impl(action="show")

        assert result["status"] == "success"
        assert result["workflow"] is None

    def test_show_after_create(self) -> None:
        """Test showing workflow after creation."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="show")

        assert result["status"] == "success"
        assert result["workflow"] is not None
        assert result["validated"] is False
        assert result["num_blocks"] == 1

    def test_list_blocks(self) -> None:
        """Test listing available blocks."""
        result = _build_workflow_impl(action="list_blocks")

        assert result["status"] == "success"
        assert "blocks" in result
        assert "ConformerGenerationBlock" in result["blocks"]

    def test_add_block_no_workflow(self) -> None:
        """Test adding block when no workflow exists."""
        result = _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_add_block_success(self) -> None:
        """Test adding a block to the workflow."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )

        assert result["status"] == "success"
        assert "ConformerGeneration" in result["message"]

    def test_add_rdkit_block(self) -> None:
        """Test adding an RDKitBlock with method path."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(
            action="add_block",
            rdkit_method="rdkit.Chem.Descriptors.MolWt",
        )

        assert result["status"] == "success"
        assert "MolWt" in result["message"]

    def test_add_unknown_block(self) -> None:
        """Test adding an unknown block type."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(
            action="add_block",
            block_type="UnknownBlock",
        )

        assert result["status"] == "error"
        assert "Unknown block type" in result["message"]

    def test_remove_block(self) -> None:
        """Test removing a block from the workflow."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )

        # Remove the conformer block (index 1)
        result = _build_workflow_impl(action="remove_block", index=1)

        assert result["status"] == "success"
        assert "Removed" in result["message"]

    def test_remove_block_no_index(self) -> None:
        """Test removing block without providing index."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="remove_block")

        assert result["status"] == "error"
        assert "Must provide index" in result["message"]

    def test_remove_block_out_of_range(self) -> None:
        """Test removing block with out-of-range index."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="remove_block", index=10)

        assert result["status"] == "error"
        assert "out of range" in result["message"]

    def test_validate_workflow(self) -> None:
        """Test validating a complete workflow."""
        _build_workflow_impl(action="create")
        # Just source and sink should be valid
        result = _build_workflow_impl(action="validate")

        assert result["status"] == "success"
        assert "valid" in result["message"]

    def test_validate_no_workflow(self) -> None:
        """Test validating when no workflow exists."""
        result = _build_workflow_impl(action="validate")

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_clear_workflow(self) -> None:
        """Test clearing the workflow."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="clear")

        assert result["status"] == "success"
        assert "cleared" in result["message"]

        # Verify workflow is gone
        show_result = _build_workflow_impl(action="show")
        assert show_result["workflow"] is None

    def test_unknown_action(self) -> None:
        """Test unknown action returns error."""
        result = _build_workflow_impl(action="unknown")

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]


class TestRunWorkflow:
    """Tests for the run_workflow tool."""

    def test_no_workflow(self) -> None:
        """Test running when no workflow exists."""
        result = _run_workflow_impl(action="get_inputs")

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_get_inputs_not_validated(self) -> None:
        """Test getting inputs when workflow not validated."""
        _build_workflow_impl(action="create")
        result = _run_workflow_impl(action="get_inputs")

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_get_inputs_success(self) -> None:
        """Test getting required inputs after validation."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")
        result = _run_workflow_impl(action="get_inputs")

        assert result["status"] == "success"
        assert "required_inputs" in result

    def test_set_inputs_not_validated(self) -> None:
        """Test setting inputs when workflow not validated."""
        _build_workflow_impl(action="create")
        result = _run_workflow_impl(action="set_inputs", inputs={})

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_set_inputs_no_inputs_provided(self) -> None:
        """Test setting inputs without providing inputs dict."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")
        result = _run_workflow_impl(action="set_inputs")

        assert result["status"] == "error"
        assert "Must provide inputs" in result["message"]

    def test_execute_not_validated(self) -> None:
        """Test executing when workflow not validated."""
        _build_workflow_impl(action="create")
        result = _run_workflow_impl(
            action="execute",
            input_file="test.sdf",
            output_file="out.sdf",
        )

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_execute_no_input_file(self) -> None:
        """Test executing without input file."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")
        result = _run_workflow_impl(action="execute")

        assert result["status"] == "error"
        assert "Must provide input_file" in result["message"]

    def test_execute_file_not_found(self) -> None:
        """Test executing with non-existent input file."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")
        result = _run_workflow_impl(
            action="execute",
            input_file="/nonexistent/file.sdf",
            output_file="out.sdf",
        )

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_unknown_action(self) -> None:
        """Test unknown action returns error."""
        _build_workflow_impl(action="create")
        result = _run_workflow_impl(action="unknown")

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_build_and_validate_complete_workflow(self) -> None:
        """Test building a complete workflow with multiple blocks."""
        # Create workflow
        result = _build_workflow_impl(action="create")
        assert result["status"] == "success"

        # Add EnumerateStereoBlock
        result = _build_workflow_impl(
            action="add_block",
            block_type="EnumerateStereoBlock",
        )
        assert result["status"] == "success"

        # Add ConformerGenerationBlock
        result = _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        assert result["status"] == "success"

        # Validate (should auto-add sink)
        result = _build_workflow_impl(action="validate")
        assert result["status"] == "success"

        # Check workflow state
        result = _build_workflow_impl(action="show")
        assert result["validated"] is True
        assert result["num_blocks"] == 4  # source + 2 operators + sink


class TestOptimizeWorkflow:
    """Tests for the optimize_workflow tool."""

    def test_optimize_no_workflow(self) -> None:
        """Test optimization when no workflow exists."""
        result = _optimize_workflow_impl(action="start", n_trials=10)

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_optimize_workflow_not_validated(self) -> None:
        """Test optimization when workflow not validated."""
        _build_workflow_impl(action="create")
        result = _optimize_workflow_impl(action="start", n_trials=10)

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_optimize_workflow_ends_with_sink(self) -> None:
        """Test optimization fails when workflow ends with SinkBlock."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")  # Auto-adds sink

        result = _optimize_workflow_impl(
            action="start",
            n_trials=10,
            input_file="test.sdf",
        )

        assert result["status"] == "error"
        assert "must end with ScoreBlock" in result["message"]

    def test_optimize_workflow_no_params(self) -> None:
        """Test optimization fails when no optimizable parameters."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        result = _optimize_workflow_impl(
            action="start",
            n_trials=10,
            input_file="test.sdf",
        )

        assert result["status"] == "error"
        assert "no optimizable parameters" in result["message"]

    def test_optimize_missing_n_trials(self) -> None:
        """Test optimization fails when n_trials not provided."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        result = _optimize_workflow_impl(action="start", input_file="test.sdf")

        assert result["status"] == "error"
        assert "Must provide n_trials" in result["message"]

    def test_optimize_missing_input_file(self) -> None:
        """Test optimization fails when input_file not provided."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        result = _optimize_workflow_impl(action="start", n_trials=10)

        assert result["status"] == "error"
        assert "Must provide input_file" in result["message"]

    def test_optimize_input_file_not_found(self) -> None:
        """Test optimization fails when input file doesn't exist."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        result = _optimize_workflow_impl(
            action="start",
            n_trials=10,
            input_file="/nonexistent/file.sdf",
        )

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_optimize_invalid_direction(self) -> None:
        """Test optimization fails with invalid direction."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        # Need a valid input file for this test
        with patch.object(Path, "is_file", return_value=True):
            result = _optimize_workflow_impl(
                action="start",
                n_trials=10,
                input_file="test.sdf",
                direction="invalid",
            )

        assert result["status"] == "error"
        assert "Invalid direction" in result["message"]

    def test_optimize_status_no_optimization(self) -> None:
        """Test status when no optimization running."""
        result = _optimize_workflow_impl(action="status")

        assert result["status"] == "no_optimization"

    def test_optimize_get_best_params_no_optimization(self) -> None:
        """Test get_best_params when no optimization has been run."""
        result = _optimize_workflow_impl(action="get_best_params")

        assert result["status"] == "error"
        assert "No optimization has been run" in result["message"]

    def test_optimize_set_best_params_no_optimization(self) -> None:
        """Test set_best_params when no optimization has been run."""
        result = _optimize_workflow_impl(action="set_best_params")

        assert result["status"] == "error"
        assert "No optimization has been run" in result["message"]

    def test_optimize_cancel_no_optimization(self) -> None:
        """Test cancel when no optimization running."""
        result = _optimize_workflow_impl(action="cancel")

        assert result["status"] == "error"
        assert "No optimization running" in result["message"]

    def test_optimize_unknown_action(self) -> None:
        """Test unknown action returns error."""
        result = _optimize_workflow_impl(action="unknown")

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]

    def test_optimize_start_success(self, tmp_path: Path) -> None:
        """Test starting optimization successfully."""
        # Create a test input file
        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        # Build workflow with ScoreBlock
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        # Mock the Optimizer to avoid actual optimization
        with patch("cmxflow.mcp.server.Optimizer") as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer_class.return_value = mock_optimizer

            result = _optimize_workflow_impl(
                action="start",
                n_trials=10,
                input_file=str(input_file),
                inputs={"2.text@target": "is_active"},
            )

        assert result["status"] == "started"
        assert result["n_trials"] == 10
        assert result["direction"] == "maximize"

    def test_optimize_status_running(self, tmp_path: Path) -> None:
        """Test status while optimization is running."""
        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        # Mock optimization to take time
        def slow_optimize(*args, **kwargs):
            time.sleep(0.5)

        with patch("cmxflow.mcp.server.Optimizer") as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize = slow_optimize
            mock_optimizer_class.return_value = mock_optimizer

            _optimize_workflow_impl(
                action="start",
                n_trials=10,
                input_file=str(input_file),
                inputs={"2.text@target": "is_active"},
            )

            # Check status immediately (should be running)
            result = _optimize_workflow_impl(action="status")

            # Could be running or completed depending on timing
            assert result["status"] in ("running", "completed")

    def test_optimize_status_completed(self, tmp_path: Path) -> None:
        """Test status when optimization is completed."""
        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        with patch("cmxflow.mcp.server.Optimizer") as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.best_params = {"num_conformers": 5}
            mock_optimizer.best_score = 0.85
            mock_optimizer_class.return_value = mock_optimizer

            _optimize_workflow_impl(
                action="start",
                n_trials=10,
                input_file=str(input_file),
                inputs={"2.text@target": "is_active"},
            )

            # Wait for completion
            state = get_global_state()
            assert state.optimization_future is not None
            state.optimization_future.result(timeout=5)

            result = _optimize_workflow_impl(action="status")

        assert result["status"] == "completed"
        assert result["best_params"] == {"num_conformers": 5}
        assert result["best_score"] == 0.85

    def test_optimize_get_best_params_success(self, tmp_path: Path) -> None:
        """Test getting best params after optimization."""
        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        with patch("cmxflow.mcp.server.Optimizer") as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.best_params = {"num_conformers": 10}
            mock_optimizer.best_score = 0.92
            mock_optimizer_class.return_value = mock_optimizer

            _optimize_workflow_impl(
                action="start",
                n_trials=10,
                input_file=str(input_file),
                inputs={"2.text@target": "is_active"},
            )

            # Wait for completion
            state = get_global_state()
            assert state.optimization_future is not None
            state.optimization_future.result(timeout=5)

            result = _optimize_workflow_impl(action="get_best_params")

        assert result["status"] == "success"
        assert result["best_params"] == {"num_conformers": 10}
        assert result["best_score"] == 0.92

    def test_optimize_set_best_params_success(self, tmp_path: Path) -> None:
        """Test setting best params after optimization."""
        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        with patch("cmxflow.mcp.server.Optimizer") as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.best_params = {"num_conformers": 7}
            mock_optimizer_class.return_value = mock_optimizer

            _optimize_workflow_impl(
                action="start",
                n_trials=10,
                input_file=str(input_file),
                inputs={"2.text@target": "is_active"},
            )

            # Wait for completion
            state = get_global_state()
            assert state.optimization_future is not None
            state.optimization_future.result(timeout=5)

            result = _optimize_workflow_impl(action="set_best_params")

        assert result["status"] == "success"
        assert "Best parameters applied" in result["message"]
        mock_optimizer.set_best_params.assert_called_once()

    def test_optimize_already_running(self, tmp_path: Path) -> None:
        """Test starting optimization when one is already running."""
        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        def slow_optimize(*args, **kwargs):
            time.sleep(1)

        with patch("cmxflow.mcp.server.Optimizer") as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize = slow_optimize
            mock_optimizer_class.return_value = mock_optimizer

            # Start first optimization
            _optimize_workflow_impl(
                action="start",
                n_trials=10,
                input_file=str(input_file),
                inputs={"2.text@target": "is_active"},
            )

            # Try to start another
            result = _optimize_workflow_impl(
                action="start",
                n_trials=5,
                input_file=str(input_file),
            )

        assert result["status"] == "error"
        assert "already in progress" in result["message"]
