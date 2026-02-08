"""Tests for the cmxflow MCP server."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cmxflow.mcp.server import (
    _build_workflow_impl,
    _manage_workflows_impl,
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
        """Test validating a complete workflow returns enriched response."""
        _build_workflow_impl(action="create")
        # Just source — validate auto-adds sink
        result = _build_workflow_impl(action="validate")

        assert result["status"] == "success"
        assert "valid" in result["message"]
        assert result["validated"] is True
        assert result["num_blocks"] == 2  # source + auto-added sink
        assert "required_inputs" in result
        assert isinstance(result["required_inputs"], dict)

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

    def test_make_parallel_no_workflow(self) -> None:
        """Test make_parallel when no workflow exists."""
        result = _build_workflow_impl(action="make_parallel", index=0)

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_make_parallel_no_index(self) -> None:
        """Test make_parallel without providing index."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="make_parallel")

        assert result["status"] == "error"
        assert "Must provide index" in result["message"]

    def test_make_parallel_index_out_of_range(self) -> None:
        """Test make_parallel with out-of-range index."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="make_parallel", index=10)

        assert result["status"] == "error"
        assert "out of range" in result["message"]

    def test_make_parallel_source_block(self) -> None:
        """Test make_parallel fails on SourceBlock."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="make_parallel", index=0)

        assert result["status"] == "error"
        assert "Cannot parallelize" in result["message"]
        assert "SourceBlock" in result["message"]

    def test_make_parallel_sink_block(self) -> None:
        """Test make_parallel fails on SinkBlock."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")  # Auto-adds sink
        # Sink is at index 1
        result = _build_workflow_impl(action="make_parallel", index=1)

        assert result["status"] == "error"
        assert "Cannot parallelize" in result["message"]
        assert "SinkBlock" in result["message"]

    def test_make_parallel_score_block(self) -> None:
        """Test make_parallel fails on ScoreBlock."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        # ScoreBlock is at index 1
        result = _build_workflow_impl(action="make_parallel", index=1)

        assert result["status"] == "error"
        assert "Cannot parallelize" in result["message"]
        assert "ScoreBlock" in result["message"]

    def test_make_parallel_success(self) -> None:
        """Test successfully parallelizing a processing block."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        # ConformerGenerationBlock is at index 1
        result = _build_workflow_impl(action="make_parallel", index=1)

        assert result["status"] == "success"
        assert "Parallelized" in result["message"]
        assert "ConformerGeneration" in result["message"]
        assert "workflow" in result

    def test_make_parallel_with_config(self) -> None:
        """Test parallelizing with custom configuration."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        result = _build_workflow_impl(
            action="make_parallel",
            index=1,
            block_config={
                "max_workers": 4,
                "chunk_size": 10,
                "ordered": False,
                "error_handling": "log",
            },
        )

        assert result["status"] == "success"
        assert "Parallelized" in result["message"]

    def test_make_parallel_already_parallel(self) -> None:
        """Test make_parallel fails on already parallelized block."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        # Parallelize the block first
        _build_workflow_impl(action="make_parallel", index=1)

        # Try to parallelize again
        result = _build_workflow_impl(action="make_parallel", index=1)

        assert result["status"] == "error"
        assert "already parallelized" in result["message"]

    def test_make_parallel_invalidates_workflow(self) -> None:
        """Test that make_parallel resets validated and inputs_set state."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        _build_workflow_impl(action="validate")

        # Verify validated is True
        show_result = _build_workflow_impl(action="show")
        assert show_result["validated"] is True

        # Parallelize the block
        _build_workflow_impl(action="make_parallel", index=1)

        # Verify validated is now False
        show_result = _build_workflow_impl(action="show")
        assert show_result["validated"] is False

    def test_get_params_no_workflow(self) -> None:
        """Test get_params when no workflow exists."""
        result = _build_workflow_impl(action="get_params")

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_get_params_empty(self) -> None:
        """Test get_params with no optimizable parameters."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="get_params")

        assert result["status"] == "success"
        assert result["params"] == []
        assert "No optimizable parameters" in result["message"]

    def test_get_params_success(self) -> None:
        """Test get_params returns param info for workflow with params."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        result = _build_workflow_impl(action="get_params")

        assert result["status"] == "success"
        assert len(result["params"]) > 0
        param = result["params"][0]
        assert "name" in param
        assert "type" in param
        assert "current" in param
        assert "block" in param

    def test_validate_auto_adds_sink_message(self) -> None:
        """Test that validate message reports auto-added sink."""
        _build_workflow_impl(action="create")
        result = _build_workflow_impl(action="validate")

        assert result["status"] == "success"
        assert "auto-added MoleculeSinkBlock" in result["message"]

    def test_validate_no_auto_add_message(self) -> None:
        """Test that validate message omits auto-add when sink present."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="MoleculeSinkBlock",
        )
        result = _build_workflow_impl(action="validate")

        assert result["status"] == "success"
        assert "auto-added" not in result["message"]


class TestRunWorkflow:
    """Tests for the run_workflow tool."""

    def test_no_workflow(self) -> None:
        """Test running when no workflow exists."""
        result = _run_workflow_impl(action="get_inputs")

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_get_inputs_invalid_workflow(self) -> None:
        """Test getting inputs on invalid workflow returns error from check()."""
        _build_workflow_impl(action="create")
        # Workflow with only source block is structurally invalid
        result = _run_workflow_impl(action="get_inputs")

        assert result["status"] == "error"
        assert "invalid" in result["message"].lower()

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

    def test_execute_sink_workflow_requires_output_file(self, tmp_path: Path) -> None:
        """Test that SinkBlock workflow errors without output_file."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")  # auto-adds sink

        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        result = _run_workflow_impl(
            action="execute",
            input_file=str(input_file),
        )

        assert result["status"] == "error"
        assert "output_file" in result["message"]

    def test_execute_score_workflow_no_output_file(self, tmp_path: Path) -> None:
        """Test that ScoreBlock workflow doesn't require output_file."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(
            action="add_block",
            block_type="EnrichmentScoreBlock",
        )
        _build_workflow_impl(action="validate")

        # Set required inputs for EnrichmentScoreBlock
        _run_workflow_impl(
            action="set_inputs",
            inputs={"1.text@target": "is_active"},
        )

        input_file = tmp_path / "test.sdf"
        input_file.write_text("")

        # Mock workflow forward to return a score
        state = get_global_state()
        assert state.workflow is not None
        with patch.object(state.workflow, "forward", return_value=0.75):
            result = _run_workflow_impl(
                action="execute",
                input_file=str(input_file),
            )

        assert result["status"] == "success"
        assert result["score"] == 0.75
        assert "output_file" not in result


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


class TestManageWorkflows:
    """Tests for the manage_workflows tool."""

    @pytest.fixture(autouse=True)
    def _use_tmp_registry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Override the default registry path so tests use tmp_path.

        Monkeypatches WorkflowRegistry.__init__ default so any state reset
        (including inside build_workflow create action) uses the tmp path.
        """
        from cmxflow.utils.serial import WorkflowRegistry

        registry_path = tmp_path / "registry"
        original_init = WorkflowRegistry.__init__

        def patched_init(
            self_reg: WorkflowRegistry, path: Path | str = registry_path
        ) -> None:
            original_init(self_reg, path)

        monkeypatch.setattr(WorkflowRegistry, "__init__", patched_init)

        # Apply to current state
        state = get_global_state()
        state.registry = WorkflowRegistry(path=registry_path)

    def _create_and_validate(self) -> None:
        """Helper to create and validate a minimal workflow."""
        _build_workflow_impl(action="create")
        _build_workflow_impl(action="validate")

    def test_save_workflow(self) -> None:
        """Test saving a validated workflow."""
        self._create_and_validate()
        result = _manage_workflows_impl(action="save", name="my_workflow")

        assert result["status"] == "success"
        assert "registered" in result["message"]

    def test_save_no_workflow(self) -> None:
        """Test saving with no active workflow returns error."""
        result = _manage_workflows_impl(action="save", name="my_workflow")

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_save_not_validated(self) -> None:
        """Test saving unvalidated workflow returns error."""
        _build_workflow_impl(action="create")
        result = _manage_workflows_impl(action="save", name="unvalidated")

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_save_no_name(self) -> None:
        """Test saving without a name returns error."""
        self._create_and_validate()
        result = _manage_workflows_impl(action="save")

        assert result["status"] == "error"
        assert "name" in result["message"]

    def test_save_duplicate_raises(self) -> None:
        """Test saving same name twice without overwrite returns error."""
        self._create_and_validate()
        _manage_workflows_impl(action="save", name="dup")
        result = _manage_workflows_impl(action="save", name="dup")

        assert result["status"] == "error"
        assert "already exists" in result["message"]

    def test_save_overwrite(self) -> None:
        """Test saving same name with overwrite=True succeeds."""
        self._create_and_validate()
        _manage_workflows_impl(action="save", name="dup")
        result = _manage_workflows_impl(action="save", name="dup", overwrite=True)

        assert result["status"] == "success"

    def test_load_workflow(self) -> None:
        """Test saving then loading a workflow into state."""
        self._create_and_validate()
        _manage_workflows_impl(action="save", name="loadme")

        # Load into current state (overwriting current workflow)
        result = _manage_workflows_impl(action="load", name="loadme")

        assert result["status"] == "success"
        assert "workflow" in result
        state = get_global_state()
        assert state.workflow is not None
        assert state.validated is True
        assert state.inputs_set is False

    def test_load_nonexistent(self) -> None:
        """Test loading a missing workflow returns error."""
        result = _manage_workflows_impl(action="load", name="nonexistent")

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_list_empty(self) -> None:
        """Test listing with no registered workflows returns table with headers."""
        result = _manage_workflows_impl(action="list")

        assert result["status"] == "success"
        assert "workflows" in result
        # Empty DataFrame to_string(index=False) still has column headers
        assert "name" in result["workflows"]

    def test_list_workflows(self) -> None:
        """Test listing after saving two workflows."""
        self._create_and_validate()
        _manage_workflows_impl(action="save", name="wf_one")
        _manage_workflows_impl(action="save", name="wf_two", overwrite=True)

        result = _manage_workflows_impl(action="list")

        assert result["status"] == "success"
        assert "wf_one" in result["workflows"]
        assert "wf_two" in result["workflows"]

    def test_remove_workflow(self) -> None:
        """Test removing a saved workflow."""
        self._create_and_validate()
        _manage_workflows_impl(action="save", name="removeme")

        result = _manage_workflows_impl(action="remove", name="removeme")
        assert result["status"] == "success"

        # Verify it's gone
        list_result = _manage_workflows_impl(action="list")
        assert "removeme" not in list_result["workflows"]

    def test_remove_nonexistent(self) -> None:
        """Test removing a missing workflow returns error."""
        result = _manage_workflows_impl(action="remove", name="nope")

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_unknown_action(self) -> None:
        """Test unknown action returns error."""
        result = _manage_workflows_impl(action="unknown")

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]
