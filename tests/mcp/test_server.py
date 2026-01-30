"""Tests for the cmxflow MCP server."""

from typing import Any

import pytest

from cmxflow.mcp.server import _build_workflow_impl, _run_workflow_impl
from cmxflow.mcp.state import (
    get_available_blocks,
    get_block_descriptions,
)


class MockContext:
    """Mock FastMCP context for testing."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def get_state(self, key: str) -> Any:
        return self._state.get(key)

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value


@pytest.fixture
def ctx() -> MockContext:
    """Create a fresh mock context for each test."""
    return MockContext()


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

    def test_create_workflow(self, ctx: MockContext) -> None:
        """Test creating a new workflow."""
        result = _build_workflow_impl(ctx=ctx, action="create")

        assert result["status"] == "success"
        assert "MoleculeSourceBlock" in result["message"]
        assert "workflow" in result

    def test_show_empty(self, ctx: MockContext) -> None:
        """Test showing workflow when none exists."""
        result = _build_workflow_impl(ctx=ctx, action="show")

        assert result["status"] == "success"
        assert result["workflow"] is None

    def test_show_after_create(self, ctx: MockContext) -> None:
        """Test showing workflow after creation."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _build_workflow_impl(ctx=ctx, action="show")

        assert result["status"] == "success"
        assert result["workflow"] is not None
        assert result["validated"] is False
        assert result["num_blocks"] == 1

    def test_list_blocks(self, ctx: MockContext) -> None:
        """Test listing available blocks."""
        result = _build_workflow_impl(ctx=ctx, action="list_blocks")

        assert result["status"] == "success"
        assert "blocks" in result
        assert "ConformerGenerationBlock" in result["blocks"]

    def test_add_block_no_workflow(self, ctx: MockContext) -> None:
        """Test adding block when no workflow exists."""
        result = _build_workflow_impl(
            ctx=ctx,
            action="add_block",
            block_type="ConformerGenerationBlock",
        )

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_add_block_success(self, ctx: MockContext) -> None:
        """Test adding a block to the workflow."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _build_workflow_impl(
            ctx=ctx,
            action="add_block",
            block_type="ConformerGenerationBlock",
        )

        assert result["status"] == "success"
        assert "ConformerGeneration" in result["message"]

    def test_add_rdkit_block(self, ctx: MockContext) -> None:
        """Test adding an RDKitBlock with method path."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _build_workflow_impl(
            ctx=ctx,
            action="add_block",
            rdkit_method="rdkit.Chem.Descriptors.MolWt",
        )

        assert result["status"] == "success"
        assert "MolWt" in result["message"]

    def test_add_unknown_block(self, ctx: MockContext) -> None:
        """Test adding an unknown block type."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _build_workflow_impl(
            ctx=ctx,
            action="add_block",
            block_type="UnknownBlock",
        )

        assert result["status"] == "error"
        assert "Unknown block type" in result["message"]

    def test_remove_block(self, ctx: MockContext) -> None:
        """Test removing a block from the workflow."""
        _build_workflow_impl(ctx=ctx, action="create")
        _build_workflow_impl(
            ctx=ctx,
            action="add_block",
            block_type="ConformerGenerationBlock",
        )

        # Remove the conformer block (index 1)
        result = _build_workflow_impl(ctx=ctx, action="remove_block", index=1)

        assert result["status"] == "success"
        assert "Removed" in result["message"]

    def test_remove_block_no_index(self, ctx: MockContext) -> None:
        """Test removing block without providing index."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _build_workflow_impl(ctx=ctx, action="remove_block")

        assert result["status"] == "error"
        assert "Must provide index" in result["message"]

    def test_remove_block_out_of_range(self, ctx: MockContext) -> None:
        """Test removing block with out-of-range index."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _build_workflow_impl(ctx=ctx, action="remove_block", index=10)

        assert result["status"] == "error"
        assert "out of range" in result["message"]

    def test_validate_workflow(self, ctx: MockContext) -> None:
        """Test validating a complete workflow."""
        _build_workflow_impl(ctx=ctx, action="create")
        # Just source and sink should be valid
        result = _build_workflow_impl(ctx=ctx, action="validate")

        assert result["status"] == "success"
        assert "valid" in result["message"]

    def test_validate_no_workflow(self, ctx: MockContext) -> None:
        """Test validating when no workflow exists."""
        result = _build_workflow_impl(ctx=ctx, action="validate")

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_clear_workflow(self, ctx: MockContext) -> None:
        """Test clearing the workflow."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _build_workflow_impl(ctx=ctx, action="clear")

        assert result["status"] == "success"
        assert "cleared" in result["message"]

        # Verify workflow is gone
        show_result = _build_workflow_impl(ctx=ctx, action="show")
        assert show_result["workflow"] is None

    def test_unknown_action(self, ctx: MockContext) -> None:
        """Test unknown action returns error."""
        result = _build_workflow_impl(ctx=ctx, action="unknown")

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]


class TestRunWorkflow:
    """Tests for the run_workflow tool."""

    def test_no_workflow(self, ctx: MockContext) -> None:
        """Test running when no workflow exists."""
        result = _run_workflow_impl(ctx=ctx, action="get_inputs")

        assert result["status"] == "error"
        assert "No workflow exists" in result["message"]

    def test_get_inputs_not_validated(self, ctx: MockContext) -> None:
        """Test getting inputs when workflow not validated."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _run_workflow_impl(ctx=ctx, action="get_inputs")

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_get_inputs_success(self, ctx: MockContext) -> None:
        """Test getting required inputs after validation."""
        _build_workflow_impl(ctx=ctx, action="create")
        _build_workflow_impl(ctx=ctx, action="validate")
        result = _run_workflow_impl(ctx=ctx, action="get_inputs")

        assert result["status"] == "success"
        assert "required_inputs" in result

    def test_set_inputs_not_validated(self, ctx: MockContext) -> None:
        """Test setting inputs when workflow not validated."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _run_workflow_impl(ctx=ctx, action="set_inputs", inputs={})

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_set_inputs_no_inputs_provided(self, ctx: MockContext) -> None:
        """Test setting inputs without providing inputs dict."""
        _build_workflow_impl(ctx=ctx, action="create")
        _build_workflow_impl(ctx=ctx, action="validate")
        result = _run_workflow_impl(ctx=ctx, action="set_inputs")

        assert result["status"] == "error"
        assert "Must provide inputs" in result["message"]

    def test_execute_not_validated(self, ctx: MockContext) -> None:
        """Test executing when workflow not validated."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _run_workflow_impl(
            ctx=ctx,
            action="execute",
            input_file="test.sdf",
            output_file="out.sdf",
        )

        assert result["status"] == "error"
        assert "not validated" in result["message"]

    def test_execute_no_input_file(self, ctx: MockContext) -> None:
        """Test executing without input file."""
        _build_workflow_impl(ctx=ctx, action="create")
        _build_workflow_impl(ctx=ctx, action="validate")
        result = _run_workflow_impl(ctx=ctx, action="execute")

        assert result["status"] == "error"
        assert "Must provide input_file" in result["message"]

    def test_execute_file_not_found(self, ctx: MockContext) -> None:
        """Test executing with non-existent input file."""
        _build_workflow_impl(ctx=ctx, action="create")
        _build_workflow_impl(ctx=ctx, action="validate")
        result = _run_workflow_impl(
            ctx=ctx,
            action="execute",
            input_file="/nonexistent/file.sdf",
            output_file="out.sdf",
        )

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_unknown_action(self, ctx: MockContext) -> None:
        """Test unknown action returns error."""
        _build_workflow_impl(ctx=ctx, action="create")
        result = _run_workflow_impl(ctx=ctx, action="unknown")

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_build_and_validate_complete_workflow(self, ctx: MockContext) -> None:
        """Test building a complete workflow with multiple blocks."""
        # Create workflow
        result = _build_workflow_impl(ctx=ctx, action="create")
        assert result["status"] == "success"

        # Add EnumerateStereoBlock
        result = _build_workflow_impl(
            ctx=ctx,
            action="add_block",
            block_type="EnumerateStereoBlock",
        )
        assert result["status"] == "success"

        # Add ConformerGenerationBlock
        result = _build_workflow_impl(
            ctx=ctx,
            action="add_block",
            block_type="ConformerGenerationBlock",
        )
        assert result["status"] == "success"

        # Validate (should auto-add sink)
        result = _build_workflow_impl(ctx=ctx, action="validate")
        assert result["status"] == "success"

        # Check workflow state
        result = _build_workflow_impl(ctx=ctx, action="show")
        assert result["validated"] is True
        assert result["num_blocks"] == 4  # source + 2 operators + sink
