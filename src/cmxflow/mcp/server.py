"""FastMCP server for building and running cmxflow workflows."""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from cmxflow import Workflow
from cmxflow.block import ScoreBlock, SinkBlock
from cmxflow.mcp.state import (
    get_available_blocks,
    get_block_descriptions,
    get_global_state,
    reset_global_state,
)
from cmxflow.operators import RDKitBlock
from cmxflow.sinks import MoleculeSinkBlock
from cmxflow.sources import MoleculeSourceBlock

mcp = FastMCP(name="cmxflow")


def _format_workflow(workflow: Workflow) -> str:
    """Format workflow as a string representation.

    Args:
        workflow: Workflow to format.

    Returns:
        String representation of the workflow.
    """
    if not workflow.blocks:
        return "Empty workflow"
    return str(workflow)


def _build_workflow_impl(
    action: str,
    block_type: str | None = None,
    block_config: dict[str, Any] | None = None,
    rdkit_method: str | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    """Implementation of build_workflow logic.

    Args:
        action: One of "create", "add_block", "remove_block", "list_blocks",
            "validate", "clear", "show".
        block_type: Block class name (e.g., "ConformerGenerationBlock").
        block_config: Block initialization parameters.
        rdkit_method: For RDKitBlock, the method path
            (e.g., "rdkit.Chem.Descriptors.MolWt").
        index: Position for insert/remove operations.

    Returns:
        Status message and current workflow state.
    """
    state = get_global_state()

    if action == "create":
        # Initialize new workflow with source block
        reset_global_state()
        state = get_global_state()
        workflow = Workflow(name="MCP Workflow")
        workflow.add(MoleculeSourceBlock())
        state.workflow = workflow
        state.validated = False
        state.inputs_set = False
        return {
            "status": "success",
            "message": "Created new workflow with MoleculeSourceBlock",
            "workflow": _format_workflow(workflow),
        }

    elif action == "add_block":
        if state.workflow is None:
            return {
                "status": "error",
                "message": "No workflow exists. Use action='create' first.",
            }

        if block_type is None and rdkit_method is None:
            return {
                "status": "error",
                "message": "Must provide block_type or rdkit_method",
            }

        # Create the block
        try:
            if rdkit_method is not None:
                block = RDKitBlock(rdkit_method)
            else:
                available = get_available_blocks()
                if block_type not in available:
                    return {
                        "status": "error",
                        "message": f"Unknown block type: {block_type}. "
                        f"Available: {list(available.keys())}",
                    }

                block_class = available[block_type]
                config = block_config or {}
                block = block_class(**config)

            # Insert at position or append
            if index is not None:
                state.workflow.insert(index, block)
            else:
                # If last block is a sink, insert before it
                if state.workflow.blocks and isinstance(
                    state.workflow.blocks[-1], (SinkBlock, ScoreBlock)
                ):
                    state.workflow.insert(len(state.workflow.blocks) - 1, block)
                else:
                    state.workflow.add(block)

            state.validated = False
            state.inputs_set = False

            return {
                "status": "success",
                "message": f"Added {block.name}",
                "workflow": _format_workflow(state.workflow),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create block: {e}",
            }

    elif action == "remove_block":
        if state.workflow is None:
            return {
                "status": "error",
                "message": "No workflow exists. Use action='create' first.",
            }

        if index is None:
            return {
                "status": "error",
                "message": "Must provide index for remove_block action",
            }

        if index < 0 or index >= len(state.workflow.blocks):
            return {
                "status": "error",
                "message": f"Index {index} out of range. "
                f"Workflow has {len(state.workflow.blocks)} blocks (0-indexed).",
            }

        removed = state.workflow.blocks.pop(index)
        state.validated = False
        state.inputs_set = False

        return {
            "status": "success",
            "message": f"Removed {removed.name} at index {index}",
            "workflow": _format_workflow(state.workflow),
        }

    elif action == "list_blocks":
        descriptions = get_block_descriptions()
        return {
            "status": "success",
            "blocks": descriptions,
        }

    elif action == "validate":
        if state.workflow is None:
            return {
                "status": "error",
                "message": "No workflow exists. Use action='create' first.",
            }

        # Ensure workflow ends with a sink or score block
        if not state.workflow.blocks or not isinstance(
            state.workflow.blocks[-1], (SinkBlock, ScoreBlock)
        ):
            # Auto-add sink if needed
            state.workflow.add(MoleculeSinkBlock())

        try:
            state.workflow.check()
            state.validated = True
            return {
                "status": "success",
                "message": "Workflow is valid",
                "workflow": _format_workflow(state.workflow),
            }
        except Exception as e:
            state.validated = False
            return {
                "status": "error",
                "message": f"Validation failed: {e}",
                "workflow": _format_workflow(state.workflow),
            }

    elif action == "clear":
        reset_global_state()
        return {
            "status": "success",
            "message": "Workflow cleared",
        }

    elif action == "show":
        if state.workflow is None:
            return {
                "status": "success",
                "message": "No workflow exists",
                "workflow": None,
                "validated": False,
                "inputs_set": False,
            }

        return {
            "status": "success",
            "workflow": _format_workflow(state.workflow),
            "validated": state.validated,
            "inputs_set": state.inputs_set,
            "num_blocks": len(state.workflow.blocks),
        }

    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. "
            "Valid actions: create, add_block, remove_block, list_blocks, "
            "validate, clear, show",
        }


def _run_workflow_impl(
    action: str,
    inputs: dict[str, str] | None = None,
    input_file: str | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Implementation of run_workflow logic.

    Args:
        action: One of "get_inputs", "set_inputs", "execute".
        inputs: Input values for "set_inputs" action. Keys should match
            the format returned by get_inputs (e.g., "1.file@reference").
        input_file: Path to input molecule file for "execute" action.
        output_file: Path for output file for "execute" action.

    Returns:
        Required inputs dict, execution status, or results.
    """
    state = get_global_state()

    if state.workflow is None:
        return {
            "status": "error",
            "message": "No workflow exists. Use build_workflow(action='create') first.",
        }

    if action == "get_inputs":
        if not state.validated:
            return {
                "status": "error",
                "message": "Workflow not validated. "
                "Use build_workflow(action='validate') first.",
            }

        try:
            required = state.workflow.get_required_input()
            return {
                "status": "success",
                "required_inputs": {k: str(v) for k, v in required.items()},
                "message": (
                    "Provide these inputs using set_inputs action"
                    if required
                    else "No additional inputs required"
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get required inputs: {e}",
            }

    elif action == "set_inputs":
        if not state.validated:
            return {
                "status": "error",
                "message": "Workflow not validated. "
                "Use build_workflow(action='validate') first.",
            }

        if inputs is None:
            return {
                "status": "error",
                "message": "Must provide inputs dict for set_inputs action",
            }

        try:
            state.workflow.set_required_input(inputs)
            state.inputs_set = True
            return {
                "status": "success",
                "message": "Inputs set successfully",
            }
        except FileNotFoundError as e:
            return {
                "status": "error",
                "message": f"File not found: {e}",
            }
        except KeyError as e:
            return {
                "status": "error",
                "message": f"Missing required input: {e}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to set inputs: {e}",
            }

    elif action == "execute":
        if not state.validated:
            return {
                "status": "error",
                "message": "Workflow not validated. "
                "Use build_workflow(action='validate') first.",
            }

        # Check if inputs are required but not set
        required = state.workflow.get_required_input()
        if required and not state.inputs_set:
            return {
                "status": "error",
                "message": "Required inputs not set. "
                "Use run_workflow(action='set_inputs') first.",
                "required_inputs": {k: str(v) for k, v in required.items()},
            }

        if input_file is None:
            return {
                "status": "error",
                "message": "Must provide input_file for execute action",
            }

        input_path = Path(input_file)
        if not input_path.is_file():
            return {
                "status": "error",
                "message": f"Input file not found: {input_file}",
            }

        output_path = Path(output_file) if output_file else Path("")

        try:
            result = state.workflow(input_path, output_path)
            return {
                "status": "success",
                "message": "Workflow executed successfully",
                "result": result,
                "output_file": str(output_path) if output_file else None,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Execution failed: {e}",
            }

    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. "
            "Valid actions: get_inputs, set_inputs, execute",
        }


@mcp.tool
def build_workflow(
    action: str,
    block_type: str | None = None,
    block_config: dict[str, Any] | None = None,
    rdkit_method: str | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    """Build a cheminformatics workflow step-by-step.

    Args:
        action: One of "create", "add_block", "remove_block", "list_blocks",
            "validate", "clear", "show".
        block_type: Block class name (e.g., "ConformerGenerationBlock").
        block_config: Block initialization parameters.
        rdkit_method: For RDKitBlock, the method path
            (e.g., "rdkit.Chem.Descriptors.MolWt").
        index: Position for insert/remove operations.

    Returns:
        Status message and current workflow state.
    """
    return _build_workflow_impl(
        action=action,
        block_type=block_type,
        block_config=block_config,
        rdkit_method=rdkit_method,
        index=index,
    )


@mcp.tool
def run_workflow(
    action: str,
    inputs: dict[str, str] | None = None,
    input_file: str | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Execute a validated workflow.

    Args:
        action: One of "get_inputs", "set_inputs", "execute".
        inputs: Input values for "set_inputs" action. Keys should match
            the format returned by get_inputs (e.g., "1.file@reference").
        input_file: Path to input molecule file for "execute" action.
        output_file: Path for output file for "execute" action.

    Returns:
        Required inputs dict, execution status, or results.
    """
    return _run_workflow_impl(
        action=action,
        inputs=inputs,
        input_file=input_file,
        output_file=output_file,
    )
