"""FastMCP server for building and running cmxflow workflows."""

import shutil
from pathlib import Path
from typing import Any

import optuna
from fastmcp import FastMCP

from cmxflow import Workflow
from cmxflow.block import Block, ScoreBlock, SinkBlock, SourceBlock
from cmxflow.mcp.state import (
    get_available_blocks,
    get_block_descriptions,
    get_executor,
    get_global_state,
    reset_global_state,
    workflow_has_3d_blocks,
)
from cmxflow.operators import RDKitBlock
from cmxflow.opt import Optimizer
from cmxflow.sinks import MoleculeSinkBlock
from cmxflow.sources import MoleculeSourceBlock
from cmxflow.utils.parallel import ParallelBlock, make_parallel
from cmxflow.utils.pymol import open_pymol_session
from cmxflow.workflow import WorkflowValidationError

_PYMOL_AVAILABLE = shutil.which("pymol") is not None

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
            "validate", "clear", "show", "make_parallel".
        block_type: Block class name (e.g., "ConformerGenerationBlock").
        block_config: Block initialization parameters. For "make_parallel" action,
            this specifies parallel execution options: max_workers, chunk_size,
            ordered, error_handling.
        rdkit_method: For RDKitBlock, the method path
            (e.g., "rdkit.Chem.Descriptors.MolWt").
        index: Position for insert/remove/make_parallel operations.

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

    elif action == "make_parallel":
        if state.workflow is None:
            return {
                "status": "error",
                "message": "No workflow exists. Use action='create' first.",
            }

        if index is None:
            return {
                "status": "error",
                "message": "Must provide index for make_parallel action",
            }

        if index < 0 or index >= len(state.workflow.blocks):
            return {
                "status": "error",
                "message": f"Index {index} out of range. "
                f"Workflow has {len(state.workflow.blocks)} blocks (0-indexed).",
            }

        target_block = state.workflow.blocks[index]

        # Check block type - only regular Block can be parallelized
        if isinstance(target_block, (SourceBlock, SinkBlock, ScoreBlock)):
            return {
                "status": "error",
                "message": f"Cannot parallelize {type(target_block).__name__}. "
                "Only processing blocks can be parallelized.",
            }

        if isinstance(target_block, ParallelBlock):
            return {
                "status": "error",
                "message": f"Block at index {index} is already parallelized.",
            }

        # Verify it's actually a Block instance
        if not isinstance(target_block, Block):
            return {
                "status": "error",
                "message": f"Cannot parallelize {type(target_block).__name__}. "
                "Only processing blocks can be parallelized.",
            }

        # Extract parallel config from block_config
        config = block_config or {}
        try:
            parallel_block = make_parallel(
                target_block,
                max_workers=config.get("max_workers"),
                chunk_size=config.get("chunk_size", 1),
                ordered=config.get("ordered", True),
                error_handling=config.get("error_handling", "skip"),
            )
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to parallelize block: {e}",
            }

        # ParallelBlock wraps BlockBase but isn't a subclass; workflow.add()
        # accepts Any, so this is safe at runtime
        state.workflow.blocks[index] = parallel_block  # type: ignore[call-overload]
        state.validated = False
        state.inputs_set = False

        return {
            "status": "success",
            "message": f"Parallelized {target_block.name} at index {index}",
            "workflow": _format_workflow(state.workflow),
        }

    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. "
            "Valid actions: create, add_block, remove_block, list_blocks, "
            "validate, clear, show, make_parallel",
        }


def _manage_workflows_impl(
    action: str,
    name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Implementation of manage_workflows logic.

    Args:
        action: One of "save", "load", "list", "remove".
        name: Workflow name (required for save, load, remove).
        overwrite: If True, overwrite an existing workflow on save.

    Returns:
        Status message and workflow information.
    """
    state = get_global_state()

    if action == "save":
        if state.workflow is None:
            return {
                "status": "error",
                "message": "No workflow exists. Use build_workflow(action='create') "
                "first.",
            }

        if name is None:
            return {
                "status": "error",
                "message": "Must provide 'name' for save action.",
            }

        try:
            state.registry.register(name, state.workflow, overwrite=overwrite)
            return {
                "status": "success",
                "message": (
                    f"Workflow registered as '{name}'. List registered "
                    "workflows with the 'list' action."
                ),
            }
        except (ValueError, WorkflowValidationError) as e:
            return {
                "status": "error",
                "message": str(e),
            }

    elif action == "load":
        if name is None:
            return {
                "status": "error",
                "message": "Must provide 'name' for load action.",
            }

        try:
            workflow = state.registry.load(name)
            state.workflow = workflow
            state.validated = True
            state.inputs_set = False
            return {
                "status": "success",
                "message": f"Workflow '{name}' loaded.",
                "workflow": _format_workflow(workflow),
            }
        except KeyError:
            return {
                "status": "error",
                "message": (
                    f"Workflow '{name}' not found. List registered workflows "
                    "with the 'list' action."
                ),
            }
        except WorkflowValidationError as e:
            return {
                "status": "error",
                "message": str(e),
            }

    elif action == "list":
        try:
            workflows = state.registry.list()
            return {
                "status": "success",
                "message": "Registered workflows:",
                "workflows": workflows.to_string(index=False),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    elif action == "remove":
        if name is None:
            return {
                "status": "error",
                "message": "Must provide 'name' for remove action.",
            }

        try:
            state.registry.remove(name)
            return {
                "status": "success",
                "message": f"Workflow '{name}' removed.",
            }
        except KeyError:
            return {
                "status": "error",
                "message": f"Workflow '{name}' not found.",
            }

    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. "
            "Valid actions: save, load, list, remove",
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
                "message": f"Missing required input(s): {e}",
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

        # Make sure the final block is a sink
        if not isinstance(state.workflow.blocks[-1], MoleculeSinkBlock):
            state.workflow.add(MoleculeSinkBlock())

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
            if output_file:
                state.last_output_file = str(output_path)
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
    """Build a cheminformatics workflow step-by-step. Always approach this in plan
    mode. IMPORTANT: If possible ALWAYS use this tool to solve problems and ALWAYS
    use manage_workflows to list saved workflows and ask if one should be used if
    appropriate for the problem.

    NOTE: required 'input_text', 'input_files', and optional mutable parameters can
    be set at instantiation as part of the 'block_config'. ONLY `input_text` and
    `input_files` can be set with the `get_inputs` and `set_inputs` actions in the
    `run_workflow` tool.

    IMPORTANT: Do not use the "clear" or "create" actions after optimizing a workflow.

    If adding a ScoreBlock YOU MUST ask users to confirm which ScoreBlock and if it
    should be minimized or maximized.

    CRITICAL: Workflows can only include steps with available blocks. You may have to
    run multiple workflows if an intermediate step is not included.

    Args:
        action: One of "create", "add_block", "remove_block", "list_blocks",
            "validate", "clear", "show", "make_parallel".
        block_type: Block class name (e.g., "ConformerGenerationBlock").
        block_config: Block initialization parameters. For "make_parallel" action,
            this specifies parallel execution options: max_workers, chunk_size,
            ordered, error_handling.
        rdkit_method: For RDKitBlock, the method path
            (e.g., "rdkit.Chem.Descriptors.MolWt").
        index: Position for insert/remove/make_parallel operations.

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
def manage_workflows(
    action: str,
    name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Manage saved workflows in the registry.

    Use this tool to save, load, list, and remove workflows. Workflows must be
    built and validated with the build_workflow tool before saving.

    IMPORTANT: The following rules MUST be followed:
    1. ALWAYS offer to save a workflow after optimizing it.
    2. Ask what name to give a workflow before saving.
    3. Confirm with users that it is OK to remove or overwrite.

    Args:
        action: One of "save", "load", "list", "remove".
        name: Workflow name (required for save, load, remove).
        overwrite: If True, overwrite an existing workflow on save.

    Returns:
        Status message and workflow information.
    """
    return _manage_workflows_impl(
        action=action,
        name=name,
        overwrite=overwrite,
    )


@mcp.tool
def run_workflow(
    action: str,
    inputs: dict[str, str] | None = None,
    input_file: str | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Set required input files and input text and execute a validated workflow.

    IMPORTANT: If you forget to add properties when you ran the 'build_workflow'
    'add_block' action they can be set with this tool using the 'get_inputs' and
    'set_inputs' actions using the `run_workflow` tool.

    YOU MUST "show" the workflow structure before using the run_workflow tool. It may
    include some annotations and added context but the text graphic MUST INCLUDE:
    1. A header with a fun name for the workflow. Be funny (e.g., a pun) and use emojis.
    2. The output from the "build_workflow" tool "show" action. DO NOT remove text.
    3. A quote from a famous chemist. It should be related to the workflow subject.

    Note: YOU MUST validate a workflow before using the run_workflow tool.

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


def _run_optimization(
    optimizer: Optimizer,
    n_trials: int,
    direction: str,
    timeout: float | None,
) -> None:
    """Run optimization in background thread.

    Args:
        optimizer: The optimizer instance.
        n_trials: Number of optimization trials.
        direction: Optimization direction ("maximize" or "minimize").
        timeout: Optional timeout in seconds.
    """
    optimizer.optimize(
        n_trials=n_trials,
        direction=direction,
        timeout=timeout,
        show_progress_bar=False,
        n_jobs=1,
    )


def _optimize_workflow_impl(
    action: str,
    n_trials: int | None = None,
    input_file: str | None = None,
    inputs: dict[str, str] | None = None,
    direction: str = "maximize",
    timeout: float | None = None,
) -> dict[str, Any]:
    """Implementation of optimize_workflow logic.

    Args:
        action: One of "start", "status", "get_best_params", "set_best_params",
            "cancel".
        n_trials: Number of optimization trials (required for "start").
        input_file: Path to input molecule file (required for "start").
        inputs: Optional required inputs (files/text) for workflow blocks.
        direction: "maximize" or "minimize" (default: "maximize").
        timeout: Optional timeout in seconds.

    Returns:
        Optimization status or results.
    """
    state = get_global_state()

    if action == "start":
        # Validation checks
        if state.workflow is None:
            return {
                "status": "error",
                "message": "No workflow exists. "
                "Use build_workflow(action='create') first.",
            }

        if not state.validated:
            return {
                "status": "error",
                "message": "Workflow not validated. "
                "Use build_workflow(action='validate') first.",
            }

        # Check workflow ends with ScoreBlock
        if not state.workflow.blocks or not isinstance(
            state.workflow.blocks[-1], ScoreBlock
        ):
            last_block = (
                type(state.workflow.blocks[-1]).__name__
                if state.workflow.blocks
                else "None"
            )
            return {
                "status": "error",
                "message": f"Workflow must end with ScoreBlock for optimization. "
                f"Current ending: {last_block}. IMPORTANT: Ask user which score to "
                "use.",
            }

        # Check for optimizable parameters
        if not state.workflow.get_params():
            return {
                "status": "error",
                "message": "Workflow has no optimizable parameters",
            }

        # Check if optimization already running
        if (
            state.optimization_future is not None
            and not state.optimization_future.done()
        ):
            return {
                "status": "error",
                "message": "Optimization already in progress. "
                "Use action='status' to check progress.",
            }

        # Validate required parameters
        if n_trials is None:
            return {
                "status": "error",
                "message": "Must provide n_trials for start action",
            }

        if input_file is None:
            return {
                "status": "error",
                "message": "Must provide input_file for start action",
            }

        input_path = Path(input_file)
        if not input_path.is_file():
            return {
                "status": "error",
                "message": f"Input file not found: {input_file}",
            }

        # Validate direction early
        if direction not in ("maximize", "minimize"):
            return {
                "status": "error",
                "message": f"Invalid direction: {direction}. "
                "Must be 'maximize' or 'minimize'.",
            }

        # Set inputs if provided
        if inputs is not None:
            try:
                state.workflow.set_required_input(inputs)
                state.inputs_set = True
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to set inputs: {e}",
                }

        # Check if required inputs are set
        required = state.workflow.get_required_input()
        if required and not state.inputs_set:
            return {
                "status": "error",
                "message": "Required inputs not set. "
                "Provide inputs parameter or use run_workflow(action='set_inputs').",
                "required_inputs": {k: str(v) for k, v in required.items()},
            }

        # Start optimization
        try:
            state.optimizer = Optimizer(state.workflow, input_path)
            state.optimization_error = None
            executor = get_executor()
            state.optimization_future = executor.submit(
                _run_optimization,
                state.optimizer,
                n_trials,
                direction,
                timeout,
            )
            return {
                "status": "started",
                "message": f"Optimization started with {n_trials} trials",
                "n_trials": n_trials,
                "direction": direction,
            }
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start optimization: {e}",
            }

    elif action == "status":
        if state.optimization_future is None:
            return {
                "status": "no_optimization",
                "message": "No optimization running",
            }

        if state.optimization_future.done():
            try:
                state.optimization_future.result()
                if state.optimizer is None:
                    return {
                        "status": "failed",
                        "error": "Optimizer was not initialized",
                    }
                return {
                    "status": "completed",
                    "message": "Optimization completed successfully",
                    "best_params": state.optimizer.best_params,
                    "best_score": state.optimizer.best_score,
                }
            except Exception as e:
                state.optimization_error = str(e)
                return {
                    "status": "failed",
                    "error": str(e),
                }

        # Future exists but not done - it's either pending or running
        # Include progress info if available
        progress: dict[str, Any] = {
            "status": "running",
            "message": "Optimization in progress",
        }
        if state.optimizer is not None and state.optimizer._study is not None:
            study = state.optimizer._study
            trials = study.trials
            progress["completed_trials"] = len(trials)
            if trials:
                # Get best score so far from completed trials
                completed = [
                    t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if completed:
                    progress["best_score_so_far"] = study.best_value
                    progress["best_params_so_far"] = dict(study.best_params)
        return progress

    elif action == "get_best_params":
        if state.optimizer is None:
            return {
                "status": "error",
                "message": "No optimization has been run",
            }

        if state.optimization_future and not state.optimization_future.done():
            return {
                "status": "error",
                "message": "Optimization still running",
            }

        try:
            return {
                "status": "success",
                "best_params": state.optimizer.best_params,
                "best_score": state.optimizer.best_score,
            }
        except RuntimeError as e:
            return {
                "status": "error",
                "message": str(e),
            }

    elif action == "set_best_params":
        if state.optimizer is None:
            return {
                "status": "error",
                "message": "No optimization has been run",
            }

        if state.optimization_future and not state.optimization_future.done():
            return {
                "status": "error",
                "message": "Optimization still running",
            }

        try:
            state.optimizer.set_best_params()
            return {
                "status": "success",
                "message": "Best parameters applied to workflow",
                "best_params": state.optimizer.best_params,
            }
        except RuntimeError as e:
            return {
                "status": "error",
                "message": str(e),
            }

    elif action == "cancel":
        if state.optimization_future is None:
            return {
                "status": "error",
                "message": "No optimization running",
            }

        if state.optimization_future.done():
            return {
                "status": "error",
                "message": "Optimization already completed",
            }

        cancelled = state.optimization_future.cancel()
        if cancelled:
            return {
                "status": "success",
                "message": "Optimization cancelled",
            }
        else:
            return {
                "status": "error",
                "message": "Could not cancel optimization (already running)",
            }

    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. "
            "Valid actions: start, status, get_best_params, set_best_params, cancel",
        }


@mcp.tool
def optimize_workflow(
    action: str,
    n_trials: int | None = None,
    input_file: str | None = None,
    inputs: dict[str, str] | None = None,
    direction: str = "maximize",
    timeout: float | None = None,
) -> dict[str, Any]:
    """Optimize a workflow using Bayesian optimization.

    Offer to use this tool if a user doesn't like the output of a workflow.

    IMPORTANT: The following rules MUST BE FOLLOWED:
    1. YOU MUST validate a workflow before using the run_workflow tool.
    2. YOU MUST confirm selection of n_trials. Typically 30 is good but users
       may want more or less.
    3. YOU MUST ask if any steps should be parallel before using "start" action.
    4. After starting optimization, DO NOT poll status automatically. Only check
       status when the user explicitly asks for progress or results.
    5. NEVER use the any other tool (especailly build_workflow) while
       optimize workflow is running.

    YOU MUST "show" the workflow structure before using the "optimize_workflow" tool. It
    may include some annotations and added context but the text graphic MUST INCLUDE:
    1. A header with a fun name for the workflow. Be funny (e.g., a pun) and use emojis.
    2. The output from the "build_workflow" tool "show" action. DO NOT remove text.
    3. A quote from a famous chemist. It should be related to the workflow subject.


    Args:
        action: One of "start", "status", "get_best_params", "set_best_params",
            "cancel".
        n_trials: Number of optimization trials (required for "start").
        input_file: Path to input molecule file (required for "start").
        inputs: Optional required inputs (files/text) for workflow blocks.
        direction: "maximize" or "minimize" (default: "maximize").
        timeout: Optional timeout in seconds.

    Returns:
        Optimization status or results.
    """
    return _optimize_workflow_impl(
        action=action,
        n_trials=n_trials,
        input_file=input_file,
        inputs=inputs,
        direction=direction,
        timeout=timeout,
    )


if _PYMOL_AVAILABLE:

    @mcp.tool
    def view_structures(
        files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Open 3D structure files in PyMOL for visualization.

        IMPORTANT: Offer to use this tool after executing a workflow that
        generates 3D input (e.g., conformer generation, alignment, docking)

        Before calling this tool, YOU MUST:
        1. Confirm with the user which 3D output file(s) to open
        2. Ask if they want to include additional structure files (e.g., PDB)

        Args:
            files: List of file paths to open. If not provided, opens the last
                workflow output file. Can include additional structure files
                (PDB, SDF, MOL2, etc.) alongside workflow output.

        Returns:
            Status message indicating success or failure.
        """
        state = get_global_state()

        if not workflow_has_3d_blocks():
            return {
                "status": "error",
                "message": "Cannot view structures: workflow has no 3D blocks "
                "(ConformerGenerationBlock, MoleculeAlignBlock, etc.)",
            }

        # Determine files to open
        files_to_open: list[str] = []

        if files:
            files_to_open = files
        elif state.last_output_file:
            files_to_open = [state.last_output_file]
        else:
            return {
                "status": "error",
                "message": "No files specified and no workflow output available. "
                "Either provide files or execute a workflow first.",
            }

        # Validate files exist
        missing = [f for f in files_to_open if not Path(f).exists()]
        if missing:
            return {
                "status": "error",
                "message": f"Files not found: {missing}",
            }

        try:
            open_pymol_session(*files_to_open)
            return {
                "status": "success",
                "message": f"Opened {len(files_to_open)} file(s) in PyMOL",
                "files": files_to_open,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to open PyMOL: {e}",
            }
