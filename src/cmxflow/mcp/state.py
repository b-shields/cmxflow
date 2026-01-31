"""State management for the cmxflow MCP server."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cmxflow import Workflow

if TYPE_CHECKING:
    from cmxflow.opt import Optimizer
from cmxflow.operators import (
    ConformerGenerationBlock,
    EnumerateStereoBlock,
    Molecule3DSimilarityBlock,
    MoleculeAlignBlock,
    MoleculeSimilarityBlock,
    RDKitBlock,
)
from cmxflow.scores import EnrichmentScoreBlock, ShapeOverlayScoreBlock
from cmxflow.sinks import MoleculeSinkBlock
from cmxflow.sources import MoleculeSourceBlock


@dataclass
class WorkflowState:
    """State container for a workflow being built.

    Attributes:
        workflow: The current workflow being built.
        validated: Whether the workflow has passed validation.
        inputs_set: Whether required inputs have been set.
        optimizer: The optimizer instance for Bayesian optimization.
        optimization_future: Future for background optimization task.
        optimization_error: Error message if optimization failed.
        last_output_file: Path to the last workflow output file.
    """

    workflow: Workflow | None = None
    validated: bool = False
    inputs_set: bool = False
    optimizer: "Optimizer | None" = field(default=None)
    optimization_future: Future | None = field(default=None)
    optimization_error: str | None = field(default=None)
    last_output_file: str | None = field(default=None)


# Global state instance for persistence across tool calls
_global_state: WorkflowState | None = None

# Thread pool executor for background tasks
_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor.

    Returns:
        The shared ThreadPoolExecutor instance.
    """
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)
    return _executor


def get_global_state() -> WorkflowState:
    """Get or create the global workflow state.

    Returns:
        The current global workflow state.
    """
    global _global_state
    if _global_state is None:
        _global_state = WorkflowState()
    return _global_state


def reset_global_state() -> None:
    """Reset the global state to a fresh instance."""
    global _global_state
    _global_state = WorkflowState()


def get_available_blocks() -> dict[str, type]:
    """Return registry of all available block types.

    Returns:
        Dictionary mapping block class names to their types.
    """
    return {
        # Sources
        "MoleculeSourceBlock": MoleculeSourceBlock,
        # Sinks
        "MoleculeSinkBlock": MoleculeSinkBlock,
        # Operators
        "ConformerGenerationBlock": ConformerGenerationBlock,
        "EnumerateStereoBlock": EnumerateStereoBlock,
        "Molecule3DSimilarityBlock": Molecule3DSimilarityBlock,
        "MoleculeAlignBlock": MoleculeAlignBlock,
        "MoleculeSimilarityBlock": MoleculeSimilarityBlock,
        "RDKitBlock": RDKitBlock,
        # Scores
        "EnrichmentScoreBlock": EnrichmentScoreBlock,
        "ShapeOverlayScoreBlock": ShapeOverlayScoreBlock,
    }


def workflow_has_3d_blocks() -> bool:
    """Check if the current workflow contains 3D-generating blocks.

    Returns:
        True if workflow contains ConformerGenerationBlock, MoleculeAlignBlock,
        Molecule3DSimilarityBlock, or ShapeOverlayScoreBlock.
    """
    state = get_global_state()
    if state.workflow is None:
        return False

    _3d_block_types = (
        ConformerGenerationBlock,
        MoleculeAlignBlock,
        Molecule3DSimilarityBlock,
        ShapeOverlayScoreBlock,
    )

    for block in state.workflow.blocks:
        if isinstance(block, _3d_block_types):
            return True
    return False


def get_block_descriptions() -> dict[str, str]:
    """Return descriptions for all available block types.

    Returns:
        Dictionary mapping block class names to their descriptions.
    """
    return {
        # Sources
        "MoleculeSourceBlock": (
            "Source block for reading molecules from various file formats "
            "(SDF, SMILES, CSV, Parquet, Mol2)."
        ),
        # Sinks
        "MoleculeSinkBlock": (
            "Sink block for writing molecules to various file formats "
            "(SDF, SMILES, CSV, Parquet)."
        ),
        # Operators
        "ConformerGenerationBlock": (
            "Generate 3D conformers using RDKit. YOU MUST have the "
            "EnumerateStereoBlock before the ConformerGenerationBlock."
        ),
        "EnumerateStereoBlock": (
            "Enumerate all possible stereoisomers of molecules. "
            "Required preprocessing step before conformer generation."
        ),
        "Molecule3DSimilarityBlock": (
            "Compute 3D shape similarity between molecules and a reference. YOU "
            "MUST have the MoleculeAlignBlock before the Molecule3DSimilarityBlock."
        ),
        "MoleculeAlignBlock": (
            "Align molecules to a reference structure using 3D coordinates."
        ),
        "MoleculeSimilarityBlock": (
            "Compute 2D fingerprint similarity between molecules and a reference."
        ),
        "RDKitBlock": (
            "Apply an arbitrary RDKit method to molecules. Provide the method "
            "as a string path (e.g., 'rdkit.Chem.Descriptors.MolWt')."
        ),
        # Scores
        "EnrichmentScoreBlock": (
            "Compute enrichment AUC for scoring molecules. Used to optimize workflows."
        ),
        "ShapeOverlayScoreBlock": (
            "Score shape similarity with references to optimize for good overlays. "
            "YOU MUST have MoleculeAlignBlock before the ShapeOverlayScoreBlock."
        ),
    }
