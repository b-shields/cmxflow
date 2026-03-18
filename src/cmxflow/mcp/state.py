"""State management for the cmxflow MCP server."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cmxflow import Workflow

if TYPE_CHECKING:
    from cmxflow.opt import Optimizer
from cmxflow.operators import (
    ConformerGenerationBlock,
    EnumerateStereoBlock,
    IonizeMoleculeBlock,
    Molecule3DSimilarityBlock,
    MoleculeAlignBlock,
    MoleculeDeduplicateBlock,
    MoleculeDockBlock,
    MoleculeSimilarityBlock,
    MoleculeStandardizeBlock,
    PropertyFilterBlock,
    PropertyHeadBlock,
    PropertyTailBlock,
    RDKitBlock,
    RepresentativeClusterBlock,
    SubstructureFilterBlock,
)
from cmxflow.scores import (
    AverageScoreBlock,
    ClusterScoreBlock,
    EnrichmentScoreBlock,
    ShapeOverlayScoreBlock,
)
from cmxflow.sinks import MoleculeSinkBlock
from cmxflow.sources import MoleculeSourceBlock
from cmxflow.utils.serial import WorkflowRegistry


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
        registry: Workflow registry.
    """

    workflow: Workflow | None = None
    validated: bool = False
    inputs_set: bool = False
    optimizer: "Optimizer | None" = field(default=None)
    optimization_future: Future | None = field(default=None)
    optimization_error: str | None = field(default=None)
    last_output_file: str | None = field(default=None)
    registry: WorkflowRegistry = field(default_factory=WorkflowRegistry)


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
        "IonizeMoleculeBlock": IonizeMoleculeBlock,
        "Molecule3DSimilarityBlock": Molecule3DSimilarityBlock,
        "MoleculeAlignBlock": MoleculeAlignBlock,
        "MoleculeDeduplicateBlock": MoleculeDeduplicateBlock,
        "MoleculeSimilarityBlock": MoleculeSimilarityBlock,
        "MoleculeStandardizeBlock": MoleculeStandardizeBlock,
        "RDKitBlock": RDKitBlock,
        "RepresentativeClusterBlock": RepresentativeClusterBlock,
        "SubstructureFilterBlock": SubstructureFilterBlock,
        "PropertyFilterBlock": PropertyFilterBlock,
        "PropertyHeadBlock": PropertyHeadBlock,
        "PropertyTailBlock": PropertyTailBlock,
        "MoleculeDockBlock": MoleculeDockBlock,
        # Scores
        "AverageScoreBlock": AverageScoreBlock,
        "ClusterScoreBlock": ClusterScoreBlock,
        "EnrichmentScoreBlock": EnrichmentScoreBlock,
        "ShapeOverlayScoreBlock": ShapeOverlayScoreBlock,
    }


def workflow_has_3d_blocks() -> bool:
    """Check if the current workflow contains 3D-generating blocks.

    Returns:
        True if workflow contains ConformerGenerationBlock, MoleculeAlignBlock,
        Molecule3DSimilarityBlock, ShapeOverlayScoreBlock or MoleculeDockBlock.
    """
    state = get_global_state()
    if state.workflow is None:
        return False

    _3d_block_types = (
        ConformerGenerationBlock,
        MoleculeAlignBlock,
        Molecule3DSimilarityBlock,
        ShapeOverlayScoreBlock,
        MoleculeDockBlock,
    )
    _3d_block_keywords = ("conf", "dock", "align", "3d", "shape")

    for block in state.workflow.blocks:
        if isinstance(block, _3d_block_types):
            return True
        for keyword in _3d_block_keywords:
            if keyword in block.name.lower():
                return True
    return False


def get_param_info(workflow: Workflow) -> list[dict[str, Any]]:
    """Extract parameter info with owning block names.

    Args:
        workflow: Workflow to extract parameters from.

    Returns:
        List of dicts with name, type, current value, options, and block name.
    """
    info: list[dict[str, Any]] = []
    for block in workflow.blocks:
        for param in block.get_params().values():
            info.append(
                {
                    "name": param.name,
                    "type": type(param).__name__,
                    "current": str(param),
                    "options": str(param.options),
                    "block": block.name,
                }
            )
    return info


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
        "MoleculeDeduplicateBlock": (
            "Remove duplicate molecules from the stream based on canonical SMILES. "
            "Keeps the first occurrence and discards subsequent duplicates. "
            "IMPORTANT: This block cannot be parallelized because it requires "
            "shared state. Place it after MoleculeStandardizeBlock for best results."
        ),
        "MoleculeSimilarityBlock": (
            "Compute 2D fingerprint similarity between molecules and a reference. "
            "This block adds the 'max_similarity' property to all input molecules."
        ),
        "MoleculeStandardizeBlock": (
            "Standardize molecules for drug discovery: disconnects metals, "
            "normalizes valence/aromaticity, strips salts (keeps largest fragment), "
            "and neutralizes charges. Set `canonicalize_tautomers=True` to also pick "
            "the heuristically preferred tautomer. IMPORTANT: This should typically "
            "be the first processing step after the source block."
        ),
        "RDKitBlock": (
            "Apply an arbitrary RDKit method to molecules. Provide the method "
            "as a string path (e.g., 'rdkit.Chem.Descriptors.MolWt'). IMPORTANT: "
            "only computes properties or modifies molecules (e.g., AddHs). (must "
            "use other blocks to act on properties)."
        ),
        "RepresentativeClusterBlock": (
            "Assign molecules to clusters using streaming leader clustering "
            "with ECFP4 Tanimoto similarity. Annotates each molecule with "
            "cluster_id, cluster_representative SMILES, and cluster_similarity. "
            "Set scaffold=True to cluster by Murcko scaffold instead of whole "
            "molecule. IMPORTANT: This block cannot be parallelized because it "
            "requires shared state. Results are order-dependent."
        ),
        "SubstructureFilterBlock": (
            "Filter molecules by substructure using SMARTS patterns and/or "
            "built-in catalogs (e.g., PAINS, BRENK, NIH, ZINC, etc.). Provide a "
            "single 'query' with space-separated catalog names or SMARTS patterns "
            "(e.g., 'PAINS BRENK [#8;!$(O=C)]'). Mode is 'remove' by default. Set to "
            "'keep' to only return matching."
        ),
        "PropertyFilterBlock": (
            "Apply any number of numerical property filters to remove molecules. "
            "Filters are specified as ',' separated values (e.g., 200<=MolWt<500, "
            "logP<5, HBD==2, HBA!=0)"
        ),
        "PropertyHeadBlock": (
            "Get the top 'count' molecules from the input stream (ranked descending) "
            "by a specified 'property'. IMPORTANT the property must already be "
            "computed."
        ),
        "PropertyTailBlock": (
            "Get the bottom 'count' molecules from the input stream (ranked "
            "descending) by a specified 'property'. IMPORTANT the property must "
            "already be computed."
        ),
        "EnumerateStereoBlock": (
            "Enumerate all possible stereoisomers of molecules. IMPORTANT: This step "
            "should always come somewhere BEFORE a ConformerGenerationBlock."
        ),
        "IonizeMoleculeBlock": (
            "Generate pH-dependent ionization states. One molecule can produce "
            "multiple protonation variants. IMPORTANT: Place after "
            "MoleculeStandardizeBlock and before ConformerGenerationBlock."
        ),
        "ConformerGenerationBlock": (
            "Generate 3D conformers. IMPORTANT: This step should always come somewhere "
            "AFTER a EnumerateStereoBlock. It is a slow block and you should offer to "
            "make it parallel (recommend max_workers=4)."
        ),
        "MoleculeAlignBlock": (
            "Align molecules to a reference structure using 3D coordinates. IMPORTANT: "
            "This step should always come somewhere BEFORE a Molecule3DSimlarityBlock, "
            "MoleculeDockBlock, or ShapeOverlayScoreBlock. This block adds the "
            "'alignment_shape_similarity' property to all input molecules."
        ),
        "Molecule3DSimilarityBlock": (
            "Compute 3D shape similarity between molecules and a reference. IMPORTANT: "
            "This step should always come somewhere AFTER a MoleculeAlignBlock. This "
            "block adds the 'similarity_3d' property to all input molecules."
        ),
        "MoleculeDockBlock": (
            "Dock an aligned 3D molecule conformer against a protein receptor from a "
            ".pdb file. IMPORTANT: This step should always come somewhere AFTER a "
            "MoleculeAlignBlock. It is a slow block and you should offer to make it "
            "parallel (recommend max_workers=8). This block adds the 'docking_score' "
            "property to all input molecules."
        ),
        # Scores
        "AverageScoreBlock": (
            "Compute the average of a property as a score. Used to optimize physically "
            "meaningful scores (e.g., ligand affinity). IMPORTANT: Ask if the score "
            "should be minimized or maximized (e.g, 'docking_score' should be "
            "minimized)."
        ),
        "EnrichmentScoreBlock": (
            "Compute enrichment AUC for scoring molecules. Used to optimize virtual "
            "screening workflows. IMPORTANT: This score should be maximized."
        ),
        "ShapeOverlayScoreBlock": (
            "Score shape similarity with references to optimize for good overlays. "
            "IMPORTANT: This score should be maximized always come somewhere AFTER a "
            "MoleculeAlignBlock."
        ),
        "ClusterScoreBlock": (
            "Score clustering quality from RepresentativeClusterBlock. Computes "
            "mean intra-cluster similarity (excluding singletons) minus the "
            "fraction of singleton molecules. Use with RepresentativeClusterBlock "
            "upstream. IMPORTANT: This score should be maximized."
        ),
    }
