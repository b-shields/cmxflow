"""State management for the cmxflow MCP server."""

from dataclasses import dataclass

from cmxflow import Workflow
from cmxflow.operators import (
    ConformerGenerationBlock,
    EnumerateStereoBlock,
    Molecule3DSimilarityBlock,
    MoleculeAlignBlock,
    MoleculeSimilarityBlock,
    RDKitBlock,
)
from cmxflow.scores import EnrichmentScoreBlock
from cmxflow.sinks import MoleculeSinkBlock
from cmxflow.sources import MoleculeSourceBlock


@dataclass
class WorkflowState:
    """State container for a workflow being built.

    Attributes:
        workflow: The current workflow being built.
        validated: Whether the workflow has passed validation.
        inputs_set: Whether required inputs have been set.
    """

    workflow: Workflow | None = None
    validated: bool = False
    inputs_set: bool = False


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
    }


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
            "Generate 3D conformers using RDKit's ETKDGv3 algorithm. "
            "Molecules must have fully specified stereochemistry."
        ),
        "EnumerateStereoBlock": (
            "Enumerate all possible stereoisomers of molecules. "
            "1:N transform that yields multiple outputs per input."
        ),
        "Molecule3DSimilarityBlock": (
            "Compute 3D shape similarity between molecules and a reference. "
            "Requires 3D conformers."
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
            "Compute enrichment AUC for scoring molecules. "
            "Used for optimization workflows."
        ),
    }
