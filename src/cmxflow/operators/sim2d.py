"""2D fingerprint similarity search block."""

from pathlib import Path
from typing import Callable, cast

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.DataStructs import ExplicitBitVect

from cmxflow.operators.base import MoleculeBlock
from cmxflow.parameter import Categorical, Integer
from cmxflow.sources.reader import read_molecules

# Type alias for fingerprint functions
FingerprintFunc = Callable[[Chem.Mol], ExplicitBitVect]

# Similarity metric functions
SIMILARITY_METRICS: dict[str, Callable] = {
    "tanimoto": DataStructs.TanimotoSimilarity,
    "dice": DataStructs.DiceSimilarity,
    "cosine": DataStructs.CosineSimilarity,
    "sokal": DataStructs.SokalSimilarity,
    "russel": DataStructs.RusselSimilarity,
}


class MoleculeSimilarityBlock(MoleculeBlock):
    """Block for 2D fingerprint similarity searching.

    Computes fingerprint similarity between input molecules and a set of
    query molecules. Attaches maximum similarity score and most similar
    query index as molecule properties.

    Required Inputs:
        queries (file): Path to query molecule file (SDF, SMILES, etc.).

    Mutable Parameters:
        fingerprint_type: Fingerprint algorithm (morgan, rdkit, maccs, atom_pair,
            topological_torsion).
        similarity_metric: Similarity function (tanimoto, dice, cosine, sokal, russel).
        radius: Morgan fingerprint radius (1-4).
        nbits: Fingerprint bit length (512-4096).

    Example:
        workflow.add(MoleculeSimilarityBlock())
        workflow.set_required_input({
            "1.file@queries": "reference_ligands.sdf",
        })
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the similarity search block."""
        super().__init__(name="Molecule2DSimilarity", input_files=["queries"])

        # Register mutable parameters
        self.mutable(
            Categorical(
                "fingerprint_type",
                default="morgan",
                choices=[
                    "morgan",
                    "rdkit",
                    "maccs",
                    "atom_pair",
                    "topological_torsion",
                ],
            ),
            Categorical(
                "similarity_metric",
                default="tanimoto",
                choices=list(SIMILARITY_METRICS.keys()),
            ),
            Integer("radius", default=2, low=1, high=4),
            Integer("nbits", default=2048, low=512, high=4096),
        )
        self.set_inputs(**kwargs)

        # Lazy-loaded query fingerprints
        self._query_fingerprints: list[ExplicitBitVect] | None = None
        self._query_names: list[str] | None = None

    def reset_cache(self) -> None:
        """Reset cached query fingerprints for a new optimization iteration."""
        self._query_fingerprints = None
        self._query_names = None

    def _get_fingerprint_func(self) -> FingerprintFunc:
        """Get the fingerprint function based on current parameters.

        Returns:
            Fingerprint function that takes a Mol and returns a fingerprint.
        """
        fp_type = self.params["fingerprint_type"].get()
        radius = self.params["radius"].get()
        nbits = self.params["nbits"].get()

        if fp_type == "morgan":
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=nbits
            )
            return cast(FingerprintFunc, generator.GetFingerprint)
        elif fp_type == "rdkit":
            generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nbits)
            return cast(FingerprintFunc, generator.GetFingerprint)
        elif fp_type == "maccs":
            return cast(FingerprintFunc, rdMolDescriptors.GetMACCSKeysFingerprint)
        elif fp_type == "atom_pair":
            generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits)
            return cast(FingerprintFunc, generator.GetFingerprint)
        elif fp_type == "topological_torsion":
            generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                fpSize=nbits
            )
            return cast(FingerprintFunc, generator.GetFingerprint)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

    def _load_query_fingerprints(self) -> None:
        """Load and compute fingerprints for query molecules.

        Reads molecules from the query file and computes fingerprints
        using the current fingerprint parameters. Results are cached
        in _query_fingerprints and _query_names.

        Raises:
            ValueError: If no valid molecules are found in the query file.
        """
        query_path = Path(self.input_files["queries"])
        fp_func = self._get_fingerprint_func()

        self._query_fingerprints = []
        self._query_names = []

        for i, mol in enumerate(read_molecules(query_path)):
            fp = fp_func(mol)
            self._query_fingerprints.append(fp)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"query_{i}"
            self._query_names.append(name)

        if not self._query_fingerprints:
            raise ValueError(f"No valid molecules found in query file: {query_path}")

    def _forward(self, mol: Chem.Mol) -> Chem.Mol:
        """Compute similarity between input molecule and query molecules.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            Input molecule with added properties:
                - max_similarity: Maximum similarity score to any query.
                - most_similar_query: Name/index of the most similar query.
        """
        # Lazy load query fingerprints
        if self._query_fingerprints is None:
            self._load_query_fingerprints()
        assert self._query_fingerprints is not None
        assert self._query_names is not None

        fp_func = self._get_fingerprint_func()
        similarity_func = SIMILARITY_METRICS[self.params["similarity_metric"].get()]

        mol_fp = fp_func(mol)

        # Compute similarity to all queries
        max_sim = 0.0
        best_query_idx = 0

        for i, query_fp in enumerate(self._query_fingerprints):
            sim = similarity_func(mol_fp, query_fp)
            if sim > max_sim:
                max_sim = sim
                best_query_idx = i

        # Attach properties to molecule
        mol.SetDoubleProp("max_similarity", max_sim)
        mol.SetProp("most_similar_query", self._query_names[best_query_idx])

        return mol
