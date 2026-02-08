"""Molecule clustering block using streaming leader algorithm."""

import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from cmxflow.operators.base import MoleculeBlock
from cmxflow.parameter import Categorical, Continuous

logger = logging.getLogger(__name__)


class RepresentativeClusterBlock(MoleculeBlock):
    """Assign molecules to clusters using streaming leader clustering.

    For each molecule, computes an ECFP4 fingerprint and compares it
    against all existing cluster representatives via Tanimoto similarity.
    If the best similarity is >= threshold, the molecule joins that cluster.
    Otherwise, a new cluster is created with this molecule as representative.

    All molecules pass through (annotator, not filter). Each molecule is
    annotated with cluster_id, cluster_representative, and cluster_similarity.

    This block cannot be parallelized because it relies on shared mutable
    state (the representative cache).

    Attributes:
        _representatives: Fingerprints of cluster representative molecules.
        _representative_smiles: SMILES strings of cluster representatives.
        _generator: Morgan fingerprint generator (ECFP4).
    """

    def __init__(self, **kwargs: str) -> None:
        """Initialize the representative cluster block."""
        super().__init__(name="RepresentativeCluster")
        self.mutable(
            Continuous("threshold", default=0.4, low=0.05, high=0.95),
            Categorical("scaffold", default=True, choices=[True, False]),
        )
        self.set_inputs(**kwargs)
        self._representatives: list[DataStructs.ExplicitBitVect] = []
        self._representative_smiles: list[str] = []
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )

    def __getstate__(self) -> dict:
        """Get state for pickling, excluding unpicklable fingerprint generator."""
        state = self.__dict__.copy()
        state.pop("_generator", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle, recreating fingerprint generator."""
        self.__dict__.update(state)
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )

    def _forward(self, mol: Chem.Mol) -> Chem.Mol:
        """Assign a molecule to a cluster and annotate it.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            The same molecule annotated with cluster_id,
            cluster_representative, and cluster_similarity.
        """
        threshold: float = self.params["threshold"].get()
        use_scaffold: bool = self.params["scaffold"].get()

        if use_scaffold:
            scaffold = GetScaffoldForMol(mol)
            fp = self._generator.GetFingerprint(scaffold)
            smiles = Chem.MolToSmiles(scaffold)
        else:
            fp = self._generator.GetFingerprint(mol)
            smiles = Chem.MolToSmiles(mol)

        if self._representatives:
            similarities = DataStructs.BulkTanimotoSimilarity(fp, self._representatives)
            max_sim = max(similarities)
            best_idx = similarities.index(max_sim)
        else:
            max_sim = 1.0
            best_idx = -1

        if max_sim >= threshold and best_idx >= 0:
            cluster_id = best_idx
            similarity = max_sim
        else:
            cluster_id = len(self._representatives)
            self._representatives.append(fp)
            self._representative_smiles.append(smiles)
            similarity = 1.0
            logger.debug("New cluster %d: %s", cluster_id, smiles)

        mol.SetIntProp("cluster_id", cluster_id)
        mol.SetProp("cluster_representative", self._representative_smiles[cluster_id])
        mol.SetDoubleProp("cluster_similarity", similarity)
        return mol

    def reset_cache(self) -> None:
        """Clear the representative cache for a new optimization iteration."""
        self._representatives.clear()
        self._representative_smiles.clear()
