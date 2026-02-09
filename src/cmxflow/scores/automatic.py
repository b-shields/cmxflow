"""Automatic scoring blocks for molecular workflow optimization."""

import logging
from collections.abc import Iterator
from typing import Any, Callable

import numpy as np
import pandas as pd
from rdkit import Chem

from cmxflow.block import ScoreBlock
from cmxflow.cmxmol import Mol as CmxMol

logger = logging.getLogger(__name__)

# Map blocks to a dict of scores and rank ascending bool
BLOCK_SCORE_MAP: dict[str, dict[str, bool]] = {
    "Molecule3DSimilarity": {"similarity_3d": False},
    "ParallelMolecule3DSimilarity": {"similarity_3d": False},
    "MoleculeAlign": {"alignment_shape_similarity": False},
    "ParallelMoleculeAlign": {"alignment_shape_similarity": False},
    "Molecule2DSimilarity": {"max_similarity": False},
    "ParallelMolecule2DSimilarity": {"max_similarity": False},
    "MoleculeDock": {
        "docking_score": True,
        "docking_vinardo": True,
        "docking_ec": False,
    },
    "ParallelMoleculeDock": {
        "docking_score": True,
        "docking_vinardo": True,
        "docking_ec": False,
    },
}


def mol_to_dataframe(mols: Iterator[Chem.Mol | CmxMol]) -> pd.DataFrame:
    """Convert molecule iterator to DataFrame with numeric properties only.

    Extracts all properties from molecules and creates a DataFrame. Non-numeric
    columns are dropped and logged to the debug channel.

    Args:
        mols: Iterator of RDKit Mol or CmxMol objects.

    Returns:
        DataFrame with numeric molecule properties as columns.
    """
    records: list[dict[str, Any]] = []

    for mol in mols:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        records.append(props)

    if not records:
        logger.debug("No molecules found, returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if df.empty:
        logger.debug("No properties found, returning empty DataFrame")
        return df

    # Identify numeric and non-numeric columns
    numeric_cols: list[str] = []
    dropped_cols: list[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            dropped_cols.append(col)

    if dropped_cols:
        logger.debug(f"Dropping non-numeric columns: {dropped_cols}")

    if numeric_cols:
        logger.debug(f"Retaining numeric columns: {numeric_cols}")
    else:
        logger.warning("No numeric columns found in molecule properties")

    return df[numeric_cols]


def enrichment_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute area under the enrichment curve.

    The enrichment curve plots fraction of hits found (y-axis) vs fraction of
    library screened (x-axis) when molecules are ranked by score (descending).
    AUC of 0.5 indicates random performance, 1.0 indicates perfect ranking.

    Args:
        scores: Predicted scores for ranking (higher = better).
        labels: Binary labels (1 = hit, 0 = non-hit).

    Returns:
        Area under the enrichment curve (0 to 1).
    """
    if len(scores) == 0 or len(labels) == 0:
        return 0.0

    total_hits = np.sum(labels)
    if total_hits == 0:
        logger.debug("No hits in labels, returning 0.0")
        return 0.0

    # Sort by scores descending (higher score = ranked first)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    # Compute cumulative hits at each position
    cumulative_hits = np.cumsum(sorted_labels)

    # Normalize to fractions
    n = len(scores)
    x = np.arange(1, n + 1) / n  # Fraction of library screened
    y = cumulative_hits / total_hits  # Fraction of hits found

    # Compute AUC using trapezoidal rule
    auc: float = float(np.trapezoid(y, x))
    return auc


class EnrichmentScoreBlock(ScoreBlock):
    """ScoreBlock for enrichment-based molecular scoring.

    Uses molecule properties as features and computes enrichment AUC as the
    optimization metric. Non-numeric properties are automatically filtered.
    """

    def __init__(
        self,
        pooler: Callable[[Iterator[Any]], pd.DataFrame] = mol_to_dataframe,
        metric: Callable[[np.ndarray, np.ndarray], float] = enrichment_auc,
        **kwargs,
    ) -> None:
        """Initialize with scoring configuration.

        Args:
            score_column: Name of the column to use as the score.
            target_column: Name of the column containing target labels.
            pooler: Function to convert iterator to DataFrame.
            metric: Function to compute metric from scores and labels.
        """
        super().__init__(input_text=["target"])
        self.pooler = pooler
        self.metric = metric
        self.set_inputs(**kwargs)
        self._score_properties: dict[str, bool] = {}
        self._best_score: float = -2.0
        self._best_score_name: str | None = None
        self._best_score_uid: tuple[str, ...] | None = None

    def _set_score_properties(self, *args: Any) -> None:
        for arg in args:
            for key, value in BLOCK_SCORE_MAP.get(arg.name, {}).items():
                self._score_properties[key] = bool(value)

    def objective(self, iter: Iterator[Chem.Mol | CmxMol]) -> float:
        """Compute enrichment AUC for the given molecules.

        Args:
            iter: Iterator of molecules with properties.

        Raises:
            KeyError: Block does not have allowed properties for ranking.
            ValueError: Score column not found in data.

        Returns:
            Enrichment AUC score.
        """
        if not self._score_properties:
            raise KeyError("EnrichmentScoreBlock has no allowed properties.")

        df = self.pooler(iter)

        if df.empty:
            logger.warning("Empty DataFrame, returning 0.0")
            return 0.0

        target_col = self.input_text["target"]
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not in input data.")

        best_score: float = -2.0
        best_score_name: str | None = None
        for score_col, ascending in self._score_properties.items():
            if score_col not in df.columns:
                raise ValueError(f"Score column '{score_col}' not found in data.")
            scores = df[score_col].to_numpy()
            if ascending:
                scores *= -1
            labels = df[target_col].to_numpy()
            metric = self.metric(scores, labels)
            if metric > best_score:
                best_score = metric
                best_score_name = score_col

        if best_score > self._best_score:
            self._best_score = best_score
            self._best_score_name = best_score_name
            self._best_score_uid = self._uid

        self._cache[self._uid] = df

        return best_score

    def forward(self, mol: Chem.Mol | CmxMol) -> Chem.Mol | CmxMol:
        """Add workflow_score property during normal (non-optimization) execution.

        Args:
            mol: Input molecule.

        Returns:
            Molecule with workflow_score property added if score column exists.
        """
        if self._best_score_name is None:
            return mol

        score = mol.GetPropsAsDict().get(self._best_score_name)
        if score is not None:
            mol.SetDoubleProp("workflow_score", float(score))

        return mol


class AverageScoreBlock(ScoreBlock):
    """ScoreBlock that computes average of a molecular property.

    Uses the same pooler approach as EnrichmentScoreBlock to convert
    molecules to a DataFrame, then computes the mean of the specified
    property column.
    """

    def __init__(
        self,
        pooler: Callable[[Iterator[Any]], pd.DataFrame] = mol_to_dataframe,
        **kwargs,
    ) -> None:
        """Initialize with pooler configuration.

        Args:
            pooler: Function to convert molecule iterator to DataFrame.
        """
        super().__init__(input_text=["property"])
        self.pooler = pooler
        self.set_inputs(**kwargs)

    def objective(self, iter: Iterator[Chem.Mol | CmxMol]) -> float:
        """Compute average of the specified property.

        Args:
            iter: Iterator of molecules with properties.

        Returns:
            Mean value of the specified property column.
        """
        df = self.pooler(iter)
        property_col = self.input_text["property"]

        if df.empty:
            logger.warning("Empty DataFrame, returning 0.0")
            return 0.0

        if property_col not in df.columns:
            raise ValueError(f"Property column '{property_col}' not found in data")

        return float(df[property_col].mean())

    def forward(self, mol: Chem.Mol | CmxMol) -> Chem.Mol | CmxMol:
        """Pass through molecule unchanged.

        Args:
            mol: Input molecule.

        Returns:
            The same molecule, unchanged.
        """
        return mol
