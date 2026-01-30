"""Automatic scoring blocks for molecular workflow optimization."""

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from cmxflow.block import ScoreBlock
from cmxflow.cmxmol import Mol as CmxMol

logger = logging.getLogger(__name__)


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
    """ScoreBlock configured for enrichment-based molecular scoring.

    Uses molecule properties as features and computes enrichment AUC as the
    optimization metric. Non-numeric properties are automatically filtered.
    """

    def __init__(self) -> None:
        """Initialize with molecule pooler and enrichment AUC metric."""
        super().__init__(
            pooler=mol_to_dataframe,
            metric=enrichment_auc,
        )
