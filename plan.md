# Plan: cmxflow.scores.automatic Module

## Overview
Create a new module for automatic scoring of molecular workflows using enrichment metrics. The module provides a ready-to-use `EnrichmentScoreBlock` that converts molecule iterators to DataFrames and computes enrichment curve metrics.

## Files to Create
1. `src/cmxflow/scores/__init__.py` - Package init with exports
2. `src/cmxflow/scores/automatic.py` - Main module

## Module: automatic.py

### Imports
```python
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from cmxflow.block import ScoreBlock
from cmxflow.cmxmol import Mol as CmxMol
```

### Function: `mol_to_dataframe(mols: Iterator[Chem.Mol | CmxMol]) -> pd.DataFrame`

**Purpose:** Convert iterator of molecules to DataFrame, keeping only numeric columns.

**Logic:**
1. Iterate through molecules, extract properties via `GetPropsAsDict()`
2. Build list of property dictionaries
3. Create DataFrame from list
4. Identify numeric columns using `pd.api.types.is_numeric_dtype()`
5. Log dropped non-numeric columns to debug channel
6. Log retained numeric columns to debug channel
7. Return DataFrame with only numeric columns

**Edge cases:**
- Empty iterator → return empty DataFrame
- No numeric columns → return empty DataFrame (log warning)
- Mixed types in same column → pandas will infer, may become object type

### Function: `enrichment_auc(scores: np.ndarray, labels: np.ndarray) -> float`

**Purpose:** Compute area under the enrichment curve.

**Definition:** Enrichment curve plots:
- X-axis: Fraction of library screened (0 to 1)
- Y-axis: Fraction of hits found (0 to 1)

When molecules are ranked by score (descending), the enrichment curve shows how quickly we find hits. AUC of 0.5 = random, AUC of 1.0 = perfect (all hits ranked first).

**Logic:**
1. Sort indices by scores (descending - higher score = better)
2. Reorder labels by sorted indices
3. Compute cumulative sum of hits at each position
4. Normalize: x = position/total, y = cumulative_hits/total_hits
5. Compute AUC using trapezoidal rule (np.trapz)

**Edge cases:**
- No hits (all labels=0) → return 0.0 or handle gracefully
- All hits (all labels=1) → return 1.0
- Empty arrays → return 0.0

### Class: `EnrichmentScoreBlock(ScoreBlock)`

**Purpose:** Pre-configured ScoreBlock for enrichment-based scoring.

**Implementation:**
```python
class EnrichmentScoreBlock(ScoreBlock):
    """ScoreBlock configured for enrichment-based molecular scoring."""

    def __init__(self) -> None:
        """Initialize with molecule pooler and enrichment AUC metric."""
        super().__init__(
            pooler=mol_to_dataframe,
            metric=enrichment_auc,
        )
```

Only overrides `__init__` - inherits all other behavior from ScoreBlock.

## Package: __init__.py

```python
"""Scoring blocks for workflow optimization."""

from cmxflow.scores.automatic import EnrichmentScoreBlock, enrichment_auc, mol_to_dataframe

__all__ = [
    "EnrichmentScoreBlock",
    "enrichment_auc",
    "mol_to_dataframe",
]
```

## Type Signatures

```python
def mol_to_dataframe(mols: Iterator[Chem.Mol | CmxMol]) -> pd.DataFrame: ...

def enrichment_auc(scores: np.ndarray, labels: np.ndarray) -> float: ...

class EnrichmentScoreBlock(ScoreBlock):
    def __init__(self) -> None: ...
```

## Testing Considerations
- Test mol_to_dataframe with molecules having mixed property types
- Test enrichment_auc with known enrichment scenarios (perfect, random, worst)
- Test empty inputs
- Test integration with ScoreBlock caching behavior
