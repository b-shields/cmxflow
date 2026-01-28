# Plan: Implement sim3d.py with Molecule3DSimilarityBlock

## Overview
Create a new operator module for computing 3D molecular similarity using RDKit's shape and USR-based methods. The block will compare input molecules against query references (both with pre-existing 3D conformers) and attach similarity scores as molecule properties.

## Available RDKit 3D Similarity Methods

### Shape-based (rdShapeHelpers)
- **ShapeTanimotoDist**: Tanimoto distance measuring shape overlap (0=identical, 1=no overlap)
- **ShapeTverskyIndex**: Asymmetric similarity with alpha/beta parameters
- **ShapeProtrudeDist**: Measures how much one shape protrudes from another

### USR-based (rdMolDescriptors)
- **USR**: Ultrafast Shape Recognition - encodes 3D shape as a 12-element descriptor
- **USRCAT**: USR with CREDO Atom Types - adds pharmacophoric information (60 elements)

## Implementation Details

### File: `src/cmxflow/operators/sim3d.py`

```python
"""3D molecular similarity block."""

from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdShapeHelpers

from cmxflow.operators.base import MoleculeBlock
from cmxflow.parameter import Categorical, Continuous
from cmxflow.sources.reader import read_molecules
```

### Class: Molecule3DSimilarityBlock

**Inherits from**: `MoleculeBlock`

**Input files**: `queries` - file containing reference molecules with 3D conformers

**Mutable parameters**:
| Parameter | Type | Default | Choices/Range |
|-----------|------|---------|---------------|
| `method` | Categorical | `"shape_tanimoto"` | `["shape_tanimoto", "shape_tversky", "usr", "usrcat"]` |
| `tversky_alpha` | Continuous | `1.0` | `[0.0, 1.0]` |
| `tversky_beta` | Continuous | `1.0` | `[0.0, 1.0]` |

### Methods

#### `__init__(self)`
- Call `super().__init__(input_files=["queries"])`
- Set `self.name = "Molecule3DSimilarity"`
- Register mutable parameters
- Initialize lazy-loaded caches: `_query_mols`, `_query_names`, `_query_descriptors`

#### `_load_queries(self)`
- Load query molecules from input file
- Validate each has 3D conformers
- For USR/USRCAT methods, precompute descriptors and cache them

#### `_compute_shape_similarity(self, mol, conf_id, ref, ref_conf_id) -> float`
- For `shape_tanimoto`: Use `rdShapeHelpers.ShapeTanimotoDist()`, convert distance to similarity (1 - dist)
- For `shape_tversky`: Use `rdShapeHelpers.ShapeTverskyIndex()` with alpha/beta params

#### `_compute_usr_similarity(self, mol, conf_id, ref_descriptor) -> float`
- Compute USR or USRCAT descriptor for mol conformer
- Use `rdMolDescriptors.GetUSRScore()` to compare with cached reference descriptor

#### `_forward(self, mol: Chem.Mol) -> Chem.Mol`
- Lazy load queries
- For each (mol_conformer, query, query_conformer) combination:
  - Compute similarity using selected method
  - Track best similarity and corresponding query
- Attach properties:
  - `similarity_3d`: Best similarity score
  - `most_similar_query_3d`: Name of most similar query
  - `similarity_3d_method`: Method used

#### `check_input(self, arg: Any) -> bool`
- Call `super().check_input(arg)`
- Verify molecule has at least one 3D conformer

### Output Properties
| Property | Type | Description |
|----------|------|-------------|
| `similarity_3d` | float | Best 3D similarity score (0-1, higher=more similar) |
| `most_similar_query_3d` | str | Name of most similar reference molecule |
| `similarity_3d_method` | str | Method used for comparison |
| `similarity_3d_conf_id` | int | Conformer ID that gave best similarity |

## Key Design Decisions

1. **Conformer handling**: Compare all conformer pairs between input and reference, report the best match
2. **Similarity normalization**: Convert all metrics to 0-1 scale where 1=most similar
3. **USR descriptor caching**: Pre-compute USR/USRCAT descriptors for queries to avoid redundant computation
4. **No alignment**: Assume molecules are pre-aligned or use alignment-independent methods (USR/USRCAT)

## Files to Create/Modify
- **Create**: `src/cmxflow/operators/sim3d.py`
- **Modify**: `src/cmxflow/operators/__init__.py` (add export)

## Testing
- Test with shape_tanimoto on aligned conformers
- Test USR/USRCAT on unaligned conformers (should still work)
- Verify properties are correctly attached
- Test with multiple conformers per molecule
- Test parallel execution preserves properties
