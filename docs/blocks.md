# Block Catalog

All available blocks at a glance. See the [API Reference](api/operators.md) for full details.

## Source Blocks

| Block | Description |
|-------|-------------|
| `MoleculeSourceBlock` | Read molecules from SDF, SDF.gz, Mol2, SMILES, CSV, or Parquet files |

## Operator Blocks

| Block | Description | Inputs |
|-------|-------------|----------------|
| `MoleculeStandardizeBlock` | Standardize molecules |  |
| `RDKitBlock` | Apply any RDKit method to molecules | `method` (text) |
| `SubstructureFilterBlock` | Filter by SMARTS or catalogs (PAINS, BRENK, etc.) | `query` (text), `mode` (text) |
| `PropertyFilterBlock` | Filter molecules by property | `filters` (text) |
| `MoleculeDeduplicateBlock` | Remove duplicate molecules |  |
| `PropertyHeadBlock` | Select top N molecules by property | `property` (text), `count` (text) |
| `PropertyTailBlock` | Select bottom N molecules by property | `property` (text), `count` (text) |
| `MoleculeSimilarityBlock` | 2D fingerprint similarity search | `queries` (file) |
| `Molecule3DSimilarityBlock` | 3D similarity search | `query` (file) |
| `IonizeMoleculeBlock` | Generate pH-dependent ionization states |  |
| `EnumerateStereoBlock` | Enumerate all stereoisomers |  |
| `ConformerGenerationBlock` | Generate 3D conformers |  |
| `MoleculeAlignBlock` | Align molecules to 3D reference | `query` (file) |
| `MoleculeDockBlock` | Dock into protein binding pocket | `receptor` (file), `site_reference` (file) |
| `RepresentativeClusterBlock` | Leader clustering |  |

## Sink Blocks

| Block | Description |
|-------|-------------|
| `MoleculeSinkBlock` | Write molecules to SDF, SDF.gz, SMILES, CSV, or Parquet files |

## Score Blocks

| Block | Description | Inputs |
|-------|-------------|--------|
| `EnrichmentScoreBlock` | Enrichment AUC for virtual screening | `target` (text) |
| `AverageScoreBlock` | Mean of a molecular property | `property` (text) |
| `ShapeOverlayScoreBlock` | Average 3D shape similarity | `query` (file) |
| `ClusterScoreBlock` | Cluster quality from representative clustering |  |

## Block Types

Workflows are built from four types:

- **SourceBlock** — produces molecules from a file
- **Block** — transforms molecules (1:1 or N:M)
- **SinkBlock** — writes molecules to a file (terminal)
- **ScoreBlock** — computes an optimization objective (terminal)

Every workflow starts with a `SourceBlock` and ends with either a `SinkBlock` or `ScoreBlock`.

## Setting Block Inputs

Blocks that require file paths or text configuration support two ways to set inputs:

**Python API** — pass at instantiation as keyword arguments:
```python
MoleculeSimilarityBlock(queries="reference.sdf")
SubstructureFilterBlock(query="PAINS BRENK", mode="remove")
```

**Agent / MCP** — Uses index-keyed strings via the `run_workflow set_inputs` action, because the workflow is already built and blocks cannot be reinstantiated mid-conversation:
```
"1.file@queries": "reference.sdf"
"2.text@query":   "PAINS BRENK"
```
The key format is `"<block_index>.<type>@<name>"` where type is `file` or `text`. Block indices start at 0 (source block) and increment for each block added.
