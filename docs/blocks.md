# Block Catalog

All available blocks at a glance. See the [API Reference](api/operators.md) for full details.

## Source Blocks

| Block | Description |
|-------|-------------|
| `MoleculeSourceBlock` | Read molecules from SDF, SDF.gz, Mol2, SMILES, CSV, or Parquet files |

## Operator Blocks

| Block | Description | Key Params |
|-------|-------------|------------|
| `MoleculeStandardizeBlock` | Standardize molecules (metals, salts, charges, tautomers) | — |
| `MoleculeDeduplicateBlock` | Remove duplicates by canonical SMILES | — |
| `RDKitBlock` | Apply any RDKit method to molecules | `method` |
| `SubstructureFilterBlock` | Filter by SMARTS or catalogs (PAINS, BRENK, NIH, ZINC) | `query`, `mode` |
| `PropertyFilterBlock` | Filter molecules by property conditions | `filters` |
| `PropertyHeadBlock` | Select top N molecules by property | `property`, `count` |
| `PropertyTailBlock` | Select bottom N molecules by property | `property`, `count` |
| `MoleculeSimilarityBlock` | 2D fingerprint similarity search | `fingerprint_type`, `similarity_metric`, `radius`, `nbits` |
| `Molecule3DSimilarityBlock` | 3D shape similarity search | `method` |
| `IonizeMoleculeBlock` | Generate pH-dependent ionization states | `precision`, `max_variants` |
| `EnumerateStereoBlock` | Enumerate all stereoisomers | — |
| `ConformerGenerationBlock` | Generate 3D conformers (ETKDGv3) | `numConfs`, `pruneRmsThresh` |
| `MoleculeAlignBlock` | Align molecules to 3D reference | `alignment_method` |
| `MoleculeDockBlock` | Dock into protein binding pocket | `w_gauss1`, `w_repulsion`, `box_size` |
| `RepresentativeClusterBlock` | Leader clustering by fingerprint similarity | `threshold`, `scaffold` |

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
| `ClusterScoreBlock` | Cluster quality from representative clustering | — |

## Block Types

Workflows are built from four types:

- **SourceBlock** — produces molecules from a file
- **Block** — transforms molecules (1:1 or N:M)
- **SinkBlock** — writes molecules to a file (terminal)
- **ScoreBlock** — computes an optimization objective (terminal)

Every workflow starts with a `SourceBlock` and ends with either a `SinkBlock` or `ScoreBlock`.
