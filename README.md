# cmxflow 🧪

Composable cheminformatics workflows with Bayesian optimization.

## Overview 🔬

**cmxflow** is a Python framework for building and optimizing cheminformatics pipelines. Chain together molecular operations as blocks, then let Bayesian optimization find the best parameters for your task.

### Two Usage Modes ⚗️

cmxflow is designed to work both as:

1. **An Agentic Tool** - via MCP (Model Context Protocol) server, allowing LLM agents to build and optimize workflows conversationally
2. **A Programmatic API** - for direct Python usage in scripts and notebooks

## Block Types 🧬

Workflows are built from four types of blocks:

| Block Type | Purpose |
|------------|---------|
| **SourceBlock** | Read molecules from files (SDF, SMILES, CSV, Parquet) |
| **Block** | Transform molecules (1:1 or N:M) |
| **SinkBlock** | Write molecules to files |
| **ScoreBlock** | Compute optimization objective |

### Example Operators 💊

| Block | Purpose |
|-------|---------|
| `RDKitBlock` | Apply any RDKit method (descriptors, transformations) |
| `PropertyFilterBlock` | Filter molecules by property conditions |
| `PropertyHeadBlock` | Select top N molecules by property |
| `PropertyTailBlock` | Select bottom N molecules by property |
| `MoleculeSimilarityBlock` | Compute 2D fingerprint similarity |
| `EnumerateStereoBlock` | Enumerate all stereoisomers |
| `ConformerGenerationBlock` | Generate 3D conformers (ETKDGv3) |
| `MoleculeAlignBlock` | Align molecules to 3D reference |
| `MoleculeDockBlock` | Dock into protein binding pocket |

### Example Score Blocks 📊

| ScoreBlock | Purpose |
|------------|---------|
| `EnrichmentScoreBlock` | Enrichment AUC for virtual screening |
| `AverageScoreBlock` | Mean of a molecular property |
| `ShapeOverlayScoreBlock` | Average 3D shape similarity |

## Features 🚀

- **Composable Pipelines** - Chain blocks with `workflow.add()`
- **Bayesian Optimization** - Find optimal parameters via Optuna
- **Parallel Execution** - `make_parallel()` for compute-intensive blocks
- **Mutable Parameters** - Categorical, Integer, and Continuous types
- **Serialization** - `save_workflow()` and `load_workflow()` for persistence
- **MCP Server** - Agentic workflow building via `build_workflow`, `run_workflow`, `optimize_workflow`

## Getting Started 📖

See [`examples/basic_usage.ipynb`](examples/basic_usage.ipynb) for a complete tutorial covering:

- Building your first workflow
- 2D similarity search
- Mutable parameters and optimization
- Parallel execution
- Analyzing results with Optuna

The tutorial uses the ABL1 kinase benchmark from the wonderful [DUD-E](http://dude.docking.org/) database.

## Installation 🛠️

Development install:

```bash
# Build base environment
conda config --set solver libmamba
conda env create -f conda.yml
conda activate cmxflow

# Install cmxflow
poetry install
pre-commit install
```

## License 📄

MIT
