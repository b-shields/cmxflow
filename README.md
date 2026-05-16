# cmxflow 🧪

<!-- mcp-name: io.github.b-shields/cmxflow -->

[![Docs](https://img.shields.io/badge/docs-b--shields.github.io%2Fcmxflow-teal)](https://b-shields.github.io/cmxflow/)
[![CI](https://github.com/b-shields/cmxflow/actions/workflows/ci.yml/badge.svg)](https://github.com/b-shields/cmxflow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/b-shields/cmxflow/branch/main/graph/badge.svg)](https://codecov.io/gh/b-shields/cmxflow)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Composable cheminformatics workflows.

## Overview 🔬

**cmxflow** is a Python framework for building and optimizing cheminformatics pipelines. Chain together molecular operations as blocks, then let Bayesian optimization find the best parameters for your task.

**[Read the full documentation &rarr;](https://b-shields.github.io/cmxflow/)**

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
| `MoleculeStandardizeBlock` | Standardize molecules (metals, salts, charges, tautomers) |
| `MoleculeDeduplicateBlock` | Remove duplicate molecules by canonical SMILES |
| `RDKitBlock` | Apply any RDKit method (descriptors, transformations) |
| `SubstructureFilterBlock` | Filter by SMARTS patterns or catalogs (PAINS, BRENK, etc.) |
| `PropertyFilterBlock` | Filter molecules by property conditions |
| `PropertyHeadBlock` | Select top N molecules by property |
| `PropertyTailBlock` | Select bottom N molecules by property |
| `MoleculeSimilarityBlock` | Compute 2D fingerprint similarity |
| `Molecule3DSimilarityBlock` | Compute 3D shape similarity |
| `IonizeMoleculeBlock` | Generate pH-dependent ionization states |
| `EnumerateStereoBlock` | Enumerate all stereoisomers |
| `ConformerGenerationBlock` | Generate 3D conformers (ETKDGv3) |
| `MoleculeAlignBlock` | Align molecules to 3D reference |
| `MoleculeDockBlock` | Dock into protein binding pocket |
| `RepresentativeClusterBlock` | Cluster molecules by fingerprint similarity (leader algorithm) |

### Example Score Blocks 📊

| ScoreBlock | Purpose |
|------------|---------|
| `EnrichmentScoreBlock` | Enrichment AUC for virtual screening |
| `AverageScoreBlock` | Mean of a molecular property |
| `ShapeOverlayScoreBlock` | Average 3D shape similarity |
| `ClusterScoreBlock` | Cluster quality from representative clustering |

## Features 🚀

- **Composable Pipelines** - Chain blocks with `workflow.add()`
- **Bayesian Optimization** - Find optimal parameters via Optuna
- **Parallel Execution** - `make_parallel()` for compute-intensive blocks
- **Mutable Parameters** - Categorical, Integer, and Continuous types
- **Serialization** - `save_workflow()` and `load_workflow()` for persistence
- **MCP Server** - Agentic workflow building via `build_workflow`, `run_workflow`, `optimize_workflow`

## Environment Variables 🔧

| Variable | Default | Description |
|----------|---------|-------------|
| `CMXFLOW_WORKER_TIMEOUT` | `30` | Seconds to wait for a single parallel worker before treating it as failed. Set to `0` to disable the timeout. Applies to all `make_parallel()` and `@parallel` blocks. |

## Getting Started 📖

See [`examples/basic_usage.ipynb`](examples/basic_usage.ipynb) for a complete tutorial covering:

- Building your first workflow
- 2D similarity search
- Mutable parameters and optimization
- Parallel execution
- Analyzing results with Optuna

The tutorial uses the ABL1 kinase benchmark from the wonderful [DUD-E](http://dude.docking.org/) database.

## Installation 🛠️

```bash
pip install cmxflow
```

### MCP Server

To use cmxflow as an agentic tool with Claude Code:

```bash
claude mcp add cmxflow -- cmxflow-mcp
```

### Optional Dependencies

**PyMOL** — Required only for 3D structure visualization (`view_structures` MCP tool). Install via conda:

```bash
conda install -c conda-forge pymol-open-source
```

All other functionality works without PyMOL.

## Contributing & Releases 🤝

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR requirements, and [RELEASING.md](RELEASING.md) for the PyPI and MCP Registry release flow.

## License 📄

MIT
