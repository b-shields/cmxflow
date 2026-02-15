# cmxflow

Composable cheminformatics workflows with Bayesian optimization.

**cmxflow** is a Python framework for building and optimizing cheminformatics pipelines. Chain together molecular operations as blocks, then let Bayesian optimization find the best parameters for your task.

## Two Usage Modes

cmxflow is designed to work both as:

1. **An Agentic Tool** — via [MCP](https://modelcontextprotocol.io/) server, allowing LLM agents like Claude to build and optimize workflows conversationally
2. **A Programmatic API** — for direct Python usage in scripts and notebooks

## Installation

### Base environment

```bash
conda config --set solver libmamba
conda env create -f conda.yml
conda activate cmxflow
poetry install
```

### MCP server (for use with Claude)

```bash
claude mcp add cmxflow -- cmxflow-mcp
```

See [Using with Claude](using-with-claude.md) for details.

### Optional: PyMOL

Required only for 3D structure visualization (`view_structures` MCP tool):

```bash
conda install -c conda-forge pymol-open-source
```

## Quick Start

### Programmatic API

```python
from cmxflow import Workflow
from cmxflow.sources import MoleculeSourceBlock
from cmxflow.operators import MoleculeSimilarityBlock, RDKitBlock
from cmxflow.sinks import MoleculeSinkBlock

# Build a workflow (everything is composable, add as many blocks as you want)
workflow = Workflow()
workflow.add(
    MoleculeSourceBlock(),                                  # Reader
    MoleculeSimilarityBlock(queries="crystal_ligand.sdf"),  # 2D Similarity
    RDKitBlock("rdkit.Chem.Descriptors.MolWt"),             # Arbitrary rdkit method
    MoleculeSinkBlock()                                     # Writer
)

# Run it
workflow("molecules.sdf", "results.sdf")
```

### Agentic (e.g., via Claude)

> I need build a ligand-based virtual screening workflow. I'm not sure if 2D or 3D is better. Can you optimize two workflows? The benchmark is in benchmark.csv with hits labeled in the active column and the query is in reference.sdf.

See the [Block Catalog](blocks.md) for all available blocks or the
[`examples/basic_usage.ipynb`](https://github.com/b-shields/cmxflow/blob/main/examples/basic_usage.ipynb)
notebook for a full tutorial covering similarity search, optimization, and parallel execution.
