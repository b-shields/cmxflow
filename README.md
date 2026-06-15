# cmxflow 🧪

<!-- mcp-name: io.github.b-shields/cmxflow -->

[![Docs](https://img.shields.io/badge/docs-b--shields.github.io%2Fcmxflow-teal)](https://b-shields.github.io/cmxflow/)
[![CI](https://github.com/b-shields/cmxflow/actions/workflows/ci.yml/badge.svg)](https://github.com/b-shields/cmxflow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/b-shields/cmxflow/branch/main/graph/badge.svg)](https://codecov.io/gh/b-shields/cmxflow)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Build cheminformatics and computational chemistry pipelines with composable blocks. Tune end-to-end with Bayesian Optimization. Or ask an LLM agent to do it.

## Quick examples

### Prepare ligands for docking

```python
from cmxflow import Workflow
from cmxflow.sources import MoleculeSourceBlock
from cmxflow.operators import (
    MoleculeStandardizeBlock,
    IonizeMoleculeBlock,
    EnumerateStereoBlock,
    ConformerGenerationBlock,
)
from cmxflow.sinks import MoleculeSinkBlock

# Standardize → ionize (pH 6.4–8.4) → enumerate stereo → generate 3D conformers
workflow = Workflow()
workflow.add(
    MoleculeSourceBlock(),
    MoleculeStandardizeBlock(),
    IonizeMoleculeBlock(),
    EnumerateStereoBlock(),
    ConformerGenerationBlock(),
    MoleculeSinkBlock(),
)
workflow("library.smi", "prepared.sdf")
```

### Dock a congeneric series

Pure-Python docking. Free docking is the default (`index_poses=False`); scaffold-indexed mode caches poses by Bemis–Murcko scaffold for ~3× faster throughput on congeneric series with consistent pose alignment.

```python
from cmxflow import Workflow
from cmxflow.sources import MoleculeSourceBlock
from cmxflow.operators import ConformerGenerationBlock, MoleculeDockBlock
from cmxflow.sinks import MoleculeSinkBlock
from cmxflow.utils.parallel import make_parallel

workflow = Workflow()
workflow.add(
    MoleculeSourceBlock(),
    ConformerGenerationBlock(),
    make_parallel(
        MoleculeDockBlock(
            receptor="receptor.pdb",
            site_reference="crystal_ligand.sdf",
            index_poses=True,  # omit for free docking
        )
    ),
    MoleculeSinkBlock(),
)
workflow("library.smi", "docked.sdf")
```

### Tune a ligand-based virtual screen

```python
from cmxflow import Workflow
from cmxflow.sources import MoleculeSourceBlock
from cmxflow.operators import MoleculeSimilarityBlock
from cmxflow.scores import EnrichmentScoreBlock
from cmxflow.opt import Optimizer

# Rank a library by 2D similarity to a known active, then tune the
# fingerprint end-to-end to maximize enrichment AUC.
workflow = Workflow()
workflow.add(
    MoleculeSourceBlock(),
    MoleculeSimilarityBlock(queries="crystal_ligand.sdf"),
    EnrichmentScoreBlock(target="active"),
)

opt = Optimizer(workflow, "benchmark.csv")
opt.optimize(n_trials=30, direction="maximize")

print(f"Best enrichment AUC: {opt.best_score:.3f}")
print(opt.best_params)
# Best enrichment AUC: 0.836
# {'fingerprint_type': 'morgan', 'similarity_metric': 'sokal', 'radius': 2, 'nbits': 2545}
```

The four fingerprint parameters above are searched automatically — every block exposes its mutable parameters to the optimizer.

### Or build it conversationally via an LLM agent

```bash
claude mcp add cmxflow -- cmxflow-mcp
```

> *"How many of the molecules in library.csv pass Lipinski's rules?"*

> *"I need to build a ligand-based virtual screening workflow. I'm not sure if 2D or 3D is better. Can you optimize two workflows?"*

> *"Dock the molecules in hits.csv against receptor.pdb with crystal_ligand.sdf as a reference."*

The agent can build, run, *and* optimize workflows. See [Using with Claude](https://b-shields.github.io/cmxflow/using-with-claude/) for full transcripts.

## What's in the box

- 15+ blocks for sourcing, transforming, filtering, clustering, scoring, and docking molecules
- Bayesian optimization of pipeline parameters via [Optuna](https://optuna.org/)
- Parallel execution for compute-heavy blocks (conformer generation, docking)
- Workflow serialization for save / load / reuse
- An MCP server with five tools: `build_workflow`, `run_workflow`, `optimize_workflow`, `manage_workflows`, `view_structures`

## Install

```bash
pip install cmxflow
```

### MCP server

```bash
claude mcp add cmxflow -- cmxflow-mcp
```

### Optional: PyMOL

Required only for the `view_structures` MCP tool (3D visualization):

```bash
conda install -c conda-forge pymol-open-source
```

## Documentation

- [Docs site](https://b-shields.github.io/cmxflow/)
- [Block catalog](https://b-shields.github.io/cmxflow/blocks/)
- [Using with Claude](https://b-shields.github.io/cmxflow/using-with-claude/) — agent transcripts
- [`examples/basic_usage.ipynb`](examples/basic_usage.ipynb) — full tutorial
- [`examples/docking/docking.ipynb`](examples/docking/docking.ipynb) — docking walkthrough (ILS, scaffold-indexed, and template modes)

## Project

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md) and [RELEASING.md](RELEASING.md).
