# Using with Claude

cmxflow includes an [MCP](https://modelcontextprotocol.io/) server that lets Claude build, run, and optimize cheminformatics workflows conversationally.

## Setup

Add the cmxflow MCP server to Claude Code:

```bash
claude mcp add cmxflow -- cmxflow-mcp
```

This gives Claude access to five tools:

| Tool | Description |
|------|-------------|
| `build_workflow` | Create and configure workflows step-by-step |
| `run_workflow` | Set inputs and execute a validated workflow |
| `optimize_workflow` | Bayesian optimization of workflow parameters |
| `manage_workflows` | Save, load, list, and remove workflows |
| `view_structures` | Open 3D output in PyMOL |

## Setting Block Inputs

Blocks that require file paths or text configuration support two ways to set inputs:

**Python API** — pass at instantiation as keyword arguments:
```python
MoleculeSimilarityBlock(queries="reference.sdf")
SubstructureFilterBlock(query="PAINS BRENK", mode="remove")
```

**Agent / MCP** — Claude uses index-keyed strings via the `run_workflow set_inputs` action, because the workflow is already built and blocks cannot be reinstantiated mid-conversation:
```
"1.file@queries": "reference.sdf"
"2.text@query":   "PAINS BRENK"
```
The key format is `"<block_index>.<type>@<name>"` where type is `file` or `text`. Block indices start at 0 (source block) and increment for each block added.

## Example Prompts

### Similarity Search

> Read molecules from screen.sdf.gz, compute 2D similarity to queries.sdf using ECFP4 fingerprints, and write the top 100 to similar.csv.

### Virtual Screening with Optimization

> I need to build a ligand-based virtual screening workflow. I'm not sure if 2D or 3D is better. Can you optimize two workflows? The benchmark is in benchmark.csv with hits labeled in the active column and the query is in reference.sdf.

### 3D Conformer Generation

> Generate 3D conformers for the molecules in mols.sdf, align them to crystal_pose.sdf, and save the aligned structures.

And if you don't like the alignments...

> These alignments are no good. Can you optimize for shape overlay?

### Filtering / Library Preparation

> How many of the molecules in library.parquet pass Lipinski's rules?

> Filter PAINS and cluster the remainder of the molecules.

### Docking

> Collect all molecules with a carboxylic acid in actives.csv and dock them against receptor.pdb with crystal_ligand.sdf as a reference.

## What Claude Can Do

With the MCP server, Claude can:

- **Build workflows** from natural language descriptions, selecting the right blocks and connecting them
- **Set parameters** like fingerprint type, similarity cutoffs, and filter conditions
- **Run workflows** end-to-end, handling file I/O automatically
- **Optimize parameters** using Bayesian optimization to maximize or minimize a score
- **Save and reload workflows** for later reuse
- **Visualize 3D results** by opening structures in PyMOL

## Tips

- Be specific about file paths — Claude needs to know where your input files are
- For optimization, specify whether to maximize or minimize the score
- Use `make_parallel` for compute-intensive blocks (conformer generation, docking)
- Ask Claude to show the workflow before running to verify the pipeline
