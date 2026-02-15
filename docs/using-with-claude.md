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

## Example Prompts

### Similarity Search

> Read molecules from screen.sdf, compute 2D similarity to queries.sdf using ECFP4 fingerprints, and write the top 100 to hits.sdf.

### Virtual Screening with Optimization

> Build a workflow that reads molecules from library.sdf, computes Tanimoto similarity to reference.sdf, and scores by enrichment against actives.smi. Optimize it with 30 trials.

### 3D Conformer Generation

> Generate 3D conformers for molecules in mols.sdf, align them to crystal_pose.sdf, and save the aligned structures.

### Property Filtering

> Filter molecules.sdf to keep only those with MolWt < 500, LogP between 1 and 5, and no PAINS alerts. Write the passing molecules to filtered.sdf.

### Docking

> Dock the molecules in ligands.sdf into receptor.pdb and save the docked poses.

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
