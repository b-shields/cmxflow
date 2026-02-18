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

> Read molecules from screen.sdf.gz, compute 2D similarity to queries.sdf using ECFP4 fingerprints, and write the top 100 to similar.csv.

```
⏺ Let me check available workflows and blocks first.

⏺ cmxflow - manage_workflows (MCP)(action: "list")
  ⎿  {
       "status": "success",
       "message": "Registered workflows:",
     … +6 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "list_blocks")
  ⎿  {
       "status": "success",
       "blocks": {
     … +37 lines (ctrl+o to expand)

⏺ Now let me build the workflow:

⏺ cmxflow - build_workflow (MCP)(action: "create")
  ⎿  {
       "status": "success",
       "message": "Created new workflow with MoleculeSourceBlock",
     … +3 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "MoleculeSimilarityBlock", block_config: {"fingerprint":"ECFP4"})
  ⎿  {
       "status": "success",
       "message": "Added Molecule2DSimilarity",
     … +8 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "PropertyHeadBlock", block_config:
                                {"property":"max_similarity","count":100})
  ⎿  {
       "status": "success",
       "message": "Added PropertyHead",
     … +11 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "MoleculeSinkBlock")
  ⎿  {
       "status": "success",
       "message": "Added MoleculeSink",
     … +12 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "validate")
  ⎿  {
       "status": "success",
       "message": "Workflow is valid",
     … +12 lines (ctrl+o to expand)

⏺ Workflow is valid. Let me set the query file input, then execute.

  ---
  🔍 Top-100 ECFP4 Similarity Screener 🧪

  ┌────────────────┐
  │ MoleculeSource │  ← screen.sdf.gz
  └────────────────┘
           ↓
  ┌──────────────────────┐   ┌─────────────────┐
  │  Molecule2DSimilarity│ ← │ queries: [FILE] │  ← queries.sdf
  │  ECFP4 / Tanimoto    │   └─────────────────┘
  └──────────────────────┘
           ↓
  ┌──────────────────────┐
  │  PropertyHead        │  top 100 by max_similarity
  └──────────────────────┘
           ↓
  ┌────────────────┐
  │  MoleculeSink  │  → similar.csv
  └────────────────┘

  "The more that you read, the more things you will know." — Not a chemist, but relevant to screening libraries.

  (A real one: "No one has yet produced a single example of the conversion of one species into another." — J.B.S. Haldane, though he'd approve of finding similar molecules.)*

⏺ cmxflow - run_workflow (MCP)(action: "set_inputs", inputs: {"1.file@queries":"queries.sdf"})
  ⎿  {
       "status": "success",
       "message": "Inputs set successfully"
     }

⏺ cmxflow - run_workflow (MCP)(action: "execute", input_file: "screen.sdf.gz", output_file: "similar.csv")
  ⎿  {
       "status": "error",
       "message": "Execution failed: 'int' object has no attribute 'strip'"
     }

⏺ The count may need to be set as a text input. Let me fix that.

⏺ cmxflow - run_workflow (MCP)(action: "set_inputs", inputs: {"2.text@property":"max_similarity","2.text@count":"100"})
  ⎿  {
       "status": "success",
       "message": "Inputs set successfully"
     }

⏺ cmxflow - run_workflow (MCP)(action: "execute", input_file: "screen.sdf.gz", output_file: "similar.csv")
  ⎿  {
       "status": "success",
       "message": "Workflow executed successfully",
     … +2 lines (ctrl+o to expand)

⏺ Done. The workflow:

  1. Read molecules from screen.sdf.gz
  2. Computed ECFP4 (Morgan, r=2, 2048 bits) Tanimoto similarity against all queries in queries.sdf, annotating each molecule with max_similarity
  3. Selected the top 100 by max_similarity
  4. Wrote results to similar.csv

  Would you like to save this workflow for future use?
```

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
