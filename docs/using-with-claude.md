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

> I need to build a ligand-based virtual screening workflow. I'm not sure if 2D or 3D is better. Can you optimize two workflows? I want to see the results of 2D first. The benchmark is in "benchmark.parquet" with hits labeled in the "active" column and the query is in "reference.sdf".

```
⏺ Let me start by checking available blocks and existing workflows.

⏺ cmxflow - build_workflow (MCP)(action: "list_blocks")
  ⎿  {
       "status": "success",
       "blocks": {
     … +37 lines (ctrl+o to expand)

⏺ cmxflow - manage_workflows (MCP)(action: "list")
  ⎿  {
       "status": "success",
       "message": "Registered workflows:",
     … +6 lines (ctrl+o to expand)

⏺ There's an existing prep workflow but not what we need. Let me build the 2D similarity workflow first, then we'll do 3D after.

  For virtual screening benchmarking, we'll use EnrichmentScoreBlock which should be maximized. Does that sound right to you before I proceed?

  Let me build the 2D workflow now.

⏺ cmxflow - build_workflow (MCP)(action: "create")
  ⎿  {
       "status": "success",
       "message": "Created new workflow with MoleculeSourceBlock",
     … +3 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "MoleculeStandardizeBlock")
  ⎿  {
       "status": "success",
       "message": "Added MoleculeStandardize",
     … +4 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "MoleculeSimilarityBlock")
  ⎿  {
       "status": "success",
       "message": "Added Molecule2DSimilarity",
     … +10 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "EnrichmentScoreBlock", block_config:
                                {"property":"max_similarity","label_column":"active"})
  ⎿  {
       "status": "success",
       "message": "Added EnrichmentScoreBlock",
     … +13 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "validate")
  ⎿  {
       "status": "success",
       "message": "Workflow is valid",
     … +15 lines (ctrl+o to expand)

⏺ Now let me check the required inputs and set them, then get the optimizable parameters.

⏺ cmxflow - run_workflow (MCP)(action: "set_inputs", inputs: {"2.file@queries":"reference.sdf","3.text@target":"active"})
  ⎿  {
       "status": "success",
       "message": "Inputs set successfully"
     }

⏺ cmxflow - build_workflow (MCP)(action: "get_params")
  ⎿  {
       "status": "success",
       "params": [
     … +30 lines (ctrl+o to expand)

⏺ The workflow is validated and ready. Let me confirm the optimization setup with you:

  The optimizer will tune these 2D similarity parameters:
  - fingerprint_type: morgan, rdkit, maccs, atom_pair, topological_torsion
  - similarity_metric: tanimoto, dice, cosine, sokal, russel
  - radius: 1–4
  - nbits: 512–4096

  The enrichment score will be maximized. 30 trials is typical — does that work for you, or would you prefer more/fewer?

⏺ User answered Claude's questions:
  ⎿  · How many optimization trials for the 2D workflow? → 30 trials (Recommended)

⏺ Great, 30 trials it is. Let me kick off the 2D optimization.

  ---
  🔬 2D Fingerprint Virtual Screening Optimization 🔬

  MoleculeSource → MoleculeStandardize → Molecule2DSimilarity → EnrichmentScoreBlock

  "The good thing about science is that it's true whether or not you believe in it." — Carl Sagan (close enough to chemistry!)

  ---

⏺ cmxflow - optimize_workflow (MCP)(action: "start", n_trials: 30, input_file: "benchmark.parquet", inputs:
                                   {"2.file@queries":"reference.sdf","3.text@target":"active"}, direction: "maximize")
  ⎿  {
       "status": "started",
       "message": "Optimization started with 30 trials",
     … +3 lines (ctrl+o to expand)

⏺ The 2D optimization is running (30 trials, maximizing enrichment AUC). I'll check the status when you're ready — just let me know when you'd like to see results, or I can check progress at any point.

  While we wait, shall I go ahead and build the 3D workflow so it's ready to optimize next?

✻ Sautéed for 53s

❯ It should have been quick. How did the optimization turn out?

⏺ cmxflow - optimize_workflow (MCP)(action: "status")
  ⎿  {
       "status": "completed",
       "message": "Optimization completed successfully",
     … +8 lines (ctrl+o to expand)

⏺ The 2D optimization is done. Here are the results:

  Best Enrichment AUC: 0.65

  ┌───────────────────┬─────────────────────┐
  │     Parameter     │     Best Value      │
  ├───────────────────┼─────────────────────┤
  │ fingerprint_type  │ topological_torsion │
  ├───────────────────┼─────────────────────┤
  │ similarity_metric │ cosine              │
  ├───────────────────┼─────────────────────┤
  │ radius            │ 1                   │
  ├───────────────────┼─────────────────────┤
  │ nbits             │ 738                 │
  └───────────────────┴─────────────────────┘

  Interesting — topological torsion fingerprints with cosine similarity came out on top over the more common Morgan/Tanimoto combo. A score of 0.65 is decent but leaves room for improvement.

  Want me to set the best parameters and save this workflow, then build and optimize the 3D workflow to see if we can do better?

❯ yes, save it and build the 3D workflow

⏺ cmxflow - optimize_workflow (MCP)(action: "set_best_params")
  ⎿  {
       "status": "success",
       "message": "Best parameters applied to workflow",
     … +7 lines (ctrl+o to expand)

⏺ User answered Claude's questions:
  ⎿  · What name would you like for the saved 2D workflow? → 2d-virtual-screen

⏺ cmxflow - manage_workflows (MCP)(action: "save", name: "2d-virtual-screen")
  ⎿  {
       "status": "success",
       "message": "Workflow registered as '2d-virtual-screen'. List registered workflows with the 'list' action."
     }

⏺ Saved. Now let me build the 3D workflow.

⏺ cmxflow - build_workflow (MCP)(action: "create")
  ⎿  {
       "status": "success",
       "message": "Created new workflow with MoleculeSourceBlock",
     … +3 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "MoleculeStandardizeBlock")
  ⎿  {
       "status": "success",
       "message": "Added MoleculeStandardize",
     … +4 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "EnumerateStereoBlock")
  ⎿  {
       "status": "success",
       "message": "Added EnumerateStereochemistry",
     … +5 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "ConformerGenerationBlock")
  ⎿  {
       "status": "success",
       "message": "Added ConformerGeneration",
     … +7 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "MoleculeAlignBlock")
  ⎿  {
       "status": "success",
       "message": "Added MoleculeAlign",
     … +12 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "Molecule3DSimilarityBlock")
  ⎿  {
       "status": "success",
       "message": "Added Molecule3DSimilarity",
     … +14 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "add_block", block_type: "EnrichmentScoreBlock", block_config:
                                {"property":"similarity_3d","label_column":"active"})
  ⎿  {
       "status": "success",
       "message": "Added EnrichmentScoreBlock",
     … +17 lines (ctrl+o to expand)

⏺ Now let me make the conformer generation step parallel (it's the slow step), validate, and set inputs.

⏺ cmxflow - build_workflow (MCP)(action: "make_parallel", index: 3, block_config: {"max_workers":4})
  ⎿  {
       "status": "success",
       "message": "Parallelized ParallelConformerGeneration at index 3",
     … +17 lines (ctrl+o to expand)

⏺ cmxflow - build_workflow (MCP)(action: "validate")
  ⎿  {
       "status": "success",
       "message": "Workflow is valid",
     … +18 lines (ctrl+o to expand)

⏺ cmxflow - run_workflow (MCP)(action: "set_inputs", inputs:
                              {"4.file@query":"reference.sdf","5.file@query":"reference.sdf","6.text@target":"active"})
  ⎿  {
       "status": "success",
       "message": "Inputs set successfully"
     }

⏺ Workflow is validated and inputs are set. Ready to optimize. The 3D workflow will be slower due to conformer generation, so 30 trials will take longer. Shall I start with 30 trials again?

  ---
  🧬 3D Shape-Based Virtual Screening Optimization 🧬

  MoleculeSource → MoleculeStandardize → EnumerateStereochemistry → ParallelConformerGeneration → MoleculeAlign → Molecule3DSimilarity →
  EnrichmentScoreBlock

  "The structure of a molecule is the key to understanding its properties." — Linus Pauling

  ---

⏺ cmxflow - optimize_workflow (MCP)(action: "start", n_trials: 30, input_file: "benchmark.parquet", inputs:
                                   {"4.file@query":"reference.sdf","5.file@query":"reference.sdf","6.text@target":"active"}, direction:
                                   "maximize")
  ⎿  {
       "status": "started",
       "message": "Optimization started with 30 trials",
     … +3 lines (ctrl+o to expand)

⏺ The 3D optimization is running. This one will take longer due to conformer generation and 3D alignment at each trial. I'll check results when you ask — just let me know!

✻ Cooked for 57s

❯ check the 3D optimization status

⏺ cmxflow - optimize_workflow (MCP)(action: "status")
  ⎿  {
       "status": "running",
       "message": "Optimization in progress",
     … +12 lines (ctrl+o to expand)

⏺ Still running — 17/30 trials completed so far.

  Best score so far: 0.65 (same as the 2D result)

  Current best parameters:

  ┌──────────────────┬───────────────┐
  │    Parameter     │     Value     │
  ├──────────────────┼───────────────┤
  │ numConfs         │ 62            │
  ├──────────────────┼───────────────┤
  │ pruneRmsThresh   │ 0.017         │
  ├──────────────────┼───────────────┤
  │ useRandomCoords  │ false         │
  ├──────────────────┼───────────────┤
  │ alignment_method │ crippen_o3a   │
  ├──────────────────┼───────────────┤
  │ method           │ shape_tversky │
  ├──────────────────┼───────────────┤
  │ tversky_alpha    │ 0.558         │
  ├──────────────────┼───────────────┤
  │ tversky_beta     │ 0.229         │
  └──────────────────┴───────────────┘

  Interesting that it's gravitating toward Tversky similarity (asymmetric) rather than standard shape Tanimoto. 13 trials left — it may still improve. Let me know when you'd like to check again.

❯ check again
  ⎿  ⧉ Selected 1 lines from ../cmxflow/docs/using-with-claude.md in Visual Studio Code

⏺ cmxflow - optimize_workflow (MCP)(action: "status")
  ⎿  {
       "status": "completed",
       "message": "Optimization completed successfully",
     … +11 lines (ctrl+o to expand)

⏺ The 3D optimization is done.

  Best Enrichment AUC: 0.65 — same as 2D.

  ┌──────────────────┬───────────────┐
  │    Parameter     │  Best Value   │
  ├──────────────────┼───────────────┤
  │ numConfs         │ 62            │
  ├──────────────────┼───────────────┤
  │ pruneRmsThresh   │ 0.017         │
  ├──────────────────┼───────────────┤
  │ useRandomCoords  │ false         │
  ├──────────────────┼───────────────┤
  │ alignment_method │ crippen_o3a   │
  ├──────────────────┼───────────────┤
  │ method           │ shape_tversky │
  ├──────────────────┼───────────────┤
  │ tversky_alpha    │ 0.558         │
  ├──────────────────┼───────────────┤
  │ tversky_beta     │ 0.229         │
  └──────────────────┴───────────────┘

  Summary: Both 2D and 3D workflows achieved the same enrichment (0.65). The 2D workflow (topological torsion + cosine) is significantly faster since it skips conformer generation and alignment. For this target, the 2D approach gives you the same performance at a fraction of the computational cost.

  Would you like me to:
  1. Save the 3D workflow as well?
  2. Set the best params and run either workflow on the full dataset?
```

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
