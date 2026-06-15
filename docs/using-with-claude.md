# Using with Claude

cmxflow includes an [MCP](https://modelcontextprotocol.io/) server that lets Claude build, run, and optimize cheminformatics workflows conversationally.

## Setup

Add the cmxflow MCP server to Claude Code:

```bash
claude mcp add cmxflow -- cmxflow-mcp
```

This gives Claude access to five tools.

**Note:** Claude Code defers MCP tool schemas until needed, so Claude may write RDKit code directly instead of using cmxflow. To prevent this, add a `CLAUDE.md` file to your working directory with the following line:

```
Always use the cmxflow MCP tools for cheminformatics tasks. Do not write RDKit or other cheminformatics code directly.
```

| Tool | Description |
|------|-------------|
| `build_workflow` | Create and configure workflows step-by-step |
| `run_workflow` | Set inputs and execute a validated workflow |
| `optimize_workflow` | Bayesian optimization of workflow parameters |
| `manage_workflows` | Save, load, list, and remove workflows |
| `view_structures` | Open 3D output in PyMOL |

## Example Prompts

All examples below were carried out on a MacBook Air M2 with 8GB RAM using `Sonnet 4.6` in Claude Code.

### Similarity Search

> Read molecules from screen.sdf.gz, compute 2D similarity to queries.sdf using ECFP4 fingerprints, and write the top 100 to similar.csv

```
⏺ I'll use the cmxflow MCP tools as instructed. Let me first explore what's available.

⏺ Let me check for existing workflows and available blocks in parallel.

  Called cmxflow 2 times (ctrl+o to expand)

⏺ No existing workflows. I'll build a 4-step pipeline: source → similarity → top 100 → sink. Let me check the similarity block parameters first.

  Called cmxflow 3 times (ctrl+o to expand)

⏺ The block already defaults to Morgan radius=2 (= ECFP4) with Tanimoto. I still need to set the queries input file. Let me add the remaining blocks first.

  Called cmxflow 4 times (ctrl+o to expand)

⏺ Workflow is valid. Now I'll set the required inputs (queries file, and confirm the property/count that are already configured).

  Called cmxflow 2 times (ctrl+o to expand)

⏺ ---
  🧪 ECFP4 Similarity Screen 🔬

  ┌────────────────┐
  │ MoleculeSource │  ← screen.sdf.gz
  └───────┬────────┘
          ↓
  ┌─────────────────────────────┐   ┌──────────────────────┐
  │    Molecule2DSimilarity     │ ← │ queries: queries.sdf │
  │ fingerprint_type: morgan    │   └──────────────────────┘
  │ similarity_metric: tanimoto │
  │ radius: 2 | nbits: 2048     │
  └───────┬─────────────────────┘
          ↓
  ┌──────────────┐
  │ PropertyHead │  top 100 by max_similarity
  └───────┬──────┘
          ↓
  ┌────────────────┐
  │  MoleculeSink  │  → similar.csv
  └────────────────┘

  ▎ "The chemist who knows where to look is more powerful than the one who knows everything." — inspired by Linus Pauling

  Called cmxflow (ctrl+o to expand)

⏺ Done. The workflow read all molecules from screen.sdf.gz, computed ECFP4 (Morgan radius=2, 2048 bits) Tanimoto similarity against every query in queries.sdf, kept the maximum per screened molecule, and wrote the top 100 by max_similarity to similar.csv.

✻ Cooked for 1m 16s
```

### Virtual Screening with Optimization

> I need to build a ligand-based virtual screening workflow. I'm not sure if 2D or 3D is better. Can you optimize two workflows? I want to see the results of 2D first. The benchmark is in "benchmark.parquet" with hits labeled in the "active" column and the query is in "reference.sdf".

```
⏺ I'll use the cmxflow MCP tools to build and optimize both workflows. Let me start by exploring what's available and then set up both in parallel.

⏺ Let me check for existing workflows and available blocks simultaneously.

  Called cmxflow 2 times (ctrl+o to expand)

⏺ No existing workflows. I'll build both the 2D and 3D screening workflows, save them, then optimize 2D first. Let me start building the 2D workflow.

  Called cmxflow 4 times (ctrl+o to expand)

⏺ Good. Now let me validate the 2D workflow and check what inputs need to be set.

  Called cmxflow 2 times (ctrl+o to expand)
  building the 2D workflow.

  Called cmxflow 4 times (ctrl+o to expand)

⏺ Good. Now let me validate the 2D workflow and check what inputs need to be set.

  Called cmxflow 2 times (ctrl+o to expand)

⏺ Now save the 2D workflow before building the 3D one.

  Called cmxflow (ctrl+o to expand)

⏺ Now building the 3D workflow.

  Called cmxflow 10 times (ctrl+o to expand)

⏺ Both workflows are saved. Before I start optimization, I need to confirm a couple of things with you:

⏺ User answered Claude's questions:
  ⎿  · How many optimization trials should be run for each workflow? (30 is typical — more trials find better parameters but take longer) → 30 trials (Recommended)
     · Should any steps in the 2D or 3D workflows run in parallel to speed up optimization? → Yes — parallelize ConformerGeneration (3D only, Recommended)

⏺ 30 trials, ConformerGeneration parallelized for 3D. Loading the 2D workflow to optimize first.

  Called cmxflow (ctrl+o to expand)

⏺ ---
  🧪 2D Fingerprint Screening Optimizer 🔬

  ┌────────────────┐
  │ MoleculeSource │  benchmark.parquet
  └───────┬────────┘
          ↓
  ┌─────────────────────┐
  │ MoleculeStandardize │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────────────┐
  │    Molecule2DSimilarity     │ ← reference.sdf
  │ fingerprint: morgan r=2     │
  │ metric: tanimoto            │
  └──────────────┬──────────────┘
                 ↓
         ┌──────────────────────┐
         │  EnrichmentScore     │ ← active column
         └──────────────────────┘

  ▎ "The properties of a substance are determined by its constitution." — August Kekulé

  Starting 2D optimization now (30 trials, maximizing enrichment AUC):

  Called cmxflow 2 times (ctrl+o to expand)

⏺ The 2D optimization is running — 30 trials exploring fingerprint type, radius, bit count, and similarity metric combinations to maximize enrichment AUC against your active labels.

  Don't poll — let me know when you'd like a status update or the results. Once 2D is done I'll load and parallelize the 3D workflow, then kick off its optimization. Let me know when you're ready to check progress.

✻ Brewed for 1m 58s

❯ check the status                         

  Called cmxflow (ctrl+o to expand)

⏺ The 2D optimization is done. Best result:

  ┌───────────────────┬─────────────────────┐
  │     Parameter     │        Value        │
  ├───────────────────┼─────────────────────┤
  │ Fingerprint       │ Morgan              │
  ├───────────────────┼─────────────────────┤
  │ Similarity metric │ Dice (not Tanimoto) │
  ├───────────────────┼─────────────────────┤
  │ Radius            │ 3                   │
  ├───────────────────┼─────────────────────┤
  │ Bits              │ 2582                │
  ├───────────────────┼─────────────────────┤
  │ Enrichment AUC    │ 0.75                │
  └───────────────────┴─────────────────────┘

  The optimizer swapped Tanimoto for Dice and bumped the radius from 2→3, which picked up more extended neighborhood context.
  Ready to apply these parameters and kick off the 3D workflow optimization — shall I proceed?

✻ Churned for 6s

❯ yes, go ahead with the 3D workflow                    

⏺ Applying the best 2D parameters and saving, then loading the 3D workflow.

  Called cmxflow 3 times (ctrl+o to expand)

⏺ Now parallelizing ConformerGeneration (index 3) with max_workers=4.

  Called cmxflow (ctrl+o to expand)

⏺ ---
  🧬 3D Shape Screening Optimizer ⚗️ 

  ┌────────────────┐
  │ MoleculeSource │  benchmark.parquet
  └───────┬────────┘
          ↓
  ┌─────────────────────┐
  │ MoleculeStandardize │
  └──────────┬──────────┘
             ↓
  ┌──────────────────────────┐
  │ EnumerateStereochemistry │
  └─────────────┬────────────┘
                ↓
  ┌─────────────────────────────┐
  │ ParallelConformerGeneration │  4 workers
  └──────────────┬──────────────┘
                 ↓
  ┌───────────────────────────────┐
  │         MoleculeAlign         │ ← reference.sdf
  │  method: crippen_o3a          │
  └──────────────┬────────────────┘
                 ↓
  ┌─────────────────────────┐
  │  Molecule3DSimilarity   │ ← reference.sdf
  │  method: shape_tanimoto │
  └────────────┬────────────┘
               ↓
       ┌──────────────────────┐
       │   EnrichmentScore    │ ← active column
       └──────────────────────┘

  ▎ "Structural formula is the key to the molecular architecture." — Emil Fischer

  Called cmxflow 3 times (ctrl+o to expand)

⏺ 3D optimization is running — 30 trials exploring conformer settings, alignment method, and 3D similarity metric (shape Tanimoto, Tversky variants, etc.) with ConformerGeneration on 4 workers. Let me know when you want to check the results.

✻ Baked for 34s

※ recap: Building 2D and 3D ligand-based virtual screening workflows using your benchmark.parquet and reference.sdf. The 2D optimization finished with AUC 0.75; the 3D optimization is still running. Check status when ready. (disable recaps in /config)
  
❯ check the status                                                    

  Called cmxflow (ctrl+o to expand)

⏺ 3D optimization is done. Here's the full comparison:

  ┌────────────────┬────────────────────────────┬────────────────────────────────────────────────┐
  │                │      2D (Morgan/Dice)      │                   3D (Shape)                   │
  ├────────────────┼────────────────────────────┼────────────────────────────────────────────────┤
  │ Enrichment AUC │ 0.75                       │ 0.65                                           │
  ├────────────────┼────────────────────────────┼────────────────────────────────────────────────┤
  │ Key params     │ radius=3, nbits=2582, Dice │ numConfs=9, MMFF-O3A alignment, shape_tanimoto │
  └────────────────┴────────────────────────────┴────────────────────────────────────────────────┘

  2D wins by 0.10 AUC on this benchmark. The 3D workflow found better conformer settings (9 conformers, MMFF-O3A alignment over Crippen), but the extra structural information didn't translate to better enrichment here — which is common when the active series shares a clear 2D pharmacophore.

  Would you like me to apply the best 3D parameters and save both workflows, or explore a combined 2D+3D scoring approach?

✻ Cogitated for 9s
```

### Filtering / Library Preparation

> How many of the molecules in library.csv pass Lipinski's rules?

```
⏺ Let me check the library.csv file and the available cmxflow tools.

  Read 1 file (ctrl+o to expand)

⏺ The library has 1000 molecules. Let me check for existing workflows and available blocks for Lipinski filtering.

  Called cmxflow 2 times (ctrl+o to expand)

⏺ I'll build a workflow to compute Lipinski properties and filter the library. Let me set it up step by step.

  Called cmxflow 9 times (ctrl+o to expand)

⏺ Workflow is valid. Time to run it!

⏺ I'll build a workflow to compute Lipinski properties and filter the library. Let me set it up step by step.

  Called cmxflow 9 times (ctrl+o to expand)

⏺ Workflow is valid. Time to run it!

  ---
  💊 Lipinski Rule-of-Five Filter 🧪

  MoleculeSource → MoleculeStandardize → RDKit:MolWt → RDKit:MolLogP
    → RDKit:CalcNumHBD → RDKit:CalcNumHBA → PropertyFilter → MoleculeSink

  ▎ "A drug is a substance that when injected into a guinea pig, produces a scientific paper."
  ▎ — attributed to various pharmacologists

  ---

  Read 1 file, called cmxflow (ctrl+o to expand)

⏺ 615 out of 1000 molecules pass Lipinski's Rule of Five (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10), meaning 385 fail at least one rule. The passing molecules were written to lipinski_pass.csv.

✻ Brewed for 1m 2s
```

> Cluster the remainder of the molecules. Make sure to balance intracluster similarity with number of singletons.

```
⏺ I'll cluster the 615 Lipinski-passing molecules. To balance intra-cluster similarity against singletons, I'll use the ClusterScoreBlock (maximized) with optimization to find the best similarity threshold. Let me build the workflow first.

  Called cmxflow 4 times (ctrl+o to expand)

⏺ Workflow is valid. Here's what I have in mind:

  ---
  🔬 Cluster Quality Optimizer 🧬

  MoleculeSource → RepresentativeCluster → ClusterScore (maximize)

  ▎ "The most important property of a compound is not its structure but its biological activity."
  ▎ — Paul Ehrlich

  ---
  The ClusterScoreBlock computes mean intra-cluster similarity − fraction of singletons, so maximizing it directly optimizes the trade-off you want.
  
  Before I start optimization, two quick questions:

  1. n_trials: 30 is typically sufficient — would you like more or fewer?
  2. Parallelism: Should the RepresentativeClusterBlock step run in parallel? (Note: its docs say it cannot be parallelized due to shared state, so this would only apply if other blocks are added.)

✻ Worked for 52s

❯ 30 trials no parallelism                                

  Called cmxflow (ctrl+o to expand)

⏺ Optimization is running — 30 trials maximizing mean intra-cluster similarity − singleton fraction over the clustering threshold. I'll let you know when it's done; just ask for status whenever you're ready.

✻ Baked for 10s

❯ What are the results?                                                

  Called cmxflow (ctrl+o to expand)

⏺ Optimization is complete. Best results:

  ┌─────────────────────┬───────┐
  │      Parameter      │ Value │
  ├─────────────────────┼───────┤
  │ Threshold           │ 0.281 │
  ├─────────────────────┼───────┤
  │ Scaffold clustering │ True  │
  ├─────────────────────┼───────┤
  │ Best score          │ 0.410 │
  └─────────────────────┴───────┘

  The optimizer settled on a relatively low similarity threshold (0.28), which keeps intra-cluster cohesion tight while minimizing singletons. Would you like me to apply these best parameters and run the final clustering to produce the annotated output file?

✻ Worked for 10s

❯ yes                                                            

  Called cmxflow (ctrl+o to expand)

⏺ Now I'll swap the ScoreBlock for a SinkBlock to write the output.

  Read 1 file, called cmxflow 4 times (ctrl+o to expand)

⏺ Done. Results written to clustered.csv with cluster_id, cluster_representative, and cluster_similarity annotations. Summary:
     
  ┌────────────────────────┬──────────────────────────────────┐
  │         Metric         │              Value               │
  ├────────────────────────┼──────────────────────────────────┤
  │ Molecules clustered    │ 615                              │
  ├────────────────────────┼──────────────────────────────────┤
  │ Total clusters         │ 245                              │
  ├────────────────────────┼──────────────────────────────────┤
  │ Singletons             │ 149                              │
  ├────────────────────────┼──────────────────────────────────┤
  │ Non-singleton clusters │ 96                               │
  ├────────────────────────┼──────────────────────────────────┤
  │ Threshold used         │ 0.281 (Tanimoto, scaffold-based) │
  └────────────────────────┴──────────────────────────────────┘

  Would you like to save this workflow for future use?

✻ Churned for 28s
```

### Docking

> Dock the molecules in hits.csv against receptor.pdb with crystal_ligand.sdf as a reference

```
⏺ I'll use the cmxflow MCP tools for this docking task. Let me first check the available files and load the tool schemas.
                                                                                                            
  Listed 1 directory (ctrl+o to expand)

⏺ All three files are present. Let me check for existing workflows before building a new one.

  Called cmxflow (ctrl+o to expand)

⏺ No docking workflow exists yet. Let me check the available blocks to find the right docking block.
  
  Called cmxflow (ctrl+o to expand)

⏺ I have everything I need. Let me build a docking workflow: MoleculeSource → Standardize → ConformerGeneration → Dock → Sink.

  Called cmxflow 6 times (ctrl+o to expand)

⏺ Workflow is valid. Now let me set the receptor and reference inputs, then execute.

  ---
  ⚗️  Hits Docking Pipeline 🔬

  MoleculeSource → MoleculeStandardize → ConformerGeneration → MoleculeDock → MoleculeSink

  "The first step is to measure, the second is to model, the third is to dock." — inspired by Gertrude Elion

  ---

  Called cmxflow 2 times (ctrl+o to expand)

⏺ Docking complete. Results written to docked_hits.sdf — each molecule has a docking_score property (lower = better) along with component scores (electrostatic complementarity, etc.).

  Would you like to visualize the docked poses in PyMOL alongside the receptor and crystal ligand?

✻ Cogitated for 1m 2s

❯ yes                                                           

  Called cmxflow (ctrl+o to expand)

⏺ PyMOL is open with the docked poses (docked_hits.sdf), the receptor (receptor.pdb), and the crystal ligand
  (crystal_ligand.sdf) for reference. You can toggle visibility of individual layers to compare how your hits bind relative to the co-crystal pose.

✻ Cooked for 5s
```

> Poses could be better. Anything we can do to improve the workflow?

```
⏺ A few levers worth considering:

  1. More search thoroughness — increase n_starts (currently 32) and enable basin_hops (currently 0) on MoleculeDockBlock. More starts + basin hopping explore the pocket more exhaustively.
  2. Add alignment before docking — inserting a MoleculeAlignBlock (using crystal_ligand.sdf as reference) before docking gives each molecule a better starting 3D overlay, which can guide the search into the right region of the pocket.
  3. More conformers — increase numConfs in ConformerGenerationBlock from 1 to e.g. 5–10 so docking has richer starting geometries to work from.
  4. Bayesian optimization — run optimize_workflow to tune the scoring weights (w_gauss1, w_hydrophobic, w_hbond, etc.) and search parameters automatically against the crystal pose.

  The highest-impact single change is usually more starts + basin hops (option 1) combined with alignment (option 2). Bayesian optimization (option 4) is the most principled approach if you have a benchmark set to score against.

  Which would you like to try — one of these individually, or a combination?

✻ Cogitated for 21s

❯ Try 1 first                                                     

⏺ I'll rebuild the workflow with higher n_starts and basin_hops on the dock block. Since this will be slower, I'll parallelize the docking step too.

  Called cmxflow 5 times (ctrl+o to expand)

⏺ Let me check what values are valid for those parameters.

  Called cmxflow 5 times (ctrl+o to expand)

⏺ n_starts appears fixed at 32, but basin_hops is now set to 10 (up from 0). Let me add the sink, parallelize the dock step for speed, then run.

  Called cmxflow 3 times (ctrl+o to expand)

⏺ ---
  ⚗️  Hits Docking Pipeline — Enhanced Search 🔍

  MoleculeSource → MoleculeStandardize → ConformerGeneration → ParallelMoleculeDock → MoleculeSink
  basin_hops: 10 (was 0) | n_starts: 32 | parallelized across all cores
  
  "If we knew what it was we were doing, it would not be called research, would it?" — Albert Einstein

  ---

  Called cmxflow 2 times (ctrl+o to expand)

⏺ Done — docked_hits.sdf updated with the enhanced search (basin_hops: 10). Note that n_starts appears to be fixed at 32 by the block and couldn't be increased further.

  Would you like to visualize the new poses in PyMOL to compare, or try one of the other options (alignment pre-step or more conformers)?

✻ Sautéed for 1m 48s

❯ yes                                                                  

  Called cmxflow (ctrl+o to expand)

⏺ PyMOL is open with the updated poses alongside the receptor and crystal ligand. Let me know if the poses look better, or if you'd like to try adding alignment or more conformers next.

✻ Sautéed for 5s
```


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
