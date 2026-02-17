# Operators

Operator blocks transform molecules within a workflow. Most are 1:1 transforms; some (conformer generation, ionization, stereo enumeration) produce multiple outputs per input.

## Reading the Reference

Each block documents the following sections where applicable:

- **Required Inputs** — file paths or text values that must be provided before
  running the workflow. Pass them as constructor keyword arguments in Python
  (`MyBlock(key="value")`), or via the MCP agent using `run_workflow set_inputs`.
- **Output Properties** — named properties attached to each output molecule.
  Downstream blocks (filters, selectors, score blocks) can reference these by name.
- **Mutable Parameters** — numeric or categorical settings tuned automatically
  by Bayesian optimization. Set defaults at construction; the optimizer adjusts
  them during `optimize_workflow`.
- **Example** — a minimal end-to-end snippet showing the block in a workflow
  with a source and sink.

## Standardization

::: cmxflow.operators.standardize.MoleculeStandardizeBlock
    options:
      members: false

## Deduplication

::: cmxflow.operators.dedup.MoleculeDeduplicateBlock
    options:
      members: false

## RDKit Methods

::: cmxflow.operators.method.RDKitBlock
    options:
      members: false

## Filtering

::: cmxflow.operators.filter.SubstructureFilterBlock
    options:
      members: false

::: cmxflow.operators.filter.PropertyFilterBlock
    options:
      members: false

## Selection

::: cmxflow.operators.select.PropertyHeadBlock
    options:
      members: false

::: cmxflow.operators.select.PropertyTailBlock
    options:
      members: false

## 2D Similarity

::: cmxflow.operators.sim2d.MoleculeSimilarityBlock
    options:
      members: false

## 3D Similarity

::: cmxflow.operators.sim3d.Molecule3DSimilarityBlock
    options:
      members: false

## Ionization

::: cmxflow.operators.ionize.IonizeMoleculeBlock
    options:
      members: false

## Stereoisomers

::: cmxflow.operators.confgen.EnumerateStereoBlock
    options:
      members: false

## Conformer Generation

::: cmxflow.operators.confgen.ConformerGenerationBlock
    options:
      members: false

## Alignment

::: cmxflow.operators.align.MoleculeAlignBlock
    options:
      members: false

## Docking

::: cmxflow.operators.dock.MoleculeDockBlock
    options:
      members: false

## Clustering

::: cmxflow.operators.cluster.RepresentativeClusterBlock
    options:
      members: false
