# Operators

Operator blocks transform molecules within a workflow. Most are 1:1 transforms; some (conformer generation, ionization, stereo enumeration) produce multiple outputs per input.

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
