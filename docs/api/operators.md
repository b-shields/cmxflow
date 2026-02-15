# Operators

Operator blocks transform molecules within a workflow. Most are 1:1 transforms; some (conformer generation, ionization, stereo enumeration) produce multiple outputs per input.

## Standardization

::: cmxflow.operators.standardize.MoleculeStandardizeBlock

## Deduplication

::: cmxflow.operators.dedup.MoleculeDeduplicateBlock

## RDKit Methods

::: cmxflow.operators.method.RDKitBlock

## Filtering

::: cmxflow.operators.filter.SubstructureFilterBlock

::: cmxflow.operators.filter.PropertyFilterBlock

## Selection

::: cmxflow.operators.select.PropertyHeadBlock

::: cmxflow.operators.select.PropertyTailBlock

## 2D Similarity

::: cmxflow.operators.sim2d.MoleculeSimilarityBlock

## 3D Similarity

::: cmxflow.operators.sim3d.Molecule3DSimilarityBlock

## Ionization

::: cmxflow.operators.ionize.IonizeMoleculeBlock

## Stereoisomers

::: cmxflow.operators.confgen.EnumerateStereoBlock

## Conformer Generation

::: cmxflow.operators.confgen.ConformerGenerationBlock

## Alignment

::: cmxflow.operators.align.MoleculeAlignBlock

## Docking

::: cmxflow.operators.dock.MoleculeDockBlock

## Clustering

::: cmxflow.operators.cluster.RepresentativeClusterBlock
