# Scores

Score blocks compute a single numeric value from a stream of molecules, used as the optimization objective during Bayesian optimization.

Note:
    Score blocks only return scores when called during Bayesian optimization. Otherwise molecules pass through. This mechanism is in place so that agents have two paths to successful conversion of optimized to runnable workflows (delete the score and add a sink or add a sink at the end).

## Enrichment

::: cmxflow.scores.automatic.EnrichmentScoreBlock
    options:
      members: false

## Average Property

::: cmxflow.scores.automatic.AverageScoreBlock
    options:
      members: false

## Shape Overlay

::: cmxflow.scores.shape.ShapeOverlayScoreBlock
    options:
      members: false

## Cluster Quality

::: cmxflow.scores.cluster.ClusterScoreBlock
    options:
      members: false
