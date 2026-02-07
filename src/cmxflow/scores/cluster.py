"""Cluster quality scoring block for workflow optimization."""

from collections.abc import Iterator
from typing import Any

from cmxflow.block import ScoreBlock


class ClusterScoreBlock(ScoreBlock):
    """Score clustering quality from RepresentativeClusterBlock.

    Computes mean intra-cluster similarity (excluding singletons) minus
    the fraction of singleton molecules. Designed to be used with
    RepresentativeClusterBlock upstream.

    Score formula:
        score = mean_similarity - (n_single / n_molecules)

    Both terms are in [0, 1], so the score range is [-1, 1].
    """

    def __init__(self) -> None:
        """Initialize the cluster score block."""
        super().__init__(name="ClusterScore")

    def objective(self, iter: Iterator[Any]) -> float:
        """Compute cluster quality score.

        Single pass over the iterator, accumulating per-cluster stats.

        Args:
            iter: Iterator of molecules with cluster_id and
                cluster_similarity properties.

        Returns:
            Cluster quality score in [-1, 1]. Returns 0.0 for empty
            input or all-singleton clusterings.
        """
        clusters: dict[int, tuple[int, float]] = {}
        for mol in iter:
            cid = mol.GetIntProp("cluster_id")
            sim = mol.GetDoubleProp("cluster_similarity")
            count, total = clusters.get(cid, (0, 0.0))
            clusters[cid] = (count + 1, total + sim)

        if not clusters:
            return 0.0

        n_molecules = sum(c for c, _ in clusters.values())
        n_single = sum(1 for c, _ in clusters.values() if c == 1)

        non_singleton_count = 0
        non_singleton_sim_sum = 0.0
        for count, sim_sum in clusters.values():
            if count > 1:
                non_singleton_count += count
                non_singleton_sim_sum += sim_sum

        if non_singleton_count == 0:
            return 0.0

        mean_similarity = non_singleton_sim_sum / non_singleton_count
        singleton_penalty = n_single / n_molecules
        return mean_similarity - singleton_penalty

    def forward(self, item: Any) -> Any:
        """Pass through molecule unchanged.

        Args:
            item: Input molecule.

        Returns:
            The same molecule, unchanged.
        """
        return item
