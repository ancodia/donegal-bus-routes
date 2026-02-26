"""Spectral clustering to detect communities (18 clusters merged to 14).

Run: python -m donegal_bus.offline.community_detection
"""

from donegal_bus.config import Settings


def build_adjacency_matrix(settings: Settings) -> None:
    """Build the adjacency matrix from the weighted graph.

    Reads donegal_osm_weights_applied.graphml.
    """
    raise NotImplementedError


def run_spectral_clustering(settings: Settings, n_clusters: int = 18) -> None:
    """Perform spectral clustering and save labels.

    Writes cluster labels to rp_clusters_path/sc.npy.
    """
    raise NotImplementedError


def merge_small_communities(settings: Settings, target: int = 14) -> None:
    """Merge small communities until target count is reached.

    Reads/writes communities.graphml.
    """
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full community detection pipeline."""
    settings = settings or Settings()
    build_adjacency_matrix(settings)
    run_spectral_clustering(settings)
    merge_small_communities(settings)


if __name__ == "__main__":
    run()
