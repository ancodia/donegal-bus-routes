"""Spectral clustering to detect communities (18 clusters merged to 14).

Run: python -m donegal_bus.offline.community_detection
"""

from donegal_bus.config import Settings


def build_adjacency_matrix(settings: Settings) -> None:
    """Build the adjacency matrix from the weighted graph.

    Reads donegal_osm_weights_applied.graphml.
    """
    # TODO: Port from route_planning/notebooks/community_detection.ipynb (cells 1-4)
    #
    # import networkx as nx
    # import numpy as np
    # from donegal_bus.graph_io import load_graphml, fix_edge_weights
    #
    # weights_path = settings.graph_graphml_path / "donegal_osm_weights_applied.graphml"
    # G = load_graphml(weights_path)
    # fix_edge_weights(G)
    # Remove weakly connected components with < 10 nodes before clustering
    # A = nx.to_numpy_array(G, weight="weight")
    # np.save(str(settings.rp_clusters_path / "adjacency.npy"), A)
    raise NotImplementedError


def run_spectral_clustering(settings: Settings, n_clusters: int = 18) -> None:
    """Perform spectral clustering and save labels.

    Writes cluster labels to rp_clusters_path/sc.npy.
    """
    # TODO: Port from route_planning/notebooks/community_detection.ipynb (cells 5-8)
    #
    # import numpy as np
    # from sklearn.cluster import spectral_clustering
    # from donegal_bus.helpers.route_planning import assign_community_labels
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    #
    # A = np.load(str(settings.rp_clusters_path / "adjacency.npy"))
    # labels = spectral_clustering(A, n_clusters=n_clusters)
    # np.save(str(settings.rp_clusters_path / "sc.npy"), labels)
    #
    # Assign labels to graph nodes and save as communities.graphml
    # weights_path = settings.graph_graphml_path / "donegal_osm_weights_applied.graphml"
    # G = load_graphml(weights_path)
    # assign_community_labels(G, labels.tolist())
    # save_graphml(G, settings.rp_graphml_path / "communities.graphml")
    raise NotImplementedError


def merge_small_communities(settings: Settings, target: int = 14) -> None:
    """Merge small communities until target count is reached.

    Reads/writes communities.graphml.
    Target=14 merges 18 clusters by combining:
    - Communities 8, 12, 17 → 8 (southmost)
    - Communities 10, 13, 16 → 10 (Letterkenny area)
    """
    # TODO: Port from route_planning/notebooks/community_detection.ipynb (cells 9-12)
    #
    # from donegal_bus.graph_io import load_graphml, save_graphml
    # G = load_graphml(settings.rp_graphml_path / "communities.graphml")
    # Merge community labels iteratively until len(unique_labels) == target
    # save_graphml(G, settings.rp_graphml_path / "communities.graphml")
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full community detection pipeline."""
    settings = settings or Settings()
    build_adjacency_matrix(settings)
    run_spectral_clustering(settings)
    merge_small_communities(settings)


if __name__ == "__main__":
    run()
