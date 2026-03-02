"""Spectral clustering to detect communities (18 clusters merged to 14).

Run: python -m donegal_bus.offline.community_detection
"""

from donegal_bus.config import Settings


def build_adjacency_matrix(settings: Settings) -> None:
    """Build the adjacency matrix from the weighted graph.

    Reads donegal_osm_weights_applied.graphml.
    Removes weakly/strongly connected components with < 10 nodes.
    Saves the cleaned graph back in place, then writes adjacency.npy.
    """
    import networkx as nx
    import numpy as np

    from donegal_bus.graph_io import fix_edge_weights, load_graphml, save_graphml

    weights_path = settings.graph_graphml_path / "donegal_osm_weights_applied.graphml"
    G = load_graphml(weights_path)

    # Remove small weakly connected components (< 10 nodes)
    small_weak = [c for c in nx.weakly_connected_components(G) if len(c) < 10]
    for component in small_weak:
        G.remove_nodes_from(component)

    # Remove small strongly connected components (< 10 nodes)
    small_strong = [c for c in nx.strongly_connected_components(G) if len(c) < 10]
    for component in small_strong:
        G.remove_nodes_from(component)

    # Save cleaned graph back to the same path so run_spectral_clustering
    # loads a consistent node set
    save_graphml(G, weights_path)  # type: ignore[arg-type]

    fix_edge_weights(G)
    A = nx.to_numpy_array(G, weight="weight")
    settings.rp_clusters_path.mkdir(parents=True, exist_ok=True)
    np.save(str(settings.rp_clusters_path / "adjacency.npy"), A)


def run_spectral_clustering(settings: Settings, n_clusters: int = 18) -> None:
    """Perform spectral clustering and save labels.

    Loads the cleaned graph + adjacency matrix, runs spectral clustering with
    a fixed random_state for reproducibility, assigns community labels, and
    writes communities.graphml.

    Note: spectral clustering is non-deterministic even with random_state due
    to the underlying eigensolver. For exact reproducibility, pre-compute
    labels once, save as sc.npy, and load them instead of re-running.
    """
    import numpy as np
    from sklearn.cluster import spectral_clustering  # type: ignore[import-untyped]

    from donegal_bus.graph_io import load_graphml, save_graphml
    from donegal_bus.helpers.route_planning import assign_community_labels

    A = np.load(str(settings.rp_clusters_path / "adjacency.npy"))

    # Check for pre-computed labels (avoids non-determinism across runs)
    sc_path = settings.rp_clusters_path / "sc.npy"
    if sc_path.exists():
        labels = np.load(str(sc_path))
    else:
        labels = spectral_clustering(A, n_clusters=n_clusters, random_state=42)
        np.save(str(sc_path), labels)  # type: ignore[arg-type]

    weights_path = settings.graph_graphml_path / "donegal_osm_weights_applied.graphml"
    G = load_graphml(weights_path)

    assign_community_labels(G, [int(x) for x in labels])  # type: ignore[arg-type]

    settings.rp_graphml_path.mkdir(parents=True, exist_ok=True)
    save_graphml(G, settings.rp_graphml_path / "communities.graphml")  # type: ignore[arg-type]


def merge_small_communities(settings: Settings, target: int = 14) -> None:
    """Merge small communities until target count is reached.

    Reads/writes communities.graphml.
    Merges 18 → 14 by combining:
      - Communities 8, 12, 17 → 8  (southernmost)
      - Communities 10, 13, 16 → 10 (Letterkenny area)
    """
    from donegal_bus.graph_io import load_graphml, save_graphml

    communities_path = settings.rp_graphml_path / "communities.graphml"
    G = load_graphml(communities_path)

    merge_map = {12: 8, 17: 8, 13: 10, 16: 10}

    for node in G.nodes:
        raw = G.nodes[node].get("community", 0)
        label = int(str(raw))
        if label in merge_map:
            G.nodes[node]["community"] = merge_map[label]

    unique = {G.nodes[n].get("community") for n in G.nodes}
    if len(unique) != target:
        print(
            f"Warning: expected {target} communities after merge, got {len(unique)}."
            f" Labels: {sorted(unique)}"  # type: ignore[type-var]
        )

    save_graphml(G, communities_path)  # type: ignore[arg-type]


def run(settings: Settings | None = None) -> None:
    """Execute the full community detection pipeline."""
    settings = settings or Settings()
    build_adjacency_matrix(settings)
    run_spectral_clustering(settings)
    merge_small_communities(settings)


if __name__ == "__main__":
    run()
