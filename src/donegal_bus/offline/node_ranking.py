"""PageRank, top-N selection, and furthest-apart endpoint detection.

Run: python -m donegal_bus.offline.node_ranking
"""

from donegal_bus.config import Settings


def compute_pagerank(settings: Settings) -> None:
    """Run PageRank on the community graph and store rank attributes.

    Reads communities.graphml, writes nodes_ranked.graphml.
    Converts the directed graph to undirected before running PageRank so that
    all nodes participate equally regardless of edge direction.
    """
    import networkx as nx

    from donegal_bus.graph_io import fix_edge_weights, load_graphml, save_graphml

    G = load_graphml(settings.rp_graphml_path / "communities.graphml")
    fix_edge_weights(G)

    pr_graph = nx.Graph(G)
    pr = nx.pagerank(pr_graph)

    nx.set_node_attributes(G, 0.0, "rank")  # type: ignore[call-overload, arg-type]
    for node, rank in pr.items():
        G.nodes[node]["rank"] = rank

    save_graphml(G, settings.rp_graphml_path / "nodes_ranked.graphml")  # type: ignore[arg-type]


def select_top_n_per_community(settings: Settings, n: int = 10) -> None:
    """Flag the top-N ranked nodes in each community.

    Updates nodes_ranked.graphml in place.
    """
    import pandas as pd

    from donegal_bus.graph_io import fix_edge_weights, load_graphml, save_graphml
    from donegal_bus.helpers.route_planning import (
        get_n_highest_ranked_nodes_in_community,
    )

    ranked_path = settings.rp_graphml_path / "nodes_ranked.graphml"
    G = load_graphml(ranked_path)
    fix_edge_weights(G)

    # Build a DataFrame matching the format expected by the helper.
    # Index = osmid (same as the osmid column) to match ox.graph_to_gdfs output.
    records = [
        {
            "osmid": int(node),
            "rank": float(data.get("rank", 0)),
            "community": int(str(data.get("community", 0))),
        }
        for node, data in G.nodes(data=True)
    ]
    all_nodes_df = pd.DataFrame(records).set_index("osmid")
    all_nodes_df["osmid"] = all_nodes_df.index

    community_labels = sorted(all_nodes_df["community"].unique())  # type: ignore[type-var]

    for label in community_labels:
        comm_df = all_nodes_df[all_nodes_df["community"] == label].copy()
        if len(comm_df) < n:
            continue
        ranked_df = get_n_highest_ranked_nodes_in_community(comm_df, n=n)  # type: ignore[arg-type]
        for osmid, row in ranked_df.iterrows():
            top_n_val = int(row["top_n"])  # type: ignore[arg-type]
            if top_n_val > 0:
                G.nodes[osmid]["top_n"] = top_n_val  # type: ignore[index]

    save_graphml(G, ranked_path)  # type: ignore[arg-type]


def flag_route_endpoints(settings: Settings) -> None:
    """Find the furthest-apart top-ranked pair per community and flag as start/end.

    Reads nodes_ranked.graphml, writes route_start_end_flagged.graphml.
    route_flag=1 → route start, route_flag=2 → route end.
    """
    from donegal_bus.graph_io import fix_edge_weights, load_graphml, save_graphml
    from donegal_bus.helpers.route_planning import (
        assign_route_start_end_points,
        greatest_distance_between_top_ranked_nodes,
    )

    G = load_graphml(settings.rp_graphml_path / "nodes_ranked.graphml")
    fix_edge_weights(G)

    community_labels = sorted(
        {int(str(d.get("community", 0))) for _, d in G.nodes(data=True)}
    )

    route_nodes = greatest_distance_between_top_ranked_nodes(  # type: ignore[arg-type]
        G, community_labels
    )
    assign_route_start_end_points(G, route_nodes, n_communities=len(community_labels))  # type: ignore[arg-type]

    out = settings.rp_graphml_path / "route_start_end_flagged.graphml"
    save_graphml(G, out)  # type: ignore[arg-type]


def run(settings: Settings | None = None) -> None:
    """Execute the full node ranking pipeline."""
    settings = settings or Settings()
    compute_pagerank(settings)
    select_top_n_per_community(settings)
    flag_route_endpoints(settings)


if __name__ == "__main__":
    run()
