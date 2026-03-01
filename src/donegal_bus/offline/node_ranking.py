"""PageRank, top-N selection, and furthest-apart endpoint detection.

Run: python -m donegal_bus.offline.node_ranking
"""

from donegal_bus.config import Settings


def compute_pagerank(settings: Settings) -> None:
    """Run PageRank on the community graph and store rank attributes.

    Reads communities.graphml, writes nodes_ranked.graphml.
    """
    # TODO: Port from route_planning/notebooks/node_ranking.ipynb (cells 1-5)
    #
    # import networkx as nx
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    #
    # G = load_graphml(settings.rp_graphml_path / "communities.graphml")
    # fix_edge_weights(G)
    # pr_graph = G.to_undirected()
    # pagerank = nx.pagerank(pr_graph)
    # nx.set_node_attributes(G, pagerank, "rank")
    # save_graphml(G, settings.rp_graphml_path / "nodes_ranked.graphml")
    raise NotImplementedError


def select_top_n_per_community(settings: Settings, n: int = 10) -> None:
    """Flag the top-N ranked nodes in each community.

    Updates nodes_ranked.graphml in place. Old notebooks used n=15.
    """
    # TODO: Port from route_planning/notebooks/node_ranking.ipynb (cells 6-9)
    #
    # import pandas as pd
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    # from donegal_bus.helpers.route_planning import (
    #     get_n_highest_ranked_nodes_in_community, split_into_community_graphs
    # )
    #
    # G = load_graphml(settings.rp_graphml_path / "nodes_ranked.graphml")
    # fix_edge_weights(G)
    # For each community subgraph:
    #   nodes_df = pd.DataFrame(dict(G.nodes(data=True))).T
    #   community_df = nodes_df[nodes_df["community"] == label]
    #   community_df = get_n_highest_ranked_nodes_in_community(community_df, n=n)
    #   For each flagged node, set G.nodes[osmid]["top_n"] = rank
    # save_graphml(G, settings.rp_graphml_path / "nodes_ranked.graphml")
    raise NotImplementedError


def flag_route_endpoints(settings: Settings) -> None:
    """Find the furthest-apart top-ranked pair per community and flag as start/end.

    Reads nodes_ranked.graphml, writes route_start_end_flagged.graphml.
    route_flag=1 → route start, route_flag=2 → route end.
    """
    # TODO: Port from route_planning/notebooks/node_ranking.ipynb (cells 10-14)
    #
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    # from donegal_bus.helpers.route_planning import (
    #     greatest_distance_between_top_ranked_nodes, assign_route_start_end_points
    # )
    #
    # G = load_graphml(settings.rp_graphml_path / "nodes_ranked.graphml")
    # fix_edge_weights(G)
    # community_labels = list(set(d["community"] for _, d in G.nodes(data=True)))
    # route_nodes = greatest_distance_between_top_ranked_nodes(G, community_labels)
    # assign_route_start_end_points(G, route_nodes, n_communities=len(community_labels))
    # save_graphml(G, settings.rp_graphml_path / "route_start_end_flagged.graphml")
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full node ranking pipeline."""
    settings = settings or Settings()
    compute_pagerank(settings)
    select_top_n_per_community(settings)
    flag_route_endpoints(settings)


if __name__ == "__main__":
    run()
