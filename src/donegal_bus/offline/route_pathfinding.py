"""Highest-weight simple paths for community and connecting routes.

Run: python -m donegal_bus.offline.route_pathfinding
"""

from donegal_bus.config import Settings


def compute_community_routes(settings: Settings, cutoff: int = 90) -> None:
    """Find highest-weight simple paths within each community.

    Reads route_start_end_flagged.graphml,
    writes community_routes_cutoff{cutoff}.graphml.

    WARNING: find_highest_weighted_simple_path is O(exponential).
    Original notebook used cutoff=110, which took ~41 minutes total.
    """
    # TODO: Port from route_planning/notebooks/route_pathfinding.ipynb (Part 1)
    #
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    # from donegal_bus.helpers.route_planning import (
    #     split_into_community_graphs, find_highest_weighted_simple_path
    # )
    #
    # G = load_graphml(settings.rp_graphml_path / "route_start_end_flagged.graphml")
    # fix_edge_weights(G)
    # community_graphs = split_into_community_graphs(G)
    # for community_graph in community_graphs:
    #   path = find_highest_weighted_simple_path(community_graph, cutoff=cutoff)
    #   for i, node in enumerate(path):
    #       G.nodes[node]["community_route"] = True
    #       G.nodes[node]["community_route_order"] = str(i)
    #
    # Post-process: filter out nodes with weighted degree < 50 (low population)
    # out = settings.rp_graphml_path / f"community_routes_cutoff{cutoff}.graphml"
    # save_graphml(G, out)
    raise NotImplementedError


def compute_connecting_routes(settings: Settings) -> None:
    """Find routes that connect adjacent communities.

    Reads community route graph, writes all_routes.graphml.
    4 connecting routes: a (comm4→5), b (comm5→9), c (comm0→1), d (comm2→15).
    """
    # TODO: Port from route_planning/notebooks/route_pathfinding.ipynb (Part 2)
    #
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    # from donegal_bus.helpers.route_planning import find_highest_weighted_simple_path
    #
    # G = load_graphml(settings.rp_graphml_path / "community_routes_cutoff110.graphml")
    # fix_edge_weights(G)
    #
    # connections = [
    #     ("a", community4_u_node, community5_u_node),
    #     ("b", community5_u_node, community9_u_node),
    #     ("c", community0_v_node, community1_v_node),
    #     ("d", community2_v_node, community15_v_node),
    # ]
    # For each connection: find path, set connection_route=True,
    # connection="x", connection_order="x-{i}" on each node
    #
    # out = settings.rp_graphml_path / "all_routes.graphml"
    # save_graphml(G, out)
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full route pathfinding pipeline."""
    settings = settings or Settings()
    compute_community_routes(settings)
    compute_connecting_routes(settings)


if __name__ == "__main__":
    run()
