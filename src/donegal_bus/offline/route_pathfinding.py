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
    import networkx as nx

    from donegal_bus.graph_io import fix_edge_weights, load_graphml, save_graphml
    from donegal_bus.helpers.route_planning import (
        find_highest_weighted_simple_path,
        split_into_community_graphs,
    )

    G = load_graphml(settings.rp_graphml_path / "route_start_end_flagged.graphml")
    fix_edge_weights(G)

    nx.set_node_attributes(G, False, "community_route")  # type: ignore[call-overload, arg-type]
    nx.set_node_attributes(G, -1, "community_route_order")  # type: ignore[call-overload, arg-type]

    community_graphs = split_into_community_graphs(G)  # type: ignore[arg-type]

    for community_graph in community_graphs:
        path = find_highest_weighted_simple_path(community_graph, cutoff=cutoff)  # type: ignore[arg-type]
        for i, node in enumerate(path):
            G.nodes[node]["community_route"] = True
            G.nodes[node]["community_route_order"] = i

    # Filter out low-population nodes (weighted degree < 50)
    weighted_degrees = dict(G.degree(weight="weight"))
    for node, deg in weighted_degrees.items():
        if float(deg) < 50:  # type: ignore[arg-type]
            G.nodes[node]["community_route"] = False

    out = settings.rp_graphml_path / f"community_routes_cutoff{cutoff}.graphml"
    save_graphml(G, out)  # type: ignore[arg-type]


def compute_connecting_routes(settings: Settings, cutoff: int = 90) -> None:
    """Find routes that connect adjacent communities.

    Reads community_routes_cutoff{cutoff}.graphml, writes all_routes.graphml.
    4 connecting routes: a (comm4→5), b (comm5→9), c (comm0→1), d (comm2→15).
    """
    import networkx as nx

    from donegal_bus.graph_io import fix_edge_weights, load_graphml, save_graphml
    from donegal_bus.helpers.route_planning import find_highest_weighted_simple_path

    G = load_graphml(
        settings.rp_graphml_path / f"community_routes_cutoff{cutoff}.graphml"
    )
    fix_edge_weights(G)

    nx.set_node_attributes(G, "", "connection")  # type: ignore[call-overload, arg-type]
    nx.set_node_attributes(G, False, "connection_route")  # type: ignore[call-overload, arg-type]
    nx.set_node_attributes(G, "", "connection_order")  # type: ignore[call-overload, arg-type]

    def _get_flag_node(community_val: str, flag_val: str) -> int:
        """Return the first node matching the given community and route_flag."""
        for node, data in G.nodes(data=True):
            if (
                str(data.get("community", "")) == community_val
                and str(data.get("route_flag", "")) == flag_val
            ):
                return int(node)  # type: ignore[arg-type]
        raise ValueError(
            f"No node found with community={community_val}, route_flag={flag_val}"
        )

    # Define the 4 connecting routes: (label, start_community, end_community, flag)
    # Routes A/B use route_flag "1" (u/start); C/D use route_flag "2" (v/end)
    connections = [
        ("a", "4", "5", "1"),
        ("b", "5", "9", "1"),
        ("c", "0", "1", "2"),
        ("d", "2", "15", "2"),
    ]

    for label, comm_start, comm_end, flag in connections:
        start_node = _get_flag_node(comm_start, flag)
        end_node = _get_flag_node(comm_end, flag)

        # Build subgraph from the two communities
        sub_nodes = [
            x
            for x, y in G.nodes(data=True)
            if str(y.get("community", "")) in (comm_start, comm_end)
        ]
        subgraph = G.subgraph(sub_nodes)

        path = find_highest_weighted_simple_path(
            subgraph,  # type: ignore[arg-type]
            cutoff=cutoff,
            start_node=start_node,
            end_node=end_node,
        )

        for j, node in enumerate(path):
            G.nodes[node]["connection_route"] = True
            existing_conn = G.nodes[node].get("connection", "")
            G.nodes[node]["connection"] = (
                label if existing_conn == "" else f"{existing_conn}, {label}"
            )
            existing_order = G.nodes[node].get("connection_order", "")
            order_entry = f"{label}-{j}"
            G.nodes[node]["connection_order"] = (
                order_entry
                if existing_order == ""
                else f"{existing_order}, {order_entry}"
            )

    # Filter out low-population nodes (weighted degree < 50)
    weighted_degrees = dict(G.degree(weight="weight"))
    for node, deg in weighted_degrees.items():
        if float(deg) < 50:  # type: ignore[arg-type]
            G.nodes[node]["connection_route"] = False

    out = settings.rp_graphml_path / "all_routes.graphml"
    save_graphml(G, out)  # type: ignore[arg-type]


def run(settings: Settings | None = None, cutoff: int = 90) -> None:
    """Execute the full route pathfinding pipeline."""
    settings = settings or Settings()
    compute_community_routes(settings, cutoff=cutoff)
    compute_connecting_routes(settings, cutoff=cutoff)


if __name__ == "__main__":
    run()
