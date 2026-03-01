"""Port of route_planning/helpers/route_planning_helper.py.

Note: graphs loaded via graph_io.load_graphml() are DiGraph (not MultiDiGraph).
Edge access is G.adj[u][v][attr] with no key layer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    import pandas as pd


def assign_community_labels(G: nx.MultiDiGraph, labels: list[int]) -> None:
    """Add a 'community' attribute to each node using the provided label list.

    Args:
        G: Road network graph.
        labels: Community label per node (same order as G.nodes).
    """
    for node, label in zip(G.nodes, labels):
        G.nodes[node]["community"] = label


def convert_edge_weights_to_floats(G: nx.MultiDiGraph) -> None:
    """Convert string edge weight attributes to floats in-place.

    Args:
        G: Graph whose edge weights need conversion.
    """
    weight_attributes = nx.get_edge_attributes(G, "weight")
    weight_attributes = {k: {"weight": float(v)} for k, v in weight_attributes.items()}
    nx.set_edge_attributes(G, weight_attributes)


def split_into_community_graphs(
    G: nx.MultiDiGraph,
) -> list[nx.MultiDiGraph]:
    """Create a subgraph for each unique community label.

    Args:
        G: Graph with community-labelled nodes.

    Returns:
        List of community subgraphs.
    """
    community_labels = set(
        data["community"] for _, data in G.nodes(data=True) if "community" in data
    )
    community_graphs: list[nx.MultiDiGraph] = []
    for label in sorted(community_labels):
        community_nodes = [
            n for n, d in G.nodes(data=True) if d.get("community") == label
        ]
        community_graphs.append(G.subgraph(community_nodes))  # type: ignore[arg-type]
    return community_graphs


def get_n_highest_ranked_nodes_in_community(
    community_df: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """Label the n highest-ranked nodes in a community DataFrame.

    Args:
        community_df: DataFrame for a single community.
        n: Number of top-ranked nodes to flag.

    Returns:
        DataFrame with a top_n column assigned.
    """
    top_n = list(community_df.nlargest(n, "rank")["osmid"].index)
    community_df = community_df.copy()
    community_df["top_n"] = 0
    for i, node_id in enumerate(top_n, start=1):
        community_df.loc[community_df["osmid"] == node_id, "top_n"] = i

    ranks = list(range(1, n + 1))
    check = community_df[community_df["top_n"].isin(ranks)]
    assert len(check) == n
    return community_df


def get_top_n_ranked_nodes_per_community(
    G: nx.MultiDiGraph, community_labels: list[int]
) -> list[dict[int, list[int]]]:
    """Get nodes with a top_n attribute from the given communities.

    Args:
        G: Graph with ranked nodes.
        community_labels: Community labels to include.

    Returns:
        List of {label: [node_ids]} dicts.
    """
    ranks_raw = nx.get_node_attributes(G, "top_n")
    unique_ranks = set(ranks_raw.values())
    unique_ranks.discard(0)
    unique_ranks.discard("0")

    top_ranked_per_community: list[dict[int, list[int]]] = []
    for label in community_labels:
        top_n_nodes = [
            x
            for x, y in G.nodes(data=True)
            if y.get("community") == label
            and int(y.get("top_n", 0)) in {int(r) for r in unique_ranks}
        ]
        top_ranked_per_community.append({label: top_n_nodes})
    return top_ranked_per_community


def get_node_coordinates(
    G: nx.MultiDiGraph, communities_nodes: list[dict[int, list[int]]]
) -> list[dict[int, list[dict[str, int | float]]]]:
    """Get lat/lng coordinates for nodes in each community.

    Args:
        G: Road network graph.
        communities_nodes: Output of get_top_n_ranked_nodes_per_community.

    Returns:
        List of {label: [{osmid, y, x}]} dicts.
    """
    node_coordinates: list[dict[int, list[dict[str, int | float]]]] = []
    for community in communities_nodes:
        label = next(iter(community.keys()))
        nodes = next(iter(community.values()))
        coordinates: list[dict[str, int | float]] = [
            {
                "osmid": node,
                "y": float(G.nodes[node]["y"]),
                "x": float(G.nodes[node]["x"]),
            }
            for node in nodes
        ]
        node_coordinates.append({label: coordinates})
    return node_coordinates


def find_furthest_apart_nodes(
    node_coordinates: list[dict[int, list[dict[str, int | float]]]],
) -> list[dict[int, dict[str, int | float]]]:
    """Find the pair of nodes furthest apart in each community.

    Args:
        node_coordinates: Output of get_node_coordinates.

    Returns:
        List of {label: {u, v, dist}} dicts.
    """
    max_distances: list[dict[int, dict[str, int | float]]] = []
    for coordinates_dict in node_coordinates:
        label = next(iter(coordinates_dict.keys()))
        coordinates = next(iter(coordinates_dict.values()))
        results: list[dict[str, int | float]] = []
        for i in range(len(coordinates)):
            start = coordinates[i]
            for j in range(i + 1, len(coordinates)):
                end = coordinates[j]
                dist = math.hypot(
                    float(start["y"]) - float(end["y"]),
                    float(start["x"]) - float(end["x"]),
                )
                results.append({"u": start["osmid"], "v": end["osmid"], "dist": dist})
        max_distance = max(results, key=lambda r: r["dist"])
        max_distances.append({label: max_distance})
    return max_distances


def greatest_distance_between_top_ranked_nodes(
    G: nx.MultiDiGraph, community_labels: list[int]
) -> list[dict[int, dict[str, int | float]]]:
    """Find the furthest-apart pair among top-ranked nodes per community.

    Args:
        G: Graph with ranked nodes.
        community_labels: Communities to process.

    Returns:
        List of {label: {u, v, dist}} dicts.
    """
    top_ranked_per_community = get_top_n_ranked_nodes_per_community(G, community_labels)
    assert len(top_ranked_per_community) == len(community_labels)
    top_ranked_node_coordinates = get_node_coordinates(G, top_ranked_per_community)
    assert len(top_ranked_node_coordinates) == len(community_labels)
    max_distance_per_community = find_furthest_apart_nodes(top_ranked_node_coordinates)
    assert len(max_distance_per_community) == len(community_labels)
    return max_distance_per_community


def assign_route_start_end_points(
    G: nx.MultiDiGraph,
    route_nodes: list[dict[int, dict[str, int]]],
    n_communities: int,
) -> tuple[list[dict], list[dict]]:  # type: ignore[type-arg]
    """Flag route start (1) and end (2) nodes in the graph.

    Args:
        G: Road network graph.
        route_nodes: List of {label: {u, v}} dicts.
        n_communities: Expected number of communities for assertion.

    Returns:
        Tuple of (start_nodes, end_nodes).
    """
    for community in route_nodes:
        node_dict = next(iter(community.values()))
        G.nodes[node_dict["u"]]["route_flag"] = 1
        G.nodes[node_dict["v"]]["route_flag"] = 2

    u_nodes = [data for _, data in G.nodes(data=True) if data.get("route_flag") == 1]
    v_nodes = [data for _, data in G.nodes(data=True) if data.get("route_flag") == 2]
    assert len(u_nodes) == n_communities and len(v_nodes) == n_communities
    return u_nodes, v_nodes


def path_weight(G: nx.MultiDiGraph, path: list[int], weight: str = "weight") -> float:
    """Calculate the total weight of a path through the graph.

    Args:
        G: Road network graph.
        path: Ordered list of node IDs.
        weight: Edge attribute to sum.

    Returns:
        Total path weight.
    """
    return sum(
        float(G.adj[u][v][weight])  # type: ignore[index]
        for u, v in zip(path, path[1:])
    )


def find_highest_weighted_simple_path(
    G: nx.MultiDiGraph,
    cutoff: int = 90,
    start_node: int | None = None,
    end_node: int | None = None,
) -> list[int]:
    """Find the simple path with highest total edge weight between two nodes.

    Args:
        G: Road network graph.
        cutoff: Maximum path depth for search.
        start_node: Starting node ID (or detected from route_flag=1).
        end_node: Ending node ID (or detected from route_flag=2).

    Returns:
        List of node IDs forming the highest-weight path.
    """
    if start_node is None:
        start_node = next(
            n for n, d in G.nodes(data=True) if str(d.get("route_flag")) == "1"
        )
    if end_node is None:
        end_node = next(
            n for n, d in G.nodes(data=True) if str(d.get("route_flag")) == "2"
        )
    result = max(
        nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=cutoff),
        key=lambda p: path_weight(G, p),  # type: ignore[arg-type]
    )
    return [int(n) for n in result]  # type: ignore[arg-type]
