"""Port of route_planning/helpers/route_planning_helper.py."""

from __future__ import annotations

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
    raise NotImplementedError


def convert_edge_weights_to_floats(G: nx.MultiDiGraph) -> None:
    """Convert string edge weight attributes to floats in-place.

    Args:
        G: Graph whose edge weights need conversion.
    """
    raise NotImplementedError


def split_into_community_graphs(
    G: nx.MultiDiGraph,
) -> list[nx.MultiDiGraph]:
    """Create a subgraph for each unique community label.

    Args:
        G: Graph with community-labelled nodes.

    Returns:
        List of community subgraphs.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def find_furthest_apart_nodes(
    node_coordinates: list[dict[int, list[dict[str, int | float]]]],
) -> list[dict[int, dict[str, int | float]]]:
    """Find the pair of nodes furthest apart in each community.

    Args:
        node_coordinates: Output of get_node_coordinates.

    Returns:
        List of {label: {u, v, dist}} dicts.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def assign_route_start_end_points(
    G: nx.MultiDiGraph,
    route_nodes: list[dict[int, dict[str, int]]],
    n_communities: int,
) -> tuple[list[dict], list[dict]]:
    """Flag route start (1) and end (2) nodes in the graph.

    Args:
        G: Road network graph.
        route_nodes: List of {label: {u, v}} dicts.
        n_communities: Expected number of communities for assertion.

    Returns:
        Tuple of (start_nodes, end_nodes).
    """
    raise NotImplementedError


def path_weight(G: nx.MultiDiGraph, path: list[int], weight: str = "weight") -> float:
    """Calculate the total weight of a path through the graph.

    Args:
        G: Road network graph.
        path: Ordered list of node IDs.
        weight: Edge attribute to sum.

    Returns:
        Total path weight.
    """
    raise NotImplementedError


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
    raise NotImplementedError
