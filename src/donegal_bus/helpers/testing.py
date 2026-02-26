"""Port of testing/helpers/testing_helper.py — GTFS parsing and path utilities."""

import networkx as nx
import pandas as pd


def get_path_of_route(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the route path from GTFS trip data.

    Finds the trip with the highest stop_sequence and returns
    all stops on that trip in order.

    Args:
        df: DataFrame with columns trip_id, stop_id, stop_sequence, route_id.

    Returns:
        DataFrame with columns route_id, stop_id, stop_sequence.
    """
    raise NotImplementedError


def find_shortest_path_to_destinations(
    G: nx.MultiDiGraph,
    source: int,
    destinations: list[int],
    weight: str = "length",
) -> tuple[list[int] | None, float | None]:
    """Find Dijkstra's shortest path from source to the nearest destination.

    Args:
        G: Road network graph.
        source: Starting node ID.
        destinations: List of target node IDs.
        weight: Edge attribute to minimise.

    Returns:
        Tuple of (shortest_path, path_weight), or (None, None) if unreachable.
    """
    raise NotImplementedError


def sample_size(
    population_size: int,
    margin_error: float = 0.05,
    confidence_level: float = 0.99,
    sigma: float = 0.5,
) -> float:
    """Calculate minimum sample size using Cochran's formula.

    Args:
        population_size: Total population size.
        margin_error: Maximum acceptable margin of error.
        confidence_level: Desired confidence level (0–1).
        sigma: Population standard deviation estimate.

    Returns:
        Required sample size.
    """
    raise NotImplementedError
