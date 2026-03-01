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
    max_number_stops = df["stop_sequence"].max()
    trip_id = df[df["stop_sequence"] == max_number_stops]["trip_id"].iloc[0]  # type: ignore[union-attr]
    df_filtered = df[df["trip_id"] == trip_id]
    return df_filtered[["route_id", "stop_id", "stop_sequence"]]  # type: ignore[return-value]


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
    shortest_path: list[int] | None = None
    shortest_path_weight: float | None = None

    for dest in destinations:
        try:
            for path in nx.all_shortest_paths(G, source, dest, weight=weight):
                pw = sum(
                    float(G.adj[u][v][weight])  # type: ignore[index]
                    for u, v in zip(path, path[1:])
                )
                if shortest_path_weight is None or pw < shortest_path_weight:
                    shortest_path = [int(n) for n in path]  # type: ignore[arg-type]
                    shortest_path_weight = pw
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    return shortest_path, shortest_path_weight


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
    alpha = 1 - confidence_level
    zdict = {
        0.90: 1.645,
        0.91: 1.695,
        0.92: 1.751,
        0.93: 1.812,
        0.94: 1.881,
        0.95: 1.96,
        0.96: 2.054,
        0.97: 2.17,
        0.98: 2.326,
        0.99: 2.576,
    }
    if confidence_level in zdict:
        z = zdict[confidence_level]
    else:
        from scipy.stats import norm  # type: ignore[import-untyped]

        z = norm.ppf(1 - (alpha / 2))
    N = population_size
    M = margin_error
    numerator = z**2 * sigma**2 * (N / (N - 1))
    denom = M**2 + ((z**2 * sigma**2) / (N - 1))
    return float(numerator / denom)  # type: ignore[arg-type]
