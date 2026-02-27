"""Cochran's formula sample sizing and node selection."""

import math
import random

import networkx as nx


def cochran_sample_size(
    population_size: int,
    margin_error: float = 0.05,
    confidence_level: float = 0.99,
    sigma: float = 0.5,
) -> int:
    """Calculate minimum sample size using Cochran's formula.

    Args:
        population_size: Total population to sample from.
        margin_error: Maximum acceptable margin of error.
        confidence_level: Desired confidence level (0–1).
        sigma: Estimated population standard deviation.

    Returns:
        Required sample size (rounded up).
    """
    # z-score lookup table for common confidence levels
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

        alpha = 1 - confidence_level
        z = norm.ppf(1 - (alpha / 2))
    N = population_size
    M = margin_error
    numerator = z**2 * sigma**2 * (N / (N - 1))
    denom = M**2 + ((z**2 * sigma**2) / (N - 1))
    return math.ceil(numerator / denom)


def select_sample_nodes(G: nx.MultiDiGraph, sample_size: int) -> list[int]:
    """Randomly select sample nodes from the graph for accessibility testing.

    Args:
        G: Road network graph.
        sample_size: Number of nodes to select.

    Returns:
        List of selected node IDs.
    """
    # Exclude bus stop nodes so we only sample non-stop nodes
    all_nodes = [
        int(data.get("osmid", node))
        for node, data in G.nodes(data=True)
        if not data.get("community_route")
        and not data.get("connection_route")
        and not data.get("actual_stop")
    ]
    return random.sample(all_nodes, min(sample_size, len(all_nodes)))
