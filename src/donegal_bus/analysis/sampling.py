"""Cochran's formula sample sizing and node selection."""

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
    raise NotImplementedError


def select_sample_nodes(G: nx.MultiDiGraph, sample_size: int) -> list[int]:
    """Randomly select sample nodes from the graph for accessibility testing.

    Args:
        G: Road network graph.
        sample_size: Number of nodes to select.

    Returns:
        List of selected node IDs.
    """
    raise NotImplementedError
