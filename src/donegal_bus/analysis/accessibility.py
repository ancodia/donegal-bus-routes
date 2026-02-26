"""Dijkstra-based accessibility testing per sample node."""

import networkx as nx

from donegal_bus.models.analysis import AccessibilitySummary, AccessibilityTestResult


def test_single_node(
    G: nx.MultiDiGraph,
    source: int,
    destinations: list[int],
    weight: str = "length",
) -> AccessibilityTestResult:
    """Test accessibility from a single source node to all destinations.

    Args:
        G: Road network graph with routes.
        source: Node ID to test from.
        destinations: Target node IDs (e.g. bus stops).
        weight: Edge attribute for path cost.

    Returns:
        Test result with reach count and average path length.
    """
    raise NotImplementedError


def run_accessibility_tests(
    G: nx.MultiDiGraph,
    sample_nodes: list[int],
    destinations: list[int],
    weight: str = "length",
) -> list[AccessibilityTestResult]:
    """Run accessibility tests for all sample nodes.

    Args:
        G: Road network graph with routes.
        sample_nodes: Nodes to test from.
        destinations: Target node IDs.
        weight: Edge attribute for path cost.

    Returns:
        List of test results, one per sample node.
    """
    raise NotImplementedError


def summarize_results(results: list[AccessibilityTestResult]) -> AccessibilitySummary:
    """Aggregate individual test results into a summary.

    Args:
        results: List of per-node test results.

    Returns:
        Summary with averages across all tests.
    """
    raise NotImplementedError
