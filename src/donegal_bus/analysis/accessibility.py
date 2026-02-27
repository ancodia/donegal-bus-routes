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
    # Create a 20km ego graph centred on source to limit search radius
    ego = nx.ego_graph(G, source, radius=20000, distance=weight)
    reachable_dests = [d for d in destinations if d in ego]

    reached = 0
    total_length = 0.0

    for dest in reachable_dests:
        try:
            length = nx.shortest_path_length(ego, source, dest, weight=weight)
            reached += 1
            total_length += float(length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    avg = total_length / reached if reached > 0 else 0.0
    return AccessibilityTestResult(
        source_node=source,
        destinations_reached=reached,
        destinations_total=len(destinations),
        avg_path_length=avg,
    )


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
    return [
        test_single_node(G, source, destinations, weight) for source in sample_nodes
    ]


def summarize_results(results: list[AccessibilityTestResult]) -> AccessibilitySummary:
    """Aggregate individual test results into a summary.

    Args:
        results: List of per-node test results.

    Returns:
        Summary with averages across all tests.
    """
    if not results:
        return AccessibilitySummary(
            total_tests=0,
            avg_destinations_reached=0.0,
            avg_path_length=0.0,
            results=[],
        )
    avg_reached = sum(r.destinations_reached for r in results) / len(results)
    avg_length = sum(r.avg_path_length for r in results) / len(results)
    return AccessibilitySummary(
        total_tests=len(results),
        avg_destinations_reached=avg_reached,
        avg_path_length=avg_length,
        results=results,
    )
