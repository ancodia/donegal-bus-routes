"""Fuel cost estimation and comparison between generated and actual routes."""

import networkx as nx

from donegal_bus.models.analysis import CostComparison, RouteCost


def estimate_route_cost(
    G: nx.MultiDiGraph,
    path: list[int],
    community: int,
    fuel_cost_per_litre: float = 1.50,
    fuel_consumption_per_km: float = 0.3,
) -> RouteCost:
    """Estimate the fuel cost for a single route path.

    Args:
        G: Road network graph.
        path: Ordered list of node IDs forming the route.
        community: Community label for this route.
        fuel_cost_per_litre: Fuel price in EUR.
        fuel_consumption_per_km: Litres consumed per km.

    Returns:
        Cost breakdown for the route.
    """
    raise NotImplementedError


def compare_costs(
    generated: list[RouteCost], actual: list[RouteCost]
) -> CostComparison:
    """Compare total costs between generated and actual route sets.

    Args:
        generated: Costs for algorithm-generated routes.
        actual: Costs for existing LocalLink routes.

    Returns:
        Side-by-side cost comparison.
    """
    raise NotImplementedError
