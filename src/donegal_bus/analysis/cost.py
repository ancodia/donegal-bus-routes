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
    total_length_m = 0.0
    for u, v in zip(path, path[1:]):
        edge_data = G.adj.get(u, {}).get(v, {})
        # MultiDiGraph: edge_data is {key: attrs}; take first key (0)
        if isinstance(edge_data, dict) and edge_data:
            first_edge = next(iter(edge_data.values()))
            total_length_m += float(first_edge.get("length", 0))

    distance_km = total_length_m / 1000.0
    fuel_litres = distance_km * fuel_consumption_per_km
    cost_eur = fuel_litres * fuel_cost_per_litre
    return RouteCost(
        route_community=community,
        distance_km=distance_km,
        estimated_fuel_litres=fuel_litres,
        estimated_cost_eur=cost_eur,
    )


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
    return CostComparison(
        generated_routes=generated,
        actual_routes=actual,
        total_generated_cost=sum(r.estimated_cost_eur for r in generated),
        total_actual_cost=sum(r.estimated_cost_eur for r in actual),
    )
