"""Analysis and comparison endpoints."""

from collections import defaultdict

import pandas as pd
from fastapi import APIRouter, Depends
from networkx import MultiDiGraph

from donegal_bus.analysis.accessibility import (
    run_accessibility_tests,
    summarize_results,
)
from donegal_bus.analysis.cost import compare_costs, estimate_route_cost
from donegal_bus.analysis.sampling import cochran_sample_size, select_sample_nodes
from donegal_bus.api.dependencies import get_routes_graph, get_settings
from donegal_bus.models.analysis import AccessibilitySummary, CostComparison
from donegal_bus.models.population import PopulationSummary

router = APIRouter()


@router.get("/accessibility")
def accessibility_summary(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> AccessibilitySummary:
    """Return accessibility test results across sampled nodes."""
    # Generated stop nodes are the destinations
    destinations = [
        int(data.get("osmid", node))
        for node, data in G.nodes(data=True)
        if data.get("community_route") is True or data.get("connection_route") is True
    ]
    # Use a small sample for the API (full Cochran sample can take minutes)
    sample_size = cochran_sample_size(G.number_of_nodes(), confidence_level=0.95)
    sample_nodes = select_sample_nodes(G, sample_size)
    results = run_accessibility_tests(G, sample_nodes, destinations)
    return summarize_results(results)


@router.get("/cost")
def cost_comparison(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> CostComparison:
    """Return fuel cost comparison between generated and actual routes."""
    from donegal_bus.api.routers.routes import _extract_community_routes

    # Generated route costs
    routes_map = _extract_community_routes(G)
    generated_costs = [
        estimate_route_cost(
            G,
            [node for node, _ in nodes],
            community=label,
        )
        for label, nodes in sorted(routes_map.items())
    ]

    # Actual route costs — group actual stop nodes by route ID (e.g. "LL11")
    route_stops: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)  # type: ignore[type-arg]
    for node, data in G.nodes(data=True):
        if data.get("actual_stop") is not True:
            continue
        raw = str(data.get("actual_route_order", ""))
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry or "-" not in entry:
                continue
            route_id, seq_str = entry.rsplit("-", 1)
            if seq_str.isdigit():
                route_stops[route_id].append((node, int(seq_str), data))

    actual_costs = []
    for route_id, stops in sorted(route_stops.items()):
        stops.sort(key=lambda t: t[1])
        path = [t[0] for t in stops]
        # Use community=-1 for actual routes
        actual_costs.append(estimate_route_cost(G, path, community=-1))

    return compare_costs(generated_costs, actual_costs)


@router.get("/population")
def population_summary(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> PopulationSummary:
    """Return population coverage summary for the route network."""
    settings = get_settings()
    df = pd.read_csv(settings.population_csv)
    total_population: int = int(df["population"].sum())  # type: ignore[arg-type]
    townlands_with_coords: int = int(df["lat"].notna().sum())  # type: ignore[arg-type]
    return PopulationSummary(
        total_townlands=len(df),
        total_population=total_population,
        townlands_with_coords=townlands_with_coords,
    )
