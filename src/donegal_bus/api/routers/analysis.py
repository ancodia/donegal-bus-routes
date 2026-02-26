"""Analysis and comparison endpoints."""

from fastapi import APIRouter, Depends
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_routes_graph
from donegal_bus.models.analysis import AccessibilitySummary, CostComparison
from donegal_bus.models.population import PopulationSummary

router = APIRouter()


@router.get("/accessibility")
def accessibility_summary(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> AccessibilitySummary:
    """Return accessibility test results across sampled nodes."""
    raise NotImplementedError


@router.get("/cost")
def cost_comparison(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> CostComparison:
    """Return fuel cost comparison between generated and actual routes."""
    raise NotImplementedError


@router.get("/population")
def population_summary(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> PopulationSummary:
    """Return population coverage summary for the route network."""
    raise NotImplementedError
