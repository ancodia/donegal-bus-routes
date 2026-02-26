"""Route query endpoints."""

from fastapi import APIRouter, Depends
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_routes_graph
from donegal_bus.models.geojson import FeatureCollection
from donegal_bus.models.route import RouteCollection, RouteDetail

router = APIRouter()


@router.get("/")
def list_routes(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> RouteCollection:
    """Return all generated routes."""
    raise NotImplementedError


@router.get("/geojson")
def routes_geojson(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return all routes as a GeoJSON FeatureCollection of LineStrings."""
    raise NotImplementedError


@router.get("/community/{label}")
def routes_by_community(
    label: int,
    G: MultiDiGraph = Depends(get_routes_graph),
) -> RouteDetail:
    """Return the route for a specific community."""
    raise NotImplementedError


@router.get("/connection/{label}")
def connection_route(
    label: int,
    G: MultiDiGraph = Depends(get_routes_graph),
) -> RouteDetail:
    """Return the connecting route for a specific community pair."""
    raise NotImplementedError
