"""Bus stop endpoints."""

from fastapi import APIRouter, Depends
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_routes_graph
from donegal_bus.models.geojson import FeatureCollection

router = APIRouter()


@router.get("/generated")
def generated_stops(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return generated bus stop locations as GeoJSON Points."""
    raise NotImplementedError


@router.get("/actual")
def actual_stops(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return actual LocalLink bus stop locations as GeoJSON Points."""
    raise NotImplementedError


@router.get("/nearest")
def nearest_stop(
    lat: float,
    lng: float,
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return the nearest bus stop to the given coordinates."""
    raise NotImplementedError
