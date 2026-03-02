"""Graph introspection endpoints."""

from fastapi import APIRouter, Depends, Query
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_road_network_graph
from donegal_bus.graph_io import edges_to_geojson, graph_summary, nodes_to_geojson
from donegal_bus.models.geojson import (
    Feature,
    FeatureCollection,
    LineStringGeometry,
    PointGeometry,
)
from donegal_bus.models.graph import GraphSummary

router = APIRouter()


def _filter_bbox(
    fc: FeatureCollection,
    min_lng: float,
    min_lat: float,
    max_lng: float,
    max_lat: float,
) -> FeatureCollection:
    """Filter a FeatureCollection to features within the given bounding box."""
    filtered: list[Feature] = []
    for f in fc.features:
        if isinstance(f.geometry, PointGeometry):
            lng, lat = f.geometry.coordinates
            if min_lng <= lng <= max_lng and min_lat <= lat <= max_lat:
                filtered.append(f)
        elif isinstance(f.geometry, LineStringGeometry):
            for lng, lat in f.geometry.coordinates:
                if min_lng <= lng <= max_lng and min_lat <= lat <= max_lat:
                    filtered.append(f)
                    break
    return FeatureCollection(features=filtered)


@router.get("/summary")
def get_summary(
    G: MultiDiGraph = Depends(get_road_network_graph),
) -> GraphSummary:
    """Return a high-level summary of the road network graph."""
    return graph_summary(G)


@router.get("/nodes")
def get_nodes(
    G: MultiDiGraph = Depends(get_road_network_graph),
    min_lng: float | None = Query(None, description="Bounding box minimum longitude"),
    min_lat: float | None = Query(None, description="Bounding box minimum latitude"),
    max_lng: float | None = Query(None, description="Bounding box maximum longitude"),
    max_lat: float | None = Query(None, description="Bounding box maximum latitude"),
) -> FeatureCollection:
    """Return graph nodes as GeoJSON. Optionally filter by bounding box."""
    fc = nodes_to_geojson(G)
    if all(v is not None for v in (min_lng, min_lat, max_lng, max_lat)):
        fc = _filter_bbox(fc, min_lng, min_lat, max_lng, max_lat)  # type: ignore[arg-type]
    return fc


@router.get("/edges")
def get_edges(
    G: MultiDiGraph = Depends(get_road_network_graph),
    min_lng: float | None = Query(None, description="Bounding box minimum longitude"),
    min_lat: float | None = Query(None, description="Bounding box minimum latitude"),
    max_lng: float | None = Query(None, description="Bounding box maximum longitude"),
    max_lat: float | None = Query(None, description="Bounding box maximum latitude"),
) -> FeatureCollection:
    """Return graph edges as GeoJSON. Optionally filter by bounding box."""
    fc = edges_to_geojson(G)
    if all(v is not None for v in (min_lng, min_lat, max_lng, max_lat)):
        fc = _filter_bbox(fc, min_lng, min_lat, max_lng, max_lat)  # type: ignore[arg-type]
    return fc
