"""Graph introspection endpoints."""

from fastapi import APIRouter, Depends
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_road_network_graph
from donegal_bus.graph_io import edges_to_geojson, graph_summary, nodes_to_geojson
from donegal_bus.models.geojson import FeatureCollection
from donegal_bus.models.graph import GraphSummary

router = APIRouter()


@router.get("/summary")
def get_summary(
    G: MultiDiGraph = Depends(get_road_network_graph),
) -> GraphSummary:
    """Return a high-level summary of the road network graph."""
    return graph_summary(G)


@router.get("/nodes")
def get_nodes(
    G: MultiDiGraph = Depends(get_road_network_graph),
) -> FeatureCollection:
    """Return all graph nodes as a GeoJSON FeatureCollection."""
    return nodes_to_geojson(G)


@router.get("/edges")
def get_edges(
    G: MultiDiGraph = Depends(get_road_network_graph),
) -> FeatureCollection:
    """Return all graph edges as a GeoJSON FeatureCollection."""
    return edges_to_geojson(G)
