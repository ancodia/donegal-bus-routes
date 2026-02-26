"""Community detection result endpoints."""

from fastapi import APIRouter, Depends
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_road_network_graph
from donegal_bus.models.community import CommunityCollection, CommunityDetail

router = APIRouter()


@router.get("/")
def list_communities(
    G: MultiDiGraph = Depends(get_road_network_graph),
) -> CommunityCollection:
    """Return all detected communities with node/edge counts."""
    raise NotImplementedError


@router.get("/{label}")
def get_community(
    label: int,
    G: MultiDiGraph = Depends(get_road_network_graph),
) -> CommunityDetail:
    """Return detailed info for a single community."""
    raise NotImplementedError
