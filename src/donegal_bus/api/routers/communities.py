"""Community detection result endpoints."""

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_routes_graph
from donegal_bus.models.community import Community, CommunityCollection, CommunityDetail

router = APIRouter()


@router.get("/")
def list_communities(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> CommunityCollection:
    """Return all detected communities with node/edge counts."""
    community_nodes: dict[int, list[int]] = defaultdict(list)
    for node, data in G.nodes(data=True):
        label = int(data.get("community", 0))
        community_nodes[label].append(node)

    communities = []
    for label, nodes in sorted(community_nodes.items()):
        subgraph = G.subgraph(nodes)
        communities.append(
            Community(
                label=label,
                num_nodes=subgraph.number_of_nodes(),
                num_edges=subgraph.number_of_edges(),
            )
        )
    return CommunityCollection(communities=communities, total=len(communities))


@router.get("/{label}")
def get_community(
    label: int,
    G: MultiDiGraph = Depends(get_routes_graph),
) -> CommunityDetail:
    """Return detailed info for a single community."""
    nodes_in_community = [
        n for n, d in G.nodes(data=True) if int(d.get("community", -1)) == label
    ]
    if not nodes_in_community:
        raise HTTPException(status_code=404, detail=f"Community {label} not found")

    subgraph = G.subgraph(nodes_in_community)
    top_ranked = [
        int(d.get("osmid", n))
        for n, d in subgraph.nodes(data=True)
        if int(d.get("top_n", 0)) > 0
    ]
    route_start = next(
        (
            int(d.get("osmid", n))
            for n, d in subgraph.nodes(data=True)
            if int(d.get("route_flag", 0)) == 1
        ),
        None,
    )
    route_end = next(
        (
            int(d.get("osmid", n))
            for n, d in subgraph.nodes(data=True)
            if int(d.get("route_flag", 0)) == 2
        ),
        None,
    )
    return CommunityDetail(
        label=label,
        num_nodes=subgraph.number_of_nodes(),
        num_edges=subgraph.number_of_edges(),
        top_ranked_nodes=top_ranked,
        route_start=route_start,
        route_end=route_end,
    )
