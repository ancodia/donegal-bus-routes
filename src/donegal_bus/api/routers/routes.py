"""Route query endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_routes_graph
from donegal_bus.models.error import ErrorResponse
from donegal_bus.models.geojson import Feature, FeatureCollection, LineStringGeometry
from donegal_bus.models.route import RouteCollection, RouteDetail, RouteStop

router = APIRouter()


def _extract_community_routes(
    G: MultiDiGraph,
) -> dict[int, list[tuple[int, dict]]]:  # type: ignore[type-arg]
    """Return a dict mapping community label → sorted list of (node, data) tuples."""
    from collections import defaultdict

    community_nodes: dict[int, list[tuple[int, dict]]] = defaultdict(list)  # type: ignore[type-arg]
    for node, data in G.nodes(data=True):
        if data.get("community_route") is True:
            label = int(data.get("community", 0))
            community_nodes[label].append((node, data))

    for nodes in community_nodes.values():
        nodes.sort(key=lambda nd: int(nd[1].get("community_route_order", 0)))

    return dict(community_nodes)


def _route_detail_from_nodes(
    community: int,
    nodes: list[tuple[int, dict]],  # type: ignore[type-arg]
) -> RouteDetail:
    stops = [
        RouteStop(
            osmid=int(data.get("osmid", node)),
            x=float(data["x"]),
            y=float(data["y"]),
            sequence=int(data.get("community_route_order", i)),
        )
        for i, (node, data) in enumerate(nodes)
    ]
    total_weight = sum(float(data.get("rank", 0)) for _, data in nodes)
    return RouteDetail(community=community, stops=stops, total_weight=total_weight)


@router.get("/")
def list_routes(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> RouteCollection:
    """Return all generated routes."""
    routes_by_community = _extract_community_routes(G)
    routes = [
        _route_detail_from_nodes(label, nodes)
        for label, nodes in sorted(routes_by_community.items())
    ]
    return RouteCollection(routes=routes, total_routes=len(routes))


@router.get("/geojson")
def routes_geojson(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return all routes as a GeoJSON FeatureCollection of LineStrings."""
    routes_by_community = _extract_community_routes(G)
    features: list[Feature] = []
    for label, nodes in sorted(routes_by_community.items()):
        coords = [(float(d["x"]), float(d["y"])) for _, d in nodes]
        geom = LineStringGeometry(coordinates=coords)
        features.append(
            Feature(
                geometry=geom,
                properties={"community": label, "num_stops": len(nodes)},
            )
        )
    return FeatureCollection(features=features)


@router.get("/community/{label}", responses={404: {"model": ErrorResponse}})
def routes_by_community(
    label: int,
    G: MultiDiGraph = Depends(get_routes_graph),
) -> RouteDetail:
    """Return the route for a specific community."""
    routes_by_community_map = _extract_community_routes(G)
    if label not in routes_by_community_map:
        raise HTTPException(
            status_code=404, detail=f"Route for community {label} not found"
        )
    return _route_detail_from_nodes(label, routes_by_community_map[label])


@router.get("/connection/{label}", responses={404: {"model": ErrorResponse}})
def connection_route(
    label: str,
    G: MultiDiGraph = Depends(get_routes_graph),
) -> RouteDetail:
    """Return the connecting route for connection label (a, b, c, or d)."""
    conn_nodes = [
        (node, data)
        for node, data in G.nodes(data=True)
        if data.get("connection_route") is True
        and label in str(data.get("connection", ""))
    ]
    if not conn_nodes:
        raise HTTPException(
            status_code=404, detail=f"Connection route '{label}' not found"
        )

    def _conn_order(nd: tuple[int, dict]) -> int:  # type: ignore[type-arg]
        raw = str(nd[1].get("connection_order", "0-0"))
        parts = raw.split("-")
        return int(parts[-1]) if parts[-1].isdigit() else 0

    conn_nodes.sort(key=_conn_order)
    stops = [
        RouteStop(
            osmid=int(data.get("osmid", node)),
            x=float(data["x"]),
            y=float(data["y"]),
            sequence=_conn_order((node, data)),
        )
        for node, data in conn_nodes
    ]
    total_weight = sum(float(data.get("rank", 0)) for _, data in conn_nodes)
    # Use -1 as community for connection routes (they span communities)
    return RouteDetail(community=-1, stops=stops, total_weight=total_weight)
