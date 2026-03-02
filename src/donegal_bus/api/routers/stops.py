"""Bus stop endpoints."""

import math

from fastapi import APIRouter, Depends
from networkx import MultiDiGraph

from donegal_bus.api.dependencies import get_routes_graph
from donegal_bus.models.geojson import Feature, FeatureCollection, PointGeometry

router = APIRouter()


@router.get("/generated")
def generated_stops(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return generated bus stop locations as GeoJSON Points."""
    features: list[Feature] = []
    for node, data in G.nodes(data=True):
        if data.get("community_route") is True or data.get("connection_route") is True:
            geom = PointGeometry(coordinates=(float(data["x"]), float(data["y"])))
            props: dict[str, str | int | float | bool | None] = {
                "osmid": int(data.get("osmid", node)),
                "community": int(data.get("community", 0)),
                "community_route": data.get("community_route"),
                "connection_route": data.get("connection_route"),
            }
            features.append(Feature(geometry=geom, properties=props))
    return FeatureCollection(features=features)


@router.get("/actual")
def actual_stops(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return actual LocalLink bus stop locations as GeoJSON Points."""
    features: list[Feature] = []
    for node, data in G.nodes(data=True):
        if data.get("actual_stop") is True:
            geom = PointGeometry(coordinates=(float(data["x"]), float(data["y"])))
            props: dict[str, str | int | float | bool | None] = {
                "osmid": int(data.get("osmid", node)),
                "actual_route_order": str(data.get("actual_route_order", "")),
            }
            features.append(Feature(geometry=geom, properties=props))
    return FeatureCollection(features=features)


@router.get("/nearest")
def nearest_stop(
    lat: float,
    lng: float,
    G: MultiDiGraph = Depends(get_routes_graph),
) -> FeatureCollection:
    """Return the nearest generated bus stop to the given coordinates."""
    stops = [
        (node, data)
        for node, data in G.nodes(data=True)
        if data.get("community_route") is True or data.get("connection_route") is True
    ]
    if not stops:
        return FeatureCollection(features=[])

    nearest_node, nearest_data = min(
        stops,
        key=lambda nd: math.hypot(float(nd[1]["x"]) - lng, float(nd[1]["y"]) - lat),
    )
    dist = math.hypot(float(nearest_data["x"]) - lng, float(nearest_data["y"]) - lat)
    geom = PointGeometry(
        coordinates=(float(nearest_data["x"]), float(nearest_data["y"]))
    )
    props: dict[str, str | int | float | bool | None] = {
        "osmid": int(nearest_data.get("osmid", nearest_node)),
        "community": int(nearest_data.get("community", 0)),
        "distance_deg": round(dist, 6),
    }
    return FeatureCollection(features=[Feature(geometry=geom, properties=props)])
