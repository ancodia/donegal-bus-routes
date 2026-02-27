"""Centralised GraphML I/O and GeoJSON conversion utilities."""

from pathlib import Path

import networkx as nx

from donegal_bus.models.geojson import (
    Feature,
    FeatureCollection,
    LineStringGeometry,
    PointGeometry,
)
from donegal_bus.models.graph import GraphSummary


def load_graphml(path: Path) -> nx.MultiDiGraph:
    """Load a GraphML file and return a NetworkX MultiDiGraph."""
    G = nx.read_graphml(str(path))
    # nx.read_graphml returns string node IDs; relabel to int to match
    # the integer osmids used throughout the codebase.
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes})
    return G  # type: ignore[return-value]


def save_graphml(G: nx.MultiDiGraph, path: Path) -> None:
    """Save a NetworkX MultiDiGraph to a GraphML file."""
    nx.write_graphml(G, str(path))


def fix_edge_weights(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Convert string edge weights to floats (GraphML stores all attrs as str)."""
    for _u, _v, data in G.edges(data=True):
        if "weight" in data:
            data["weight"] = float(data["weight"])
        if "length" in data:
            data["length"] = float(data["length"])
    return G


def fix_bool_attributes(G: nx.MultiDiGraph, attr: str) -> nx.MultiDiGraph:
    """Convert string boolean attributes ('True'/'False') to actual bools."""
    for _node, data in G.nodes(data=True):
        if attr in data:
            val = data[attr]
            if val == "True":
                data[attr] = True
            elif val == "False":
                data[attr] = False
    return G


def graph_summary(G: nx.MultiDiGraph) -> GraphSummary:
    """Return a summary of the graph's nodes, edges, and communities."""
    communities: set[object] = set()
    for _node, data in G.nodes(data=True):
        if "community" in data:
            communities.add(data["community"])
    return GraphSummary(
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        num_communities=len(communities),
        is_connected=nx.is_weakly_connected(G),
    )


def nodes_to_geojson(G: nx.MultiDiGraph) -> FeatureCollection:
    """Convert all graph nodes to a GeoJSON FeatureCollection of Points."""
    features: list[Feature] = []
    for node, data in G.nodes(data=True):
        geom = PointGeometry(coordinates=(float(data["x"]), float(data["y"])))
        props: dict[str, str | int | float | bool | None] = {
            "osmid": int(data.get("osmid", node)),
        }
        for key in (
            "community",
            "rank",
            "top_n",
            "route_flag",
            "community_route",
            "connection_route",
            "actual_stop",
        ):
            if key in data:
                props[key] = data[key]
        features.append(Feature(geometry=geom, properties=props))
    return FeatureCollection(features=features)


def edges_to_geojson(G: nx.MultiDiGraph) -> FeatureCollection:
    """Convert all graph edges to a GeoJSON FeatureCollection of LineStrings."""
    features: list[Feature] = []
    for u, v, data in G.edges(data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        coords = [
            (float(u_data["x"]), float(u_data["y"])),
            (float(v_data["x"]), float(v_data["y"])),
        ]
        geom = LineStringGeometry(coordinates=coords)
        props: dict[str, str | int | float | bool | None] = {
            "u": int(u) if isinstance(u, str) else u,
            "v": int(v) if isinstance(v, str) else v,
            "weight": float(data.get("weight", 0)),
            "length": float(data.get("length", 0)),
        }
        features.append(Feature(geometry=geom, properties=props))
    return FeatureCollection(features=features)


def route_path_to_geojson(G: nx.MultiDiGraph, path: list[int]) -> FeatureCollection:
    """Convert a route path (list of node IDs) to a GeoJSON FeatureCollection."""
    coords = [(float(G.nodes[node]["x"]), float(G.nodes[node]["y"])) for node in path]
    geom = LineStringGeometry(coordinates=coords)
    feature = Feature(geometry=geom, properties={"num_stops": len(path)})
    return FeatureCollection(features=[feature])
