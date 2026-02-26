"""Centralised GraphML I/O and GeoJSON conversion utilities."""

from pathlib import Path

import networkx as nx

from donegal_bus.models.geojson import FeatureCollection
from donegal_bus.models.graph import GraphSummary


def load_graphml(path: Path) -> nx.MultiDiGraph:
    """Load a GraphML file and return a NetworkX MultiDiGraph."""
    raise NotImplementedError


def save_graphml(G: nx.MultiDiGraph, path: Path) -> None:
    """Save a NetworkX MultiDiGraph to a GraphML file."""
    raise NotImplementedError


def fix_edge_weights(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Convert string edge weights to floats (GraphML stores all attrs as str)."""
    raise NotImplementedError


def fix_bool_attributes(G: nx.MultiDiGraph, attr: str) -> nx.MultiDiGraph:
    """Convert string boolean attributes ('True'/'False') to actual bools."""
    raise NotImplementedError


def graph_summary(G: nx.MultiDiGraph) -> GraphSummary:
    """Return a summary of the graph's nodes, edges, and communities."""
    raise NotImplementedError


def nodes_to_geojson(G: nx.MultiDiGraph) -> FeatureCollection:
    """Convert all graph nodes to a GeoJSON FeatureCollection of Points."""
    raise NotImplementedError


def edges_to_geojson(G: nx.MultiDiGraph) -> FeatureCollection:
    """Convert all graph edges to a GeoJSON FeatureCollection of LineStrings."""
    raise NotImplementedError


def route_path_to_geojson(G: nx.MultiDiGraph, path: list[int]) -> FeatureCollection:
    """Convert a route path (list of node IDs) to a GeoJSON FeatureCollection."""
    raise NotImplementedError
