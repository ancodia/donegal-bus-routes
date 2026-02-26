"""Shared dependencies: graph loaders (cached) and settings provider."""

from functools import lru_cache

import networkx as nx

from donegal_bus.config import Settings


@lru_cache
def get_settings() -> Settings:
    """Return the application settings (cached)."""
    return Settings()


_road_network: nx.MultiDiGraph | None = None
_routes_graph: nx.MultiDiGraph | None = None


def load_graphs() -> None:
    """Load all graphs into module-level cache. Called during app lifespan startup."""
    raise NotImplementedError


def get_road_network_graph() -> nx.MultiDiGraph:
    """Return the cached road network graph."""
    if _road_network is None:
        raise RuntimeError("Road network graph not loaded — is the app started?")
    return _road_network


def get_routes_graph() -> nx.MultiDiGraph:
    """Return the cached routes graph."""
    if _routes_graph is None:
        raise RuntimeError("Routes graph not loaded — is the app started?")
    return _routes_graph
