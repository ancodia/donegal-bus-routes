"""Download OSM road network for Donegal and simplify to primary roads.

Run: python -m donegal_bus.offline.create_graph
"""

from donegal_bus.config import Settings


def download_osm_network(settings: Settings) -> None:
    """Download the full OSM road network for County Donegal.

    Saves raw graph to graph_graphml_path/donegal_osm.graphml.
    """
    # TODO: Port from graph/notebooks/create_graph.ipynb (cells 1-3)
    #
    # import osmnx as ox
    # G = ox.graph_from_place("Donegal, Ireland", network_type="drive")
    # out = settings.graph_graphml_path / "donegal_osm.graphml"
    # ox.save_graphml(G, filepath=str(out))   # osmnx v2.1: filepath kwarg
    raise NotImplementedError


def simplify_to_primary_roads(settings: Settings) -> None:
    """Simplify the raw OSM graph, keeping only primary road types.

    Reads donegal_osm.graphml, writes donegal_osm_simplified.graphml.
    """
    # TODO: Port from graph/notebooks/create_graph.ipynb (cells 4-8)
    #
    # import osmnx as ox
    # import networkx as nx
    # from donegal_bus.graph_io import load_graphml, save_graphml
    #
    # raw = settings.graph_graphml_path / "donegal_osm.graphml"
    # G = ox.load_graphml(filepath=str(raw))
    # nodes, edges = ox.graph_to_gdfs(G)
    # keep_types = {"primary", "secondary", "tertiary", "trunk"}
    # mask = edges["highway"].isin(keep_types)
    # edges = edges[mask]
    # G_simplified = ox.graph_from_gdfs(nodes, edges)
    # G_simplified.remove_nodes_from(list(nx.isolates(G_simplified)))
    # out = settings.graph_graphml_path / "donegal_osm_simplified.graphml"
    # ox.save_graphml(G_simplified, filepath=str(out))
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full graph creation pipeline."""
    settings = settings or Settings()
    download_osm_network(settings)
    simplify_to_primary_roads(settings)


if __name__ == "__main__":
    run()
