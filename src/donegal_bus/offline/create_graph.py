"""Download OSM road network for Donegal and simplify to primary roads.

Run: python -m donegal_bus.offline.create_graph
"""

from donegal_bus.config import Settings


def download_osm_network(settings: Settings) -> None:
    """Download the full OSM road network for County Donegal.

    Saves raw graph to graph_graphml_path/donegal_osm.graphml.
    """
    import osmnx as ox  # type: ignore[import-untyped]

    G = ox.graph_from_place("Donegal, Ireland", network_type="drive")
    out = settings.graph_graphml_path / "donegal_osm.graphml"
    ox.save_graphml(G, filepath=str(out))


def simplify_to_primary_roads(settings: Settings) -> None:
    """Simplify the raw OSM graph, keeping only primary road types.

    Reads donegal_osm.graphml, writes donegal_osm_simplified.graphml.
    Keeps edges whose highway tag is primary, secondary, tertiary, or trunk.
    """
    import networkx as nx
    import osmnx as ox  # type: ignore[import-untyped]

    raw = settings.graph_graphml_path / "donegal_osm.graphml"
    G = ox.load_graphml(filepath=str(raw))

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    keep_types = {"primary", "secondary", "tertiary", "trunk"}

    def _is_keep(hw: object) -> bool:
        if isinstance(hw, list):
            return any(t in keep_types for t in hw)
        return str(hw) in keep_types

    mask = edges["highway"].apply(_is_keep)
    exclude = edges[~mask]

    edge_tuples = list(
        exclude[["u", "v"]].itertuples(index=False, name=None)  # type: ignore[union-attr]
    )
    G.remove_edges_from(edge_tuples)
    G.remove_nodes_from(list(nx.isolates(G)))

    out = settings.graph_graphml_path / "donegal_osm_simplified.graphml"
    ox.save_graphml(G, filepath=str(out))


def run(settings: Settings | None = None) -> None:
    """Execute the full graph creation pipeline."""
    settings = settings or Settings()
    download_osm_network(settings)
    simplify_to_primary_roads(settings)


if __name__ == "__main__":
    run()
