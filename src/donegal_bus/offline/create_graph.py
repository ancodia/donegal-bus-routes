"""Download OSM road network for Donegal and simplify to primary roads.

Run: python -m donegal_bus.offline.create_graph
"""

from donegal_bus.config import Settings


def download_osm_network(settings: Settings) -> None:
    """Download the full OSM road network for County Donegal.

    Saves raw graph to graph_graphml_path/donegal_osm.graphml.
    """
    raise NotImplementedError


def simplify_to_primary_roads(settings: Settings) -> None:
    """Simplify the raw OSM graph, keeping only primary road types.

    Reads donegal_osm.graphml, writes donegal_osm_simplified.graphml.
    """
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full graph creation pipeline."""
    settings = settings or Settings()
    download_osm_network(settings)
    simplify_to_primary_roads(settings)


if __name__ == "__main__":
    run()
