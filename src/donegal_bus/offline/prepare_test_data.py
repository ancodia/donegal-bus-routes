"""GTFS parsing, actual stop injection, and merged test graph preparation.

Run: python -m donegal_bus.offline.prepare_test_data
"""

from donegal_bus.config import Settings


def parse_gtfs_stops(settings: Settings) -> None:
    """Parse LocalLink GTFS data and extract Donegal bus stops.

    Reads google_transit_locallink.zip from testing_data_path,
    writes donegal_stops_updated.csv.
    """
    raise NotImplementedError


def add_actual_stops_to_graph(settings: Settings) -> None:
    """Add actual LocalLink bus stops as nodes in the road network.

    Reads donegal_stops_updated.csv and all_routes.graphml,
    writes actual_routes_added.graphml.
    """
    raise NotImplementedError


def build_merged_test_graph(settings: Settings) -> None:
    """Merge generated and actual routes into a single test graph.

    Reads actual_routes_added.graphml,
    writes testing_graph.graphml and merged_routes.graphml.
    """
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full test data preparation pipeline."""
    settings = settings or Settings()
    parse_gtfs_stops(settings)
    add_actual_stops_to_graph(settings)
    build_merged_test_graph(settings)


if __name__ == "__main__":
    run()
