"""GTFS parsing, actual stop injection, and merged test graph preparation.

Run: python -m donegal_bus.offline.prepare_test_data
"""

from donegal_bus.config import Settings


def parse_gtfs_stops(settings: Settings) -> None:
    """Parse LocalLink GTFS data and extract Donegal bus stops.

    Reads GTFS txt files from testing_data_path (or google_transit_locallink.zip),
    writes donegal_stops_updated.csv.
    Result: 122 actual Donegal LocalLink bus stops.
    """
    # TODO: Port from add_locallink_bus_stops_to_graph.ipynb (cells 1-6)
    #
    # The GTFS data is already extracted to data/locallink/*.txt
    # No need for partridge library — use pandas directly:
    #
    # import pandas as pd
    # agency = pd.read_csv(settings.locallink_data_path / "agency.txt")
    # routes = pd.read_csv(settings.locallink_data_path / "routes.txt")
    # trips = pd.read_csv(settings.locallink_data_path / "trips.txt")
    # stop_times = pd.read_csv(settings.locallink_data_path / "stop_times.txt")
    # stops = pd.read_csv(settings.locallink_data_path / "stops.txt")
    #
    # Filter for Donegal routes (agency_id == "LLDL")
    # donegal_routes = routes[routes["agency_id"] == "LLDL"]
    # Filter trips by route_id, merge with stop_times, then with stops
    # Remove Derry stops (cross-border), write donegal_stops_updated.csv
    raise NotImplementedError


def add_actual_stops_to_graph(settings: Settings) -> None:
    """Add actual LocalLink bus stops as nodes in the road network.

    Reads donegal_stops_updated.csv and all_routes.graphml,
    writes actual_routes_added.graphml.

    osmnx v0.15 → v2.1 migration:
      OLD: ox.get_nearest_node(G, (lat, lng))
      NEW: ox.nearest_nodes(G, X=lng, Y=lat)  (note X=longitude first)
    """
    # TODO: Port from add_locallink_bus_stops_to_graph.ipynb (cells 7-14)
    #
    # import pandas as pd
    # import osmnx as ox
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    # from donegal_bus.helpers.testing import get_path_of_route
    #
    # G = load_graphml(settings.rp_graphml_path / "all_routes.graphml")
    # fix_edge_weights(G)
    # df = pd.read_csv(settings.testing_data_path / "donegal_stops_updated.csv")
    #
    # For each unique stop, find nearest graph node:
    # node_id = ox.nearest_nodes(G, X=stop["stop_lon"], Y=stop["stop_lat"])
    # G.nodes[node_id]["actual_stop"] = True
    # G.nodes[node_id]["actual_route_order"] = f"LL{route_num}-{seq}"
    # (comma-separate multiple routes on same node)
    #
    # save_graphml(G, settings.testing_graphml_path / "actual_routes_added.graphml")
    raise NotImplementedError


def build_merged_test_graph(settings: Settings) -> None:
    """Merge generated and actual routes into a single test graph.

    Reads actual_routes_added.graphml,
    writes testing_graph.graphml and merged_routes.graphml.
    Final graph: 539 generated + 122 actual stops merged onto full Donegal road network.
    """
    # TODO: Port from testing/notebooks/prepare_test_graph.ipynb (cells 1-10)
    #
    # import osmnx as ox
    # import geopandas as gpd
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    #
    # test_G = ox.graph_from_place("Donegal, Ireland", network_type="all")
    # actual = settings.testing_graphml_path / "actual_routes_added.graphml"
    # route_G = load_graphml(actual)
    # fix_edge_weights(route_G)
    #
    # Spatial merge: gpd.sjoin on node geometry to transfer attributes
    # For unmatched nodes: use ox.nearest_nodes(test_G, X=lng, Y=lat)
    # Transfer: actual_stop, community_route, connection_route, *_order attrs
    #
    # save_graphml(test_G, settings.testing_graphml_path / "testing_graph.graphml")
    # save_graphml(route_G, settings.testing_graphml_path / "merged_routes.graphml")
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full test data preparation pipeline."""
    settings = settings or Settings()
    parse_gtfs_stops(settings)
    add_actual_stops_to_graph(settings)
    build_merged_test_graph(settings)


if __name__ == "__main__":
    run()
