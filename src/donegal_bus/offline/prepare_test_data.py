"""GTFS parsing, actual stop injection, and merged test graph preparation.

Run: python -m donegal_bus.offline.prepare_test_data
"""

from donegal_bus.config import Settings


def parse_gtfs_stops(settings: Settings) -> None:
    """Parse LocalLink GTFS data and extract Donegal bus stops.

    Reads GTFS txt files from locallink_data_path,
    writes donegal_stops_updated.csv.
    Result: 122 actual Donegal LocalLink bus stops.
    """
    import pandas as pd

    from donegal_bus.helpers.testing import get_path_of_route

    routes = pd.read_csv(str(settings.locallink_data_path / "routes.txt"))
    trips = pd.read_csv(str(settings.locallink_data_path / "trips.txt"))
    stop_times = pd.read_csv(str(settings.locallink_data_path / "stop_times.txt"))
    stops = pd.read_csv(str(settings.locallink_data_path / "stops.txt"))

    # Filter for Donegal routes (agency_id == "LLDL")
    donegal_routes = routes[routes["agency_id"] == "LLDL"]
    donegal_route_ids = donegal_routes["route_id"].tolist()

    # Join stop_times with trips to get route_id for Donegal trips
    donegal_trips = trips[trips["route_id"].isin(donegal_route_ids)][
        ["trip_id", "route_id"]
    ]
    routes_df = stop_times.merge(donegal_trips, on="trip_id", how="inner")  # type: ignore[arg-type]
    routes_df = routes_df[["trip_id", "route_id", "stop_id", "stop_sequence"]].copy()

    # Remove cross-border Derry stops
    ids_to_exclude = ["7000B158241", "gen:31400:890:0:1"]
    routes_df = routes_df[~routes_df["stop_id"].isin(ids_to_exclude)]  # type: ignore[union-attr]

    # Select the representative trip per route (longest trip by stop_sequence)
    route_dfs = []
    for i, route_id in enumerate(donegal_route_ids, start=1):
        route_subset = routes_df[routes_df["route_id"] == route_id]
        if route_subset.empty:  # type: ignore[union-attr]
            continue
        path_df = get_path_of_route(route_subset).copy()  # type: ignore[arg-type]
        path_df["route_order"] = (
            "LL" + str(i) + "-" + path_df["stop_sequence"].astype(str)
        )
        route_dfs.append(path_df)

    if not route_dfs:
        return

    combined = pd.concat(route_dfs, axis=0, ignore_index=True)
    combined = combined.drop(columns=["route_id", "stop_sequence"], errors="ignore")

    # Add lat/lng from stops table
    combined = combined.merge(
        stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left"
    )
    combined = combined.rename(columns={"stop_lat": "lat", "stop_lon": "lng"})

    settings.testing_data_path.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        str(settings.testing_data_path / "donegal_stops_updated.csv"), index=False
    )


def add_actual_stops_to_graph(settings: Settings) -> None:
    """Add actual LocalLink bus stops as nodes in the road network.

    Reads donegal_stops_updated.csv and all_routes.graphml,
    writes actual_routes_added.graphml.

    osmnx v0.15 → v2.1 migration:
      OLD: ox.get_nearest_node(G, (lat, lng))
      NEW: ox.nearest_nodes(G, X=lng, Y=lat)  (note X=longitude first)
    """
    import networkx as nx
    import osmnx as ox  # type: ignore[import-untyped]
    import pandas as pd

    from donegal_bus.graph_io import fix_edge_weights, load_graphml

    G = load_graphml(settings.rp_graphml_path / "all_routes.graphml")
    fix_edge_weights(G)

    nx.set_node_attributes(G, "", "actual_route_order")  # type: ignore[call-overload, arg-type]
    nx.set_node_attributes(G, False, "actual_stop")  # type: ignore[call-overload, arg-type]

    df = pd.read_csv(str(settings.testing_data_path / "donegal_stops_updated.csv"))

    for _, row in df.iterrows():
        lng = float(row["lng"])  # type: ignore[arg-type]
        lat = float(row["lat"])  # type: ignore[arg-type]
        nearest_node = int(ox.nearest_nodes(G, X=lng, Y=lat))  # type: ignore[arg-type]
        G.nodes[nearest_node]["actual_stop"] = True

        existing = str(G.nodes[nearest_node].get("actual_route_order", ""))
        order = str(row["route_order"])
        G.nodes[nearest_node]["actual_route_order"] = (
            order if existing == "" else f"{existing}, {order}"
        )

    settings.testing_graphml_path.mkdir(parents=True, exist_ok=True)
    out = settings.testing_graphml_path / "actual_routes_added.graphml"
    ox.save_graphml(G, filepath=str(out))  # type: ignore[arg-type]


def build_merged_test_graph(settings: Settings) -> None:
    """Merge generated and actual routes into a single test graph.

    Reads actual_routes_added.graphml,
    writes testing_graph.graphml and merged_routes.graphml.
    Final graph: 539 generated + 122 actual stops merged onto full Donegal road network.
    """
    import geopandas as gpd
    import osmnx as ox  # type: ignore[import-untyped]

    # Download full Donegal road network (all types) and save as base testing graph
    settings.testing_graphml_path.mkdir(parents=True, exist_ok=True)
    testing_graph_path = settings.testing_graphml_path / "testing_graph.graphml"

    G_test = ox.graph_from_place("Donegal, Ireland", network_type="all")
    ox.save_graphml(G_test, filepath=str(testing_graph_path))

    # Load route graph with actual stops added
    actual_path = settings.testing_graphml_path / "actual_routes_added.graphml"
    G_routes = ox.load_graphml(filepath=str(actual_path))

    test_nodes, test_edges = ox.graph_to_gdfs(G_test)
    route_nodes, _ = ox.graph_to_gdfs(G_routes)

    # Convert boolean string attributes to booleans
    bool_map = {"True": True, "False": False}
    for col in ["community_route", "actual_stop", "connection_route"]:
        if col in route_nodes.columns:
            route_nodes[col] = route_nodes[col].map(bool_map).fillna(False)  # type: ignore[call-overload]

    # Spatial join: transfer route attributes to test graph nodes
    merged_nodes = gpd.sjoin(  # type: ignore[call-overload]
        test_nodes, route_nodes, how="left", predicate="intersects"
    )

    # Remove duplicate _right columns, strip _left suffix from renamed columns
    merged_nodes = merged_nodes.loc[  # type: ignore[assignment]
        :, ~merged_nodes.columns.str.endswith("_right")
    ]
    merged_nodes.columns = [  # type: ignore[assignment]
        col.replace("_left", "") for col in merged_nodes.columns
    ]

    # For route nodes missing from the test graph, transfer attrs to nearest test node
    route_attrs = [
        "actual_stop",
        "community_route",
        "connection_route",
        "actual_route_order",
        "community_route_order",
        "connection_order",
        "connection",
    ]
    important_route_mask = (
        (route_nodes.get("actual_stop") == True)  # noqa: E712
        | (route_nodes.get("community_route") == True)  # noqa: E712
        | (route_nodes.get("connection_route") == True)  # noqa: E712
    )
    important_route_nodes = route_nodes[important_route_mask]

    for osmid in important_route_nodes.index:
        if osmid in merged_nodes.index:
            continue
        rn = route_nodes.loc[osmid]
        nearest = int(  # type: ignore[call-overload]
            ox.nearest_nodes(G_test, X=float(rn["x"]), Y=float(rn["y"]))
        )
        for attr in route_attrs:
            if attr in route_nodes.columns and attr in merged_nodes.columns:
                merged_nodes.loc[merged_nodes.index == nearest, attr] = rn[attr]  # type: ignore[index]

    # Fill NaN boolean columns with False
    for col in ["community_route", "actual_stop", "connection_route"]:
        if col in merged_nodes.columns:
            merged_nodes[col] = merged_nodes[col].fillna(False)  # type: ignore[call-overload]

    # Rebuild graph and save
    G_merged = ox.graph_from_gdfs(merged_nodes, test_edges)
    merged_path = settings.testing_graphml_path / "merged_routes.graphml"
    ox.save_graphml(G_merged, filepath=str(merged_path))


def run(settings: Settings | None = None) -> None:
    """Execute the full test data preparation pipeline."""
    settings = settings or Settings()
    parse_gtfs_stops(settings)
    add_actual_stops_to_graph(settings)
    build_merged_test_graph(settings)


if __name__ == "__main__":
    run()
