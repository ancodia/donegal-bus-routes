"""Map townlands to nearest edges and apply population weights.

Run: python -m donegal_bus.offline.apply_edge_weights
"""

from donegal_bus.config import Settings


def load_population_data(settings: Settings) -> None:
    """Load geocoded townland data from CSV.

    Reads donegal_townlands_all_coordinates.csv.
    """
    # TODO: Port from graph/notebooks/apply_edge_weights.ipynb (cells 1-3)
    #
    # import pandas as pd
    # df = pd.read_csv(str(settings.population_csv))
    # return df  (or store in module-level variable for use by assign_weights_to_edges)
    raise NotImplementedError


def assign_weights_to_edges(settings: Settings) -> None:
    """Map each townland to its nearest graph edge and set population weight.

    Reads donegal_osm_simplified.graphml, writes donegal_osm_weights_applied.graphml.
    """
    # TODO: Port from graph/notebooks/apply_edge_weights.ipynb (cells 4-8)
    #
    # osmnx v0.15 → v2.1 migration note:
    #   OLD: ox.get_nearest_edges(G, lngs, lats, method="balltree")
    #   NEW: ox.nearest_edges(G, X=lngs, Y=lats)  (X=longitude, Y=latitude)
    #
    # import osmnx as ox
    # import networkx as nx
    # from donegal_bus.graph_io import load_graphml, save_graphml, fix_edge_weights
    #
    # G = load_graphml(settings.graph_graphml_path / "donegal_osm_simplified.graphml")
    # fix_edge_weights(G)
    # nx.set_edge_attributes(G, 1.0, "weight")  # initialise all edges to weight 1.0
    #
    # lngs = df["lng"].values
    # lats = df["lat"].values
    # nearest = ox.nearest_edges(G, X=lngs, Y=lats)  # returns array of (u, v, key)
    # for (u, v, key), pop in zip(nearest, df["population"]):
    #     G[u][v][key]["weight"] += pop
    #
    # out = settings.graph_graphml_path / "donegal_osm_weights_applied.graphml"
    # save_graphml(G, out)
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full edge weight pipeline."""
    settings = settings or Settings()
    load_population_data(settings)
    assign_weights_to_edges(settings)


if __name__ == "__main__":
    run()
