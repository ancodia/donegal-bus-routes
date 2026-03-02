"""Map townlands to nearest edges and apply population weights.

Run: python -m donegal_bus.offline.apply_edge_weights
"""

import pandas as pd

from donegal_bus.config import Settings


def load_population_data(settings: Settings) -> pd.DataFrame:
    """Load geocoded townland data from CSV.

    Reads donegal_townlands_all_coordinates.csv.
    Returns the DataFrame for use by assign_weights_to_edges.
    """
    return pd.read_csv(str(settings.population_csv))


def assign_weights_to_edges(settings: Settings, df: pd.DataFrame | None = None) -> None:
    """Map each townland to its nearest graph edge and set population weight.

    Reads donegal_osm_simplified.graphml, writes donegal_osm_weights_applied.graphml.
    Uses ox.nearest_edges (osmnx v2.1 API: X=longitude, Y=latitude).
    """
    import networkx as nx
    import osmnx as ox  # type: ignore[import-untyped]

    if df is None:
        df = load_population_data(settings)

    simplified = settings.graph_graphml_path / "donegal_osm_simplified.graphml"
    G = ox.load_graphml(filepath=str(simplified))

    # Initialise all edge weights to 1.0
    nx.set_edge_attributes(G, 1.0, "weight")  # type: ignore[call-overload]

    # Drop rows missing coordinates
    df_valid = df.dropna(subset=["lat", "lng"])

    lngs = df_valid["lng"].to_numpy()
    lats = df_valid["lat"].to_numpy()

    # nearest_edges returns an ndarray of shape (N, 3) with columns [u, v, key]
    nearest = ox.nearest_edges(G, X=lngs, Y=lats)  # type: ignore[var-annotated]
    populations = df_valid["population"].to_numpy()

    for edge_row, pop in zip(nearest, populations):
        u, v, key = int(edge_row[0]), int(edge_row[1]), int(edge_row[2])
        current = float(G[u][v][key].get("weight", 1.0))  # type: ignore[index]
        G[u][v][key]["weight"] = current + float(pop)  # type: ignore[index]

    out = settings.graph_graphml_path / "donegal_osm_weights_applied.graphml"
    ox.save_graphml(G, filepath=str(out))


def run(settings: Settings | None = None) -> None:
    """Execute the full edge weight pipeline."""
    settings = settings or Settings()
    df = load_population_data(settings)
    assign_weights_to_edges(settings, df)


if __name__ == "__main__":
    run()
