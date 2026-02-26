"""Map townlands to nearest edges and apply population weights.

Run: python -m donegal_bus.offline.apply_edge_weights
"""

from donegal_bus.config import Settings


def load_population_data(settings: Settings) -> None:
    """Load geocoded townland data from CSV.

    Reads donegal_townlands_all_coordinates.csv.
    """
    raise NotImplementedError


def assign_weights_to_edges(settings: Settings) -> None:
    """Map each townland to its nearest graph edge and set population weight.

    Reads donegal_osm_simplified.graphml, writes donegal_osm_weights_applied.graphml.
    """
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full edge weight pipeline."""
    settings = settings or Settings()
    load_population_data(settings)
    assign_weights_to_edges(settings)


if __name__ == "__main__":
    run()
