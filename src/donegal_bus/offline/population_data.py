"""Extract census townlands and geocode via Nominatim.

Run: python -m donegal_bus.offline.population_data
"""

from donegal_bus.config import Settings


def extract_townlands(settings: Settings) -> None:
    """Extract Donegal townlands from census source data.

    Reads COP2016_Townlands.xlsx, writes donegal_townlands.csv.
    """
    raise NotImplementedError


def geocode_townlands(settings: Settings) -> None:
    """Geocode townland addresses using OSM Nominatim.

    Reads donegal_townlands.csv, writes donegal_townlands_with_coordinates.csv.
    """
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full population data pipeline."""
    settings = settings or Settings()
    extract_townlands(settings)
    geocode_townlands(settings)


if __name__ == "__main__":
    run()
