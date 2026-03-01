"""Extract census townlands and geocode via Nominatim.

Run: python -m donegal_bus.offline.population_data
"""

from donegal_bus.config import Settings


def extract_townlands(settings: Settings) -> None:
    """Extract Donegal townlands from census source data.

    Reads COP2016_Townlands.xlsx, writes donegal_townlands.csv.
    """
    # TODO: Port from graph/notebooks/donegal_population_data.ipynb (cells 1-5)
    #
    # from donegal_bus.helpers.population import (
    #     extract_county_townlands_from_source_data,
    # )
    # extract_county_townlands_from_source_data(
    #     source_xlsx=str(settings.population_source_xlsx),
    #     output_csv=str(settings.population_data_path / "donegal_townlands.csv"),
    #     county_abbr="DL",
    # )
    raise NotImplementedError


def geocode_townlands(settings: Settings) -> None:
    """Geocode townland addresses using OSM Nominatim.

    Reads donegal_townlands.csv, writes donegal_townlands_with_coordinates.csv.
    Final output (donegal_townlands_all_coordinates.csv) merges ~96 manual lookups.
    """
    # TODO: Port from graph/notebooks/donegal_population_data.ipynb (cells 6-12)
    #
    # import pandas as pd
    # from donegal_bus.helpers.population import lookup_osm_coordinates
    #
    # csv = settings.population_data_path / "donegal_townlands.csv"
    # df = pd.read_csv(str(csv))
    # df = df.apply(lookup_osm_coordinates, column="townland", axis=1)
    # out = settings.population_data_path / "donegal_townlands_with_coordinates.csv"
    # df.to_csv(str(out), index=False)
    #
    # NOTE: ~96 townlands fail Nominatim and need manual coordinates.
    # See notebook for manual fill-in and merge into
    # donegal_townlands_all_coordinates.csv
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full population data pipeline."""
    settings = settings or Settings()
    extract_townlands(settings)
    geocode_townlands(settings)


if __name__ == "__main__":
    run()
