"""Extract census townlands and geocode via Nominatim.

Run: python -m donegal_bus.offline.population_data
"""

from donegal_bus.config import Settings


def extract_townlands(settings: Settings) -> None:
    """Extract Donegal townlands from census source data.

    Reads COP2016_Townlands.xlsx, writes donegal_townlands.csv.
    """
    from donegal_bus.helpers.population import (
        extract_county_townlands_from_source_data,
    )

    extract_county_townlands_from_source_data(
        source_xlsx=str(settings.population_source_xlsx),
        output_csv=str(settings.population_data_path / "donegal_townlands.csv"),
        county_abbr="DL",
    )


def geocode_townlands(settings: Settings) -> None:
    """Geocode townland addresses using OSM Nominatim.

    Reads donegal_townlands.csv, writes donegal_townlands_with_coordinates.csv.

    Note: ~96 townlands fail Nominatim geocoding and require manual coordinate
    input. After manual fill-in, merge into donegal_townlands_all_coordinates.csv
    following the steps in graph/notebooks/donegal_population_data.ipynb.
    """
    import pandas as pd

    from donegal_bus.helpers.population import lookup_osm_coordinates

    csv = settings.population_data_path / "donegal_townlands.csv"
    df = pd.read_csv(str(csv))

    # First pass: geocode by townland name alone
    df = df.apply(lookup_osm_coordinates, args=("townland",), axis=1)

    # Second pass: retry failures using "townland, town" as the address
    missing = df["lat"].isna()  # type: ignore[union-attr]
    if missing.any():  # type: ignore[union-attr]
        df.loc[missing, "address"] = (
            df.loc[missing, "townland"] + ", " + df.loc[missing, "town"]
        )
        df.loc[missing] = df.loc[missing].apply(
            lookup_osm_coordinates, args=("address",), axis=1
        )
        df = df.drop(columns=["address"], errors="ignore")

    out = settings.population_data_path / "donegal_townlands_with_coordinates.csv"
    df.to_csv(str(out), index=False)


def run(settings: Settings | None = None) -> None:
    """Execute the full population data pipeline."""
    settings = settings or Settings()
    extract_townlands(settings)
    geocode_townlands(settings)


if __name__ == "__main__":
    run()
