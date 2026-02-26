"""Port of graph/helpers/population_helper.py — population data extraction."""

import pandas as pd


def extract_county_townlands_from_source_data(
    source_xlsx: str, output_csv: str, county_abbr: str = "DL"
) -> pd.DataFrame:
    """Filter source census data to extract county townland addresses and populations.

    Args:
        source_xlsx: Path to COP2016_Townlands.xlsx.
        output_csv: Path to write the filtered CSV.
        county_abbr: Two-letter Irish county abbreviation.

    Returns:
        DataFrame with columns: townland, town, population.
    """
    raise NotImplementedError


def extract_lat_long_from_nominatim(address: str) -> tuple[float | None, float | None]:
    """Query OSM Nominatim API for the lat/lng coordinates of an address.

    Args:
        address: Free-text address string to geocode.

    Returns:
        Tuple of (latitude, longitude), or (None, None) if not found.
    """
    raise NotImplementedError


def lookup_osm_coordinates(row: pd.Series, column: str) -> pd.Series:
    """Look up an address via Nominatim and add lat/lng columns to the row.

    Args:
        row: DataFrame row containing the address.
        column: Name of the column holding the address string.

    Returns:
        Row with lat and lng columns populated.
    """
    raise NotImplementedError
