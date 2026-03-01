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
    from openpyxl import load_workbook  # type: ignore[import-untyped]

    population_data = load_workbook(source_xlsx)
    raw: pd.DataFrame = (  # type: ignore[assignment]
        pd.DataFrame(population_data.worksheets[0].values).T.set_index(0).T
    )
    cols = ["TLANDNAME", "EDNAMES_3409S", "TOTAL2016"]
    filtered = raw[raw["COUNTY"] == county_abbr][cols].copy()
    filtered.columns = pd.Index(["townland", "town", "population"])  # type: ignore[assignment]
    result: pd.DataFrame = filtered[filtered["population"] > 0]  # type: ignore[assignment]
    result.to_csv(output_csv, index=False)
    return result


def extract_lat_long_from_nominatim(address: str) -> tuple[float | None, float | None]:
    """Query OSM Nominatim API for the lat/lng coordinates of an address.

    Args:
        address: Free-text address string to geocode.

    Returns:
        Tuple of (latitude, longitude), or (None, None) if not found.
    """
    from OSMPythonTools.nominatim import Nominatim  # type: ignore[import-untyped]

    lat: float | None = None
    lng: float | None = None
    nominatim = Nominatim()
    area = nominatim.query(address)
    if area is None:
        return None, None
    try:
        osm_json = area.toJSON()
        json_item = None
        for item in osm_json:
            if "Donegal" in item["display_name"]:
                json_item = item
                break
        if json_item is not None:
            lat = float(json_item["lat"])
            lng = float(json_item["lon"])
    except Exception:
        pass
    return lat, lng


def lookup_osm_coordinates(row: pd.Series, column: str) -> pd.Series:  # type: ignore[type-arg]
    """Look up an address via Nominatim and add lat/lng columns to the row.

    Args:
        row: DataFrame row containing the address.
        column: Name of the column holding the address string.

    Returns:
        Row with lat and lng columns populated.
    """
    address_value: str = str(row[column])  # type: ignore[index]
    address_lat, address_lng = extract_lat_long_from_nominatim(address_value)
    row["lat"] = address_lat
    row["lng"] = address_lng
    return row
