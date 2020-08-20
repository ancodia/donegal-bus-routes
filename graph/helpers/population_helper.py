import pandas as pd
from openpyxl import load_workbook
from OSMPythonTools.nominatim import Nominatim
import config


def extract_county_townlands_from_source_data(county_abbr="DL"):
    """
    Filter source population data to extract:
        - Donegal townland addresses
        - 2016 census totals
    :param county_abbr: 2 letter Irish county abbreviation
    :return: Dataframe containing county townland records (2016)
    """
    # load excel data
    population_data = load_workbook(
        "../../data/population/COP2016_Townlands.xlsx")

    # convert to pandas dataframe, set first row as header
    df = pd.DataFrame(population_data.worksheets[0].values).T.set_index(0).T
    # filter so only Donegal townlands remain
    df = df[df["COUNTY"] == county_abbr]
    # keep only relevant data, i.e. address and 2016 census total
    df = df[["TLANDNAME", "EDNAMES_3409S", "TOTAL2016"]]
    df.columns = ["townland", "town", "population"]
    # remove any rows where population is -1 or 0 as these are not useful
    df = df[df["population"] > 0]
    df.to_csv(f"{config.population_data_path}/donegal_townlands.csv")
    return df


def extract_lat_long_from_nominatim(address):
    """
    Query OSM Nominatim API for the latitude and longitude coordinates of address
    :param address: String
    :return: lat, lng
    """
    lat, lng = None, None
    nominatim = Nominatim()
    area = nominatim.query(address)
    if area is None:
        return None, None
    try:
        """
        Try block in case any inputs are invalid
        """
        osm_json = area.toJSON()
        json_item = None
        for item in osm_json:
            if "Donegal" in item["display_name"]:
                json_item = item
                break
        lat = json_item["lat"]
        lng = json_item["lon"]
    except:
        pass
    return lat, lng


def lookup_osm_coordinates(row, column):
    """
    Lookup address with OpenStreetMap API to get central latitude and longitude values
    :param row: Dataframe row
    :param column: Address column
    :return: Dataframe row w/ lat and lng
    """
    address_value = row[column]
    address_lat, address_lng = extract_lat_long_from_nominatim(address_value)
    row["lat"] = address_lat
    row["lng"] = address_lng
    return row
