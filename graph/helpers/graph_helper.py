import osmnx as ox


def get_nearest_edges_to_townland(row, graph):
    """
    !!!!!!!
    Lookup address with OpenStreetMap API to get central latitude and longitude values
    :return:
    """
    # use balltree method as recommended in osmnx doc for large graphs when using lat/lng coordinates
    nearest_edges = ox.get_nearest_edges(graph, row["lng"], row["lat"], method="balltree")
    row["edges"] = nearest_edges
    return row

