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

if __name__ == '__main__':
    import config
    import osmnx as ox
    import networkx as nx
    import pandas as pd

    graph_file = "../donegal_osm.graphml"
    G = ox.load_graphml(graph_file)
    population_data = pd.read_csv(f"{config.population_data_path}/donegal_townlands_all_coordinates.csv")
    population_data.head()
    nearest_edges = ox.get_nearest_edges(G, population_data["lng"], population_data["lat"], method="balltree")
    population_data["nearest_edge"] = nearest_edges.tolist()

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    # get all non-primary/secondary/tertiary
    exclude_roads = edges[(edges["highway"] != "primary") |
                          (edges["highway"] != "secondary") |
                          (edges["highway"] != "tertiary")]
    #G[2907242498][1514179595]["weight"]