from itertools import count
import networkx as nx
import osmnx as ox


def assign_community_labels(G, labels):
    nx.set_node_attributes(G, 0, "community")
    i = 0
    for node in G.nodes:
        G.nodes[node]["community"] = labels[i]
        i += 1


def convert_edge_weights_to_floats(G):
    weight_attributes = nx.get_edge_attributes(G, "weight")

    weight_attributes = dict([k, {"weight": float(v)}]
                             for k, v in weight_attributes.items())
    nx.set_edge_attributes(G, weight_attributes)
    nx.get_edge_attributes(G, "weight")


def get_n_highest_ranked_nodes_in_community(community_df, n=10):
    # get node ids of n highest ranked nodes
    top_n = list(community_df.nlargest(n, "rank")["osmid"].index)

    # add new column for recording the top ranked nodes
    community_df["top_n"] = 0
    # assign rank number to each of those nodes
    i = 1
    for node_id in top_n:
        community_df.loc[community_df["osmid"] == node_id, "top_n"] = i
        i += 1

    ranks = [*range(1, n+1)]
    # confirm top_n has been updated
    check = community_df[community_df["top_n"].isin(ranks)]
    assert len(check) == n

    return community_df


def greatest_distance_between_top_ranked_nodes(G, community_labels):
    """
    From the top n nodes in each community,
    find the pair of points that have the greatest distance between them
    :param G:
    :return:
    """
    # get lists of the top ranked nodes in each community
    top_ranked_per_community = get_top_n_ranked_nodes_per_community(G, community_labels)
    assert len(top_ranked_per_community) == len(community_labels)

    # get coordinates of nodes
    top_ranked_node_coordinates = get_node_coordinates(G, top_ranked_per_community)
    assert len(top_ranked_node_coordinates) == len(community_labels)

    max_distance_per_community = find_furthest_apart_nodes(top_ranked_node_coordinates)
    assert len(max_distance_per_community) == len(community_labels)

    return max_distance_per_community


def get_top_n_ranked_nodes_per_community(G, community_labels):
    # get the ranks of the graph's nodes
    ranks = nx.get_node_attributes(G, "top_n")
    # get unique ranks
    ranks = list(set(val for val in ranks.values()))
    # remove 0, the default "top_n" value
    ranks.remove(0)

    top_ranked_per_community = []

    for label in community_labels:
        top_n_nodes = [x for x, y in G.nodes(data=True)
                       if (y["community"] == label) & (int(y["top_n"]) in ranks)]
        top_ranked_per_community.append({label: top_n_nodes})

    return top_ranked_per_community


def get_node_coordinates(G, communities_nodes):
    """
    Get the latitude/longitude coordinates for nodes in a list of community dictionaries
    :param G:
    :param communities_nodes:
    :return:
    """
    # get coordinates of nodes
    node_coordinates = []

    for community in communities_nodes:
        coordinates = []
        label = list(iter(community.keys()))[0]
        nodes = list(iter(community.values()))[0]
        for node in nodes:
            coords_dict = {"osmid": node,
                           "y": G.nodes[node]["y"],
                           "x": G.nodes[node]["x"]}
            coordinates.append(coords_dict)
        node_coordinates.append({label: coordinates})

    return node_coordinates


def find_furthest_apart_nodes(node_coordinates):
    max_distances = []
    # check distance between points
    for coordinates_dict in node_coordinates:
        results = []
        label = list(iter(coordinates_dict.keys()))[0]
        coordinates = list(iter(coordinates_dict.values()))[0]
        range_end = len(coordinates)
        for i in range(0, range_end):
            start_point = coordinates[i]
            for j in range(i + 1, range_end):
                if j == range_end:
                    break
                end_point = coordinates[j]
                distance = {"u": start_point["osmid"],
                            "v": end_point["osmid"],
                            "dist": ox.distance.euclidean_dist_vec(y1=start_point["y"],
                                                                   x1=start_point["x"],
                                                                   y2=end_point["y"],
                                                                   x2=end_point["x"])}
                results.append(distance)
        # get maximum distance from results
        max_distance = max(results, key=lambda x: x["dist"])
        max_distances.append({label: max_distance})
    return max_distances


def assign_route_start_end_points(G, route_nodes, n_communities):
    """
    Add a route_flag to the graph's nodes.
    Then assign start (u) or end (v) flags to the nodes
    found in route_nodes list of community dictionaries
    :param G:
    :param route_nodes:
    :return:
    """
    # add flags to start/end nodes for community routes
    nx.set_node_attributes(G, None, "route_flag")

    for community in route_nodes:
        dict = list(iter(community.values()))[0]
        G.nodes[dict["u"]]["route_flag"] = "u"
        G.nodes[dict["v"]]["route_flag"] = "v"

    # verify the expected number of start and end points were added
    u_nodes = [y for x, y in G.nodes(data=True) if y["route_flag"] == "u"]
    v_nodes = [y for x, y in G.nodes(data=True) if y["route_flag"] == "v"]
    assert len(u_nodes) == n_communities & len(v_nodes) == n_communities

    return u_nodes, v_nodes
