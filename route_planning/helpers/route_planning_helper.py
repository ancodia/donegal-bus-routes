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