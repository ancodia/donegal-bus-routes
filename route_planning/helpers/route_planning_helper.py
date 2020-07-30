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