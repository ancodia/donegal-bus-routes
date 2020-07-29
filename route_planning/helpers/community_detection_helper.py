from itertools import count
import networkx as nx
import osmnx as ox


def plot_osmnx_graph_with_colour_coded_communities(G):
    node_colours = ox.plot.get_node_colors_by_attr(G, attr="community", cmap="Set3")
    fig, ax = ox.plot_graph(G, node_color=node_colours)


def assign_community_labels(G, labels):
    nx.set_node_attributes(G, 0, "community")
    i = 0
    for node in G.nodes:
        G.nodes[node]["community"] = labels[i]
        i += 1