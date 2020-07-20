import random

import networkx as nx
import matplotlib.pyplot as plt


def add_edge_weights_to_graph_partitions(graph_model, total_weight):
    # spread total_weight among edges in each partition
    # graph_model must have weight attribute on all edges
    partitions = graph_model.graph["partition"]
    i = 1
    for p in partitions:
        print(f"Adding weights to partition {i}")
        spread = total_weight
        while spread > 0:
            n = random.randint(15, 200)
            u = random.choice(tuple(p))
            v = random.choice(tuple(p))
            if graph_model.has_edge(u, v) and (spread - n > 0):
                weight_before = graph_model[u][v]["weight"]
                graph_model[u][v]["weight"] = weight_before + n
                spread = spread - n
            elif spread - n < 0:
                break
        print(f"Partition {i} weights assigned"
              f"\n===========================")
        i += 1