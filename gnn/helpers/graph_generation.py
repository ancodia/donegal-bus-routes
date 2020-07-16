import networkx as nx
import matplotlib.pyplot as plt

sizes = [75, 75, 300]
probs = [[0.25, 0.05, 0.02],
         [0.05, 0.35, 0.07],
         [0.02, 0.07, 0.40]]
g = nx.stochastic_block_model(sizes, probs, seed=0)
len(g)
nx.draw_networkx(g)
plt.draw()