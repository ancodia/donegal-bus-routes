"""
This example implements the experiments for node clustering on citation networks
from the paper:

Mincut pooling in Graph Neural Networks (https://arxiv.org/abs/1907.00481)
Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi

https://github.com/danielegrattarola/spektral/blob/master/examples/other/node_clustering_mincut.py
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm import tqdm

from spektral.datasets import citation
from spektral.layers.convolutional import GraphConvSkip
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency

import config
import networkx as nx
import numpy as np
import osmnx as ox
from sklearn.model_selection import train_test_split


@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        _, S_pool = model(inputs, training=True)
        loss = sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return model.losses[0], model.losses[1], S_pool


np.random.seed(1)
epochs = 5000  # Training iterations
lr = 5e-4      # Learning rate

################################################################################
# LOAD DATASET
################################################################################
# A, X, y, _, _, _ = citation.load_data('cora')
# A_norm = normalized_adjacency(A)
# X = X.todense()
# F = X.shape[-1]
# y = np.argmax(y, axis=-1)
# n_clusters = y.max() + 1

graph_file = f"{config.graph_graphml_path}/donegal_osm_weights_applied.graphml"
G = ox.load_graphml(graph_file)
#data = nx.read_gpickle(config.train_graph)
#
# dataset = 'cora'
# A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# edge weights loaded as strings, need to convert to numeric
weight_attributes = nx.get_edge_attributes(G, "weight")
weight_attributes = dict([k, {"weight": float(v)}] for k, v in weight_attributes.items())
nx.set_edge_attributes(G, weight_attributes)
A = nx.to_numpy_array(G, weight="weight")
A_norm = normalized_adjacency(A)
N = A.shape[0]

# node features of x and y coordinates
x_coords = list(nx.get_node_attributes(G, "x").values())
y_coords = list(nx.get_node_attributes(G, "y").values())

# X = np.zeros((N, 2))
# X[:, 0] = x_coords
# X[:, 1] = y_coords
X = np.ones((N, 1))

# node_labels = nx.get_node_attributes(train, "block")
# # to array
# labels_array = np.fromiter(node_labels.values(), dtype=int)
# # create matrix representation of node community labels
# n_values = np.max(labels_array) + 1
# node_labels = np.eye(n_values)[labels_array]
# y = node_labels


F = X.shape[-1]
#y = np.argmax(y, axis=-1)

# same number as total locallink donegal bus routes
n_clusters = 23 #y.max() + 1

# def sample_mask(idx, l):
#     # from https://github.com/danielegrattarola/spektral/blob/master/spektral/datasets/citation.py
#     # mask labels depending on which stage is being executed
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return np.array(mask, dtype=np.bool)
#
# indices = np.arange(node_labels.shape[0])
# n_classes = node_labels.shape[1]
# # 80:20 train:test split
# idx_train, idx_test, y_train, y_test = train_test_split(
#     indices, node_labels, train_size=int(indices.shape[0] * 0.8), stratify=node_labels)
# # take 10% as validation set
# idx_val, idx_test, y_val, y_test = train_test_split(
#     idx_test, y_test, train_size=int(idx_train.shape[0] * 0.1), stratify=y_test)
#
# train_mask = sample_mask(idx_train, node_labels.shape[0])
# val_mask = sample_mask(idx_val, node_labels.shape[0])
# test_mask = sample_mask(idx_test, node_labels.shape[0])

################################################################################
# MODEL
################################################################################
X_in = Input(shape=(F,), name='X_in')
# A_in = Input(shape=(None, ), name='A_in', sparse=True)
A_in = Input(shape=(N, ), name='A_in', sparse=True)

X_1 = GraphConvSkip(N, activation='elu')([X_in, A_in])
X_1, A_1, S = MinCutPool(n_clusters, return_mask=True)([X_1, A_in])

model = Model([X_in, A_in], [X_1, S])


################################################################################
# TRAINING
################################################################################
# Setup
inputs = [X, sp_matrix_to_sp_tensor(A_norm)]
opt = tf.keras.optimizers.Adam(learning_rate=lr)
# model.compile(optimizer=opt,
#               loss='categorical_crossentropy',
#               weighted_metrics=['acc'])
# model.summary()

# # Train model
# fltr = GraphConvSkip.preprocess(A_norm).astype('f4')
# validation_data = ([X, fltr], y, val_mask)
# model.fit([X, fltr],
#           y,
#           sample_weight=train_mask,
#           epochs=epochs,
#           batch_size=N,
#           validation_data=validation_data,
#           shuffle=False,  # Shuffling data means shuffling the whole graph
#           callbacks=[
#               EarlyStopping(patience=200,  restore_best_weights=True)
#           ])

# Fit model
loss_history = []
nmi_history = []
for epoch in tqdm(range(epochs)):
    outs = train_step(inputs)
    outs = [o.numpy() for o in outs]
    loss_history.append((outs[0], outs[1], (outs[0] + outs[1])))
    s = np.argmax(outs[2], axis=-1)

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, (outs[0] + outs[1])))
    #nmi_history.append(v_measure_score(y, s))
loss_history = np.array(loss_history)

# # Evaluate model
# print('Evaluating model.')
# eval_results = model.evaluate([X, fltr],
#                               y,
#                               sample_weight=test_mask,
#                               batch_size=N)
# print('Done.\n'
#       'Test loss: {}\n'
#       'Test accuracy: {}'.format(*eval_results))

################################################################################
# RESULTS
################################################################################
# _, S_ = model(inputs, training=False)
# s = np.argmax(S_, axis=-1)
# hom = homogeneity_score(y, s)
# com = completeness_score(y, s)
# nmi = v_measure_score(y, s)
# print('Homogeneity: {:.3f}; Completeness: {:.3f}; NMI: {:.3f}'.format(hom, com, nmi))

# Plots
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(loss_history[:, 0], label='MinCUT loss')
plt.plot(loss_history[:, 1], label='Ortho. loss')
plt.plot(loss_history[:, 2], label='Total loss')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Iteration')

# plt.subplot(122)
# plt.plot(nmi_history, label='NMI')
# plt.legend()
# plt.ylabel('NMI')
# plt.xlabel('Iteration')

plt.show()

outs
print("a")