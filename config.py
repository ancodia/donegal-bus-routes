import rootpath
ROOT_DIR = rootpath.detect()

########## Data #############
data_path = f"{ROOT_DIR}/data"
population_data_path = f"{data_path}/population"
population_csv = f"{population_data_path}/donegal_townlands_all_coordinates.csv"

locallink_data_path = f"{data_path}/locallink"

#############################

####### Graph creation ######
graph_path = f"{ROOT_DIR}/graph"
graph_graphml_path = f"{graph_path}/graphml"
#############################

########## GNN ##############
gnn_path = f"{ROOT_DIR}/gnn"
gnn_pickle_path = f"{gnn_path}/pickle"

# paths to GNN train/test graph pickles
train_graph = f"{gnn_path}/pickle/train_graph.gpickle"
#############################
