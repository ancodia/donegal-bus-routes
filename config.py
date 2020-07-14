import rootpath
ROOT_DIR = rootpath.detect()
data_path = f"{ROOT_DIR}/data"
population_data_path = f"{data_path}/population"
graph_path = f"{ROOT_DIR}/graph"
graph_graphml_path = f"{graph_path}/graphml"
gnn_path = f"{ROOT_DIR}/gnn"
gnn_pickle_path = f"{gnn_path}/pickle"