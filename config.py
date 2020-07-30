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

########## Route Planning ##############
rp_path = f"{ROOT_DIR}/route_planning"
rp_graphml_path = f"{rp_path}/graphml"
#############################
