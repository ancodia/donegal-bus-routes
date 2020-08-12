

def get_path_of_route(df):
    max_number_stops = df["stop_sequence"].max()

    # get trip_id of first occurence of the max sequence value
    trip_id = df[df["stop_sequence"] == max_number_stops]["trip_id"].values[0]

    df_filtered = df[df["trip_id"] == trip_id]
    return df_filtered[["route_id", "stop_id", "stop_sequence"]]