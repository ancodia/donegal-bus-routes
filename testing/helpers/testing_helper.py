import networkx as nx
import route_planning.helpers.route_planning_helper as route_helper


def get_path_of_route(df):
    max_number_stops = df["stop_sequence"].max()

    # get trip_id of first occurence of the max sequence value
    trip_id = df[df["stop_sequence"] == max_number_stops]["trip_id"].values[0]

    df_filtered = df[df["trip_id"] == trip_id]
    return df_filtered[["route_id", "stop_id", "stop_sequence"]]


def find_shortest_path_to_destinations(G,
                                       source,
                                       destinations,
                                       weight="length",
                                       print_all=True):
    shortest_path = None
    shortest_path_weight = None

    for dest in destinations:
        for path in nx.all_shortest_paths(G, source, dest, weight=weight):
            path_weight = route_helper.path_weight(G, path, weight=weight)
            if print_all: print(f"{path_weight} - {path}")
            if shortest_path_weight is None or \
                    path_weight < shortest_path_weight:
                shortest_path = path
                shortest_path_weight = path_weight
    return shortest_path, shortest_path_weight


def sample_size(population_size, margin_error=.05, confidence_level=.99, sigma=1/2):
    """
    From: https://github.com/shawnohare/samplesize/blob/master/samplesize.py
    Calculate the minimal sample size to use to achieve a certain
    margin of error and confidence level for a sample estimate
    of the population mean.
    Inputs
    -------
    population_size: integer
        Total size of the population that the sample is to be drawn from.
    margin_error: number
        Maximum expected difference between the true population parameter,
        such as the mean, and the sample estimate.
    confidence_level: number in the interval (0, 1)
        If we were to draw a large number of equal-size samples
        from the population, the true population parameter
        should lie within this percentage
        of the intervals (sample_parameter - e, sample_parameter + e)
        where e is the margin_error.
    sigma: number
        The standard deviation of the population.  For the case
        of estimating a parameter in the interval [0, 1], sigma=1/2
        should be sufficient.
    """
    alpha = 1 - (confidence_level)
    # dictionary of confidence levels and corresponding z-scores
    # computed via norm.ppf(1 - (alpha/2)), where norm is
    # a normal distribution object in scipy.stats.
    # Here, ppf is the percentile point function.
    zdict = {
        .90: 1.645,
        .91: 1.695,
        .99: 2.576,
        .97: 2.17,
        .94: 1.881,
        .93: 1.812,
        .95: 1.96,
        .98: 2.326,
        .96: 2.054,
        .92: 1.751
    }
    if confidence_level in zdict:
        z = zdict[confidence_level]
    else:
        from scipy.stats import norm
        z = norm.ppf(1 - (alpha/2))
    N = population_size
    M = margin_error
    numerator = z**2 * sigma**2 * (N / (N-1))
    denom = M**2 + ((z**2 * sigma**2)/(N-1))
    return numerator/denom