"""PageRank, top-N selection, and furthest-apart endpoint detection.

Run: python -m donegal_bus.offline.node_ranking
"""

from donegal_bus.config import Settings


def compute_pagerank(settings: Settings) -> None:
    """Run PageRank on the community graph and store rank attributes.

    Reads communities.graphml, writes nodes_ranked.graphml.
    """
    raise NotImplementedError


def select_top_n_per_community(settings: Settings, n: int = 10) -> None:
    """Flag the top-N ranked nodes in each community.

    Updates nodes_ranked.graphml in place.
    """
    raise NotImplementedError


def flag_route_endpoints(settings: Settings) -> None:
    """Find the furthest-apart top-ranked pair per community and flag as start/end.

    Reads nodes_ranked.graphml, writes route_start_end_flagged.graphml.
    """
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full node ranking pipeline."""
    settings = settings or Settings()
    compute_pagerank(settings)
    select_top_n_per_community(settings)
    flag_route_endpoints(settings)


if __name__ == "__main__":
    run()
