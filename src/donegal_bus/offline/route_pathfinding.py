"""Highest-weight simple paths for community and connecting routes.

Run: python -m donegal_bus.offline.route_pathfinding
"""

from donegal_bus.config import Settings


def compute_community_routes(settings: Settings, cutoff: int = 90) -> None:
    """Find highest-weight simple paths within each community.

    Reads route_start_end_flagged.graphml,
    writes community_routes_cutoff{cutoff}.graphml.
    """
    raise NotImplementedError


def compute_connecting_routes(settings: Settings) -> None:
    """Find routes that connect adjacent communities.

    Reads community route graphs, writes all_routes.graphml.
    """
    raise NotImplementedError


def run(settings: Settings | None = None) -> None:
    """Execute the full route pathfinding pipeline."""
    settings = settings or Settings()
    compute_community_routes(settings)
    compute_connecting_routes(settings)


if __name__ == "__main__":
    run()
