from pydantic import BaseModel


class Community(BaseModel):
    label: int
    num_nodes: int
    num_edges: int


class CommunityCollection(BaseModel):
    communities: list[Community]
    total: int


class CommunityDetail(BaseModel):
    label: int
    num_nodes: int
    num_edges: int
    top_ranked_nodes: list[int]
    route_start: int | None = None
    route_end: int | None = None
