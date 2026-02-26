from pydantic import BaseModel


class NodeProperties(BaseModel):
    osmid: int
    x: float
    y: float
    community: int = 0
    rank: float = 0.0
    top_n: int = 0
    route_flag: int = 0


class EdgeProperties(BaseModel):
    u: int
    v: int
    key: int = 0
    length: float
    weight: float = 0.0
    highway: str = ""
    name: str = ""


class GraphSummary(BaseModel):
    num_nodes: int
    num_edges: int
    num_communities: int
    is_connected: bool
