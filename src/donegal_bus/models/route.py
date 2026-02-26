from pydantic import BaseModel


class RouteStop(BaseModel):
    osmid: int
    x: float
    y: float
    sequence: int


class RouteInfo(BaseModel):
    community: int
    num_stops: int
    total_weight: float


class RouteDetail(BaseModel):
    community: int
    stops: list[RouteStop]
    total_weight: float


class RouteCollection(BaseModel):
    routes: list[RouteDetail]
    total_routes: int
