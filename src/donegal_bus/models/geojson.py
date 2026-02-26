from typing import Literal

from pydantic import BaseModel


class PointGeometry(BaseModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float]


class LineStringGeometry(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]]


class Feature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PointGeometry | LineStringGeometry
    properties: dict[str, str | int | float | bool | None]


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature]
