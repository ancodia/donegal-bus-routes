from pydantic import BaseModel


class Townland(BaseModel):
    townland: str
    town: str
    population: int
    lat: float | None = None
    lng: float | None = None


class PopulationSummary(BaseModel):
    total_townlands: int
    total_population: int
    townlands_with_coords: int
