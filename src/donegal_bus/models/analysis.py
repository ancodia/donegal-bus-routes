from pydantic import BaseModel


class AccessibilityTestResult(BaseModel):
    source_node: int
    destinations_reached: int
    destinations_total: int
    avg_path_length: float


class AccessibilitySummary(BaseModel):
    total_tests: int
    avg_destinations_reached: float
    avg_path_length: float
    results: list[AccessibilityTestResult]


class RouteCost(BaseModel):
    route_community: int
    distance_km: float
    estimated_fuel_litres: float
    estimated_cost_eur: float


class CostComparison(BaseModel):
    generated_routes: list[RouteCost]
    actual_routes: list[RouteCost]
    total_generated_cost: float
    total_actual_cost: float
