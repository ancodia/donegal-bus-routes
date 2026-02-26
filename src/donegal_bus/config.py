from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised configuration for Donegal Bus Routes."""

    root_dir: Path = Path(".")

    # ── Data paths ──────────────────────────────────────────────
    data_path: Path = Path("data")
    population_data_path: Path = Path("data/population")
    locallink_data_path: Path = Path("data/locallink")

    # ── Graph creation paths ────────────────────────────────────
    graph_graphml_path: Path = Path("graph/graphml")

    # ── Route planning paths ────────────────────────────────────
    rp_graphml_path: Path = Path("route_planning/graphml")
    rp_clusters_path: Path = Path("route_planning/clusters")

    # ── Testing paths ───────────────────────────────────────────
    testing_graphml_path: Path = Path("testing/graphml")
    testing_data_path: Path = Path("testing/data")

    model_config = {"env_file": ".env", "env_prefix": "DONEGAL_"}

    # ── Derived paths ───────────────────────────────────────────

    @property
    def population_csv(self) -> Path:
        return self.population_data_path / "donegal_townlands_all_coordinates.csv"

    @property
    def population_source_xlsx(self) -> Path:
        return self.population_data_path / "COP2016_Townlands.xlsx"
