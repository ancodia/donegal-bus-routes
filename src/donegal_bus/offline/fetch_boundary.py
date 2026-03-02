"""Fetch County Donegal OSM boundary and write to web/public/.

Run: python -m donegal_bus.offline.fetch_boundary
"""

from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT = _THIS_DIR.parents[2] / "web" / "public" / "donegal-boundary.geojson"


def fetch_donegal_boundary(output: Path = _OUTPUT) -> None:
    """Download County Donegal admin boundary from OSM, simplify, save as GeoJSON."""
    import osmnx as ox  # type: ignore[import-untyped]

    print("Fetching boundary from Nominatim…")
    gdf = ox.geocode_to_gdf("County Donegal, Ireland")

    print("Simplifying geometry (tolerance=0.001°)…")
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(output), driver="GeoJSON")
    print(f"Written: {output}")


def run() -> None:
    fetch_donegal_boundary()


if __name__ == "__main__":
    run()
