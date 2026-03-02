"""Fetch County Donegal OSM boundary and write to web/public/.

Run: python -m donegal_bus.offline.fetch_boundary
"""

import json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT = _THIS_DIR.parents[2] / "web" / "public" / "donegal-boundary.geojson"
_MASK_OUTPUT = _THIS_DIR.parents[2] / "web" / "public" / "donegal-mask.geojson"


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


def generate_donegal_mask(
    boundary_path: Path = _OUTPUT,
    mask_output: Path = _MASK_OUTPUT,
) -> None:
    """Generate world-minus-Donegal mask polygon for map shading outside the county."""
    from shapely.geometry import box, mapping, shape
    from shapely.ops import unary_union

    with open(boundary_path) as f:
        fc = json.load(f)

    donegal = unary_union([shape(feat["geometry"]) for feat in fc["features"]])
    world = box(-180, -90, 180, 90)
    mask = world.difference(donegal)

    feature_collection = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": mapping(mask), "properties": {}}],
    }

    mask_output.parent.mkdir(parents=True, exist_ok=True)
    with open(mask_output, "w") as f:
        json.dump(feature_collection, f)
    print(f"Written: {mask_output}")


def run() -> None:
    fetch_donegal_boundary()
    generate_donegal_mask()


if __name__ == "__main__":
    run()
