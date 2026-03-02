# Implementation Plan: Web UI Integration Readiness

## Context

The Donegal Bus Routes FastAPI backend is fully implemented with 12 working endpoints across 5 routers. All stubs are done. The API serves GeoJSON and structured JSON, but is not yet ready for consumption by a web frontend. This plan addresses the gaps: CORS, compression, large payload handling, slow endpoint caching, a health check, and improved OpenAPI documentation.

---

## Phase 1: CORS + GZip + Health Endpoint

**Why:** Without CORS middleware, any frontend on a different origin is blocked. GZip dramatically reduces transfer size for GeoJSON payloads. A health endpoint is needed for deployment tooling.

### `src/donegal_bus/config.py`

Add `cors_origins` field to `Settings`:

```python
cors_origins: list[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]
```

Overridable via `DONEGAL_CORS_ORIGINS` env var (pydantic-settings parses JSON lists).

### `src/donegal_bus/api/app.py`

Add three things to `create_app()`:

1. **CORS middleware** — `allow_origins` from settings, `allow_methods=["GET"]` (read-only API)
2. **GZip middleware** — `minimum_size=1000` (only compress responses >1KB)
3. **`GET /health`** — returns `{"status": "ok"}`, registered directly on the app

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from donegal_bus.api.dependencies import get_settings

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(...)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # ... existing include_router calls unchanged ...
```

### Verification

```bash
# Health check
curl -s http://localhost:8000/health

# CORS preflight
curl -s -X OPTIONS http://localhost:8000/graph/summary \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" -v 2>&1 | grep access-control

# GZip
curl -s -H "Accept-Encoding: gzip" http://localhost:8000/graph/nodes -o /dev/null -w '%{size_download}'
```

---

## Phase 2: Bbox Filtering for `/graph/nodes` and `/graph/edges`

**Why:** These endpoints return ~3,579 and ~7,195 features respectively. A map UI only needs features in the current viewport. Bbox filtering lets the frontend request just what's visible.

### `src/donegal_bus/api/routers/graph.py`

Add a private `_filter_bbox()` helper and optional `min_lng`, `min_lat`, `max_lng`, `max_lat` query params to both `get_nodes` and `get_edges`. All four params must be provided together; if any is missing, return the full unfiltered collection (backward compatible).

```python
from donegal_bus.models.geojson import (
    Feature,
    FeatureCollection,
    LineStringGeometry,
    PointGeometry,
)

def _filter_bbox(
    fc: FeatureCollection,
    min_lng: float,
    min_lat: float,
    max_lng: float,
    max_lat: float,
) -> FeatureCollection:
    filtered: list[Feature] = []
    for f in fc.features:
        if isinstance(f.geometry, PointGeometry):
            lng, lat = f.geometry.coordinates
            if min_lng <= lng <= max_lng and min_lat <= lat <= max_lat:
                filtered.append(f)
        elif isinstance(f.geometry, LineStringGeometry):
            for lng, lat in f.geometry.coordinates:
                if min_lng <= lng <= max_lng and min_lat <= lat <= max_lat:
                    filtered.append(f)
                    break
    return FeatureCollection(features=filtered)

@router.get("/nodes")
def get_nodes(
    G: MultiDiGraph = Depends(get_road_network_graph),
    min_lng: float | None = Query(None),
    min_lat: float | None = Query(None),
    max_lng: float | None = Query(None),
    max_lat: float | None = Query(None),
) -> FeatureCollection:
    fc = nodes_to_geojson(G)
    if all(v is not None for v in (min_lng, min_lat, max_lng, max_lat)):
        fc = _filter_bbox(fc, min_lng, min_lat, max_lng, max_lat)  # type: ignore
    return fc
```

Same pattern for `get_edges`.

### Verification

```bash
# Filtered (small bbox around Letterkenny)
curl -s "http://localhost:8000/graph/nodes?min_lng=-7.8&min_lat=54.9&max_lng=-7.7&max_lat=55.0" \
  | python -c "import json,sys; print(len(json.load(sys.stdin)['features']), 'features')"

# Unfiltered (backward compat)
curl -s http://localhost:8000/graph/nodes \
  | python -c "import json,sys; print(len(json.load(sys.stdin)['features']), 'features')"
```

---

## Phase 3: Cache `/analysis/accessibility` Results

**Why:** This endpoint runs Dijkstra sampling across hundreds of nodes and takes 10–30s. The graph never changes at runtime, so results are deterministic.

### `src/donegal_bus/api/dependencies.py`

Add a module-level cache following the existing `_road_network` / `_routes_graph` pattern:

```python
from donegal_bus.models.analysis import AccessibilitySummary

_accessibility_cache: AccessibilitySummary | None = None

def get_cached_accessibility() -> AccessibilitySummary | None:
    return _accessibility_cache

def set_cached_accessibility(summary: AccessibilitySummary) -> None:
    global _accessibility_cache
    _accessibility_cache = summary
```

### `src/donegal_bus/api/routers/analysis.py`

Check cache before computing, store result after:

```python
from donegal_bus.api.dependencies import (
    get_cached_accessibility,
    set_cached_accessibility,
    ...
)

@router.get("/accessibility")
def accessibility_summary(
    G: MultiDiGraph = Depends(get_routes_graph),
) -> AccessibilitySummary:
    cached = get_cached_accessibility()
    if cached is not None:
        return cached
    # ... existing computation unchanged ...
    result = summarize_results(results)
    set_cached_accessibility(result)
    return result
```

### Verification

```bash
# First call — slow (~10-30s)
time curl -s http://localhost:8000/analysis/accessibility | python -m json.tool
# Second call — instant
time curl -s http://localhost:8000/analysis/accessibility | python -m json.tool
```

---

## Phase 4: Enrich `/stops/nearest` Response

**Why:** The frontend needs the distance to display "nearest stop is X away." Currently the response has no distance info.

### `src/donegal_bus/api/routers/stops.py`

Add `distance_deg` to the nearest stop properties (approximate distance in degrees — cheap to compute, already calculated for the min-finding):

```python
@router.get("/nearest")
def nearest_stop(lat: float, lng: float, ...) -> FeatureCollection:
    ...
    nearest_node, nearest_data = min(
        stops,
        key=lambda nd: math.hypot(float(nd[1]["x"]) - lng, float(nd[1]["y"]) - lat),
    )
    dist = math.hypot(float(nearest_data["x"]) - lng, float(nearest_data["y"]) - lat)
    ...
    props = {
        "osmid": int(nearest_data.get("osmid", nearest_node)),
        "community": int(nearest_data.get("community", 0)),
        "distance_deg": round(dist, 6),
    }
```

Keep returning `FeatureCollection` — it's valid GeoJSON and changing the shape now would be premature since no frontend exists yet.

---

## Phase 5: Document Error Responses in OpenAPI

**Why:** Without error models, the Swagger UI shows no schema for 404/500 responses. Frontend devs need to know the error shape.

### `src/donegal_bus/models/error.py` (new file)

```python
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    detail: str
```

### Add `responses` to endpoints that raise `HTTPException`

Three endpoints to annotate:

- `src/donegal_bus/api/routers/communities.py` — `get_community`
- `src/donegal_bus/api/routers/routes.py` — `routes_by_community`, `connection_route`

```python
from donegal_bus.models.error import ErrorResponse

@router.get(
    "/{label}",
    responses={404: {"model": ErrorResponse}},
)
def get_community(label: int, ...) -> CommunityDetail:
    ...
```

Same pattern for the two routes endpoints.

---

## Summary

| Phase | Files | What |
|-------|-------|------|
| 1 | `config.py`, `app.py` | CORS + GZip + `/health` |
| 2 | `routers/graph.py` | Bbox filtering for nodes/edges |
| 3 | `dependencies.py`, `routers/analysis.py` | Accessibility result caching |
| 4 | `routers/stops.py` | Add distance to nearest stop |
| 5 | `models/error.py` (new), `routers/communities.py`, `routers/routes.py` | Error response docs |

### Verification (end-to-end)

```bash
ruff check src/
pyright src/
python main.py &
sleep 2
curl -s http://localhost:8000/health
curl -s http://localhost:8000/graph/summary
curl -s "http://localhost:8000/graph/nodes?min_lng=-7.8&min_lat=54.9&max_lng=-7.7&max_lat=55.0"
curl -s http://localhost:8000/analysis/accessibility  # first: slow, second: instant
curl -s "http://localhost:8000/stops/nearest?lat=54.95&lng=-7.73"  # check distance_deg
kill %1
```
