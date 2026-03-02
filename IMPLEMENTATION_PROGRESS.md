# Implementation Progress

Plan: `/home/danny/.claude/plans/functional-cooking-mist.md`

## Status

| Phase | File(s) | Status | Notes |
|---|---|---|---|
| 1 | `src/donegal_bus/graph_io.py` | ✅ Done | |
| 2 | `src/donegal_bus/api/dependencies.py` | ✅ Done | |
| 3 | `src/donegal_bus/api/routers/graph.py` | ✅ Done | |
| 4 | `src/donegal_bus/api/routers/communities.py` | ✅ Done | |
| 5 | `src/donegal_bus/api/routers/routes.py` | ✅ Done | |
| 6 | `src/donegal_bus/api/routers/stops.py` | ✅ Done | |
| 7 | `src/donegal_bus/analysis/{sampling,accessibility,cost}.py` | ✅ Done | |
| 8 | `src/donegal_bus/api/routers/analysis.py` | ✅ Done | |
| 9 | `src/donegal_bus/helpers/{route_planning,testing,population}.py` | ✅ Done | | - got to here
| 10 | `src/donegal_bus/offline/*.py` | ✅ Done | TODO comments added |

## Key Decisions (from plan)
- Routes graph: `testing/graphml/actual_routes_added.graphml`
- Communities router: uses `get_routes_graph` (not road network)
- Connection route param: `str` (accepts 'a','b','c','d')
- Offline scripts: stubs with detailed TODO comments

## Gotchas to Remember
- `nx.read_graphml` returns **string** node IDs → relabelled to int in `load_graphml()`
- All GraphML attrs are strings → cast when reading community, top_n, route_flag, rank, weight, length
- MultiDiGraph edge access: `G.adj[u][v][0][attr]`
- GeoJSON coordinate order: `[x, y]` = `[lng, lat]`
- `community_route_order` is stored as `"0-0"`, `"1-1"` etc (not plain int) in some graph files

## COMPLETED ✅
All 48 stubs implemented, 17 offline stubs have detailed TODO comments.
All endpoints verified working. ruff + pyright: 0 errors.

## Key Bug Fixed During Implementation
- `nx.read_graphml` returns **DiGraph** (not MultiDiGraph) after `relabel_nodes`.
  `G.adj[u][v]` gives edge attrs directly — NO `[0]` key layer needed.
  Fixed in: `analysis/cost.py`, `helpers/route_planning.py`, `helpers/testing.py`

---

## Offline Pipeline Modules

| Module | Status | Notes |
|---|---|---|
| `offline/population_data.py` | ✅ Done | Two-pass Nominatim geocoding (townland, then townland+town) |
| `offline/create_graph.py` | ✅ Done | `ox.graph_from_place`, filter to primary/secondary/tertiary/trunk |
| `offline/apply_edge_weights.py` | ✅ Done | `ox.nearest_edges(G, X=lngs, Y=lats)` (v2.1 API) |
| `offline/community_detection.py` | ✅ Done | Spectral clustering; `sc.npy` cache for reproducibility; 18→14 merge |
| `offline/node_ranking.py` | ✅ Done | PageRank on undirected view; top-N per community; furthest-apart endpoint detection |
| `offline/route_pathfinding.py` | ✅ Done | Community routes + 4 connecting routes (a/b/c/d); weighted-degree filter |
| `offline/prepare_test_data.py` | ✅ Done | GTFS parsing (pandas, no partridge); `ox.nearest_nodes` (v2.1); `gpd.sjoin(predicate=)` |

All offline modules complete. ruff + pyright: 0 errors.

### Key Decisions (offline modules)
- `load_graphml` (DiGraph) used for all nx-only scripts; `ox.load_graphml` (MultiDiGraph) only where osmnx spatial functions are needed
- `community_route_order` stored as int (matches notebook), `connection_order` as `"label-i"` string
- `compute_connecting_routes` reads the file written by `compute_community_routes` (same `cutoff` param, default 90)
- `build_merged_test_graph`: downloads full Donegal network (`network_type="all"`), spatial-joins route attrs, handles missing nodes via `ox.nearest_nodes`
- osmnx v0.15→v2.1: `ox.nearest_edges(X=lng, Y=lat)`, `ox.nearest_nodes(X=lng, Y=lat)`, `gpd.sjoin(predicate="intersects")`

---

## Web UI Integration (implementation-plan2.md)

| Phase | File(s) | Status | Notes |
|---|---|---|---|
| 1 | `config.py`, `api/app.py` | ✅ Done | CORS + GZip + `/health` endpoint |
| 2 | `api/routers/graph.py` | ✅ Done | Bbox filtering (`min_lng/lat`, `max_lng/lat`) for nodes + edges |
| 3 | `api/dependencies.py`, `api/routers/analysis.py` | ✅ Done | Accessibility result caching (module-level, first-request only) |
| 4 | `api/routers/stops.py` | ✅ Done | Added `distance_deg` property to `/stops/nearest` response |
| 5 | `models/error.py` (new), `api/routers/communities.py`, `api/routers/routes.py` | ✅ Done | `ErrorResponse` model + 404 docs on 3 endpoints |

All phases complete. ruff + pyright: 0 errors.

### Key Decisions (web UI plan)
- `cors_origins` in `Settings`, overridable via `DONEGAL_CORS_ORIGINS` env var
- Default CORS origins: `localhost:3000`, `localhost:5173`, `localhost:8080`
- Bbox filter: all 4 params required together; omitting any returns full collection
- Accessibility cache: module-level `_accessibility_cache`, never expires (graph is static)
- `/stops/nearest` keeps `FeatureCollection` shape, adds `distance_deg` to properties
- `ErrorResponse(detail: str)` added to OpenAPI for `communities/{label}`, `routes/community/{label}`, `routes/connection/{label}`

## Verification Commands
```bash
ruff check src/
pyright src/
python main.py &
sleep 2
curl -s http://localhost:8000/health
curl -s -X OPTIONS http://localhost:8000/graph/summary \
  -H "Origin: http://localhost:3000" -H "Access-Control-Request-Method: GET" -v 2>&1 | grep access-control
curl -s "http://localhost:8000/graph/nodes?min_lng=-7.8&min_lat=54.9&max_lng=-7.7&max_lat=55.0" \
  | python -c "import json,sys; print(len(json.load(sys.stdin)['features']), 'features')"
curl -s "http://localhost:8000/stops/nearest?lat=54.95&lng=-7.73" | python -m json.tool
kill %1
```
