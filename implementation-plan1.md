# Implementation Plan: Port Donegal Bus Routes to API Package

## Context

The `src/donegal_bus/` package was scaffolded with 63 stub functions (`NotImplementedError`) across 20 files. All Pydantic models are complete. All GraphML data files exist on disk. The goal is to implement all stubs so the FastAPI API serves live data, and the analysis/helper modules are functional. Offline scripts will be left as stubs with detailed TODO comments since the data already exists.

### Decisions Made
- **Routes graph**: Load `testing/graphml/actual_routes_added.graphml` (has both generated + actual stop data)
- **Communities router**: Switch `Depends` from `get_road_network_graph` to `get_routes_graph` (routes graph has community/rank/route_flag attributes)
- **Connection route labels**: Change `/routes/connection/{label}` param from `int` to `str` (accepts 'a','b','c','d')
- **Offline scripts**: Leave as stubs with TODO comments

---

## Phase 1: `graph_io.py` — Foundation (8 functions)

**File**: `src/donegal_bus/graph_io.py`

Every other module depends on this. Implement all 8 functions.

### 1. `load_graphml(path)` (line 11)
- Use `nx.read_graphml(str(path))`
- `nx.read_graphml` returns string node IDs — relabel to int with `nx.relabel_nodes(G, {n: int(n) for n in G.nodes})`
- Return `nx.MultiDiGraph`

### 2. `save_graphml(G, path)` (line 16)
- `nx.write_graphml(G, str(path))`

### 3. `fix_edge_weights(G)` (line 21)
- Port from `route_planning/helpers/route_planning_helper.py:21-32`
- Iterate `G.edges(data=True, keys=True)`, convert `weight` and `length` attrs from str to float
- Mutate in-place AND return G

### 4. `fix_bool_attributes(G, attr)` (line 26)
- Iterate `G.nodes(data=True)`, map `"True"` → `True`, `"False"` → `False` for the given attr name
- Also check edges (some bool attrs may be on edges)
- Return G

### 5. `graph_summary(G)` (line 31)
- Count nodes, edges, unique `community` values from node data, `nx.is_weakly_connected(G)`
- Return `GraphSummary(num_nodes=, num_edges=, num_communities=, is_connected=)`

### 6. `nodes_to_geojson(G)` (line 36)
- For each node: `PointGeometry(coordinates=(float(x), float(y)))`, wrap in `Feature` with properties including osmid and any available attrs (community, rank, top_n, route_flag, community_route, connection_route, actual_stop)
- GeoJSON coordinate order: `[longitude, latitude]` = `[x, y]`

### 7. `edges_to_geojson(G)` (line 41)
- For each edge: `LineStringGeometry(coordinates=[(u.x, u.y), (v.x, v.y)])`, wrap in `Feature`
- Properties: u, v, weight, length

### 8. `route_path_to_geojson(G, path)` (line 46)
- Build single `LineStringGeometry` from path node coordinates
- Wrap in `Feature` with `num_stops` property, return as single-feature `FeatureCollection`

### New imports needed
- Add `from donegal_bus.models.geojson import Feature, LineStringGeometry, PointGeometry` to existing imports

### Verification
```bash
python -c "
from pathlib import Path
from donegal_bus.graph_io import load_graphml, fix_edge_weights, graph_summary, nodes_to_geojson
G = load_graphml(Path('graph/graphml/donegal_osm_weights_applied.graphml'))
G = fix_edge_weights(G)
s = graph_summary(G)
print(f'Nodes: {s.num_nodes}, Edges: {s.num_edges}')
fc = nodes_to_geojson(G)
print(f'GeoJSON features: {len(fc.features)}')
print('Phase 1 PASSED')
"
```

---

## Phase 2: `api/dependencies.py` — Graph Loading (1 function)

**File**: `src/donegal_bus/api/dependencies.py`

### `load_graphs()` (line 20)
- Add imports: `from donegal_bus.graph_io import load_graphml, fix_edge_weights, fix_bool_attributes`
- Load `_road_network` from `settings.graph_graphml_path / "donegal_osm_weights_applied.graphml"`
- Apply `fix_edge_weights(_road_network)`
- Load `_routes_graph` from `settings.testing_graphml_path / "actual_routes_added.graphml"`
- Apply `fix_edge_weights`, then `fix_bool_attributes` for `"community_route"`, `"connection_route"`, `"actual_stop"`
- Use `global _road_network, _routes_graph`

### Verification
```bash
python -c "
from donegal_bus.api.app import create_app
app = create_app()
from donegal_bus.api.dependencies import get_road_network_graph, get_routes_graph
G1 = get_road_network_graph()
G2 = get_routes_graph()
print(f'Road network: {G1.number_of_nodes()} nodes, Routes: {G2.number_of_nodes()} nodes')
"
```

---

## Phase 3: API Routers — Graph (3 endpoints)

**File**: `src/donegal_bus/api/routers/graph.py`

### `get_summary` (line 13)
- `return graph_summary(G)` — add import from `donegal_bus.graph_io`

### `get_nodes` (line 21)
- `return nodes_to_geojson(G)`

### `get_edges` (line 29)
- `return edges_to_geojson(G)`

### Verification
```bash
curl http://localhost:8000/graph/summary
curl http://localhost:8000/graph/nodes | python -c "import json,sys; print(len(json.load(sys.stdin)['features']))"
```

---

## Phase 4: API Routers — Communities (2 endpoints)

**File**: `src/donegal_bus/api/routers/communities.py`

### Scaffold change: Switch dependency
- Line 6: Change import from `get_road_network_graph` to `get_routes_graph`
- Lines 14, 23: Change `Depends(get_road_network_graph)` to `Depends(get_routes_graph)`

### `list_communities` (line 12)
- Group nodes by `int(data.get("community", 0))`
- For each community: count nodes and edges via `G.subgraph(nodes)`
- Return `CommunityCollection(communities=[...], total=len(communities))`

### `get_community` (line 20)
- Filter nodes for the given label, 404 if none found
- Extract `top_ranked_nodes` (where `int(top_n) > 0`)
- Extract `route_start` (where `int(route_flag) == 1`), `route_end` (where `int(route_flag) == 2`)
- Add `from fastapi import HTTPException` import

---

## Phase 5: API Routers — Routes (4 endpoints)

**File**: `src/donegal_bus/api/routers/routes.py`

### Scaffold change: Connection route param type
- Line 38: Change `/{label}` param from `label: int` to `label: str`

### Route extraction helper (private function in the router module)
Build a helper `_extract_community_routes(G)` that returns a dict mapping community label → ordered list of route node IDs. Logic:
- Filter nodes where `community_route is True`
- Group by `int(community)` attribute
- Sort within each group by `int(community_route_order)` attribute
- Also handle `community_route_order` being a string

### `list_routes` (line 13)
- Use `_extract_community_routes(G)` to get all routes
- For each route: compute path weight using edge lengths, build `RouteDetail`
- Return `RouteCollection(routes=[...], total_routes=len(routes))`

### `routes_geojson` (line 21)
- For each community route: build `LineStringGeometry` from ordered node coordinates
- Return `FeatureCollection`

### `routes_by_community` (line 29)
- Extract single community route, 404 if not found
- Return `RouteDetail` with ordered `RouteStop` list

### `connection_route` (line 38)
- Filter nodes where `connection_route is True` and `connection` attr contains the label string
- Connection attr can be comma-separated (e.g. `"a, b"`), use `label in str(data.get("connection", ""))`
- Sort by `connection_order` attr, 404 if not found

---

## Phase 6: API Routers — Stops (3 endpoints)

**File**: `src/donegal_bus/api/routers/stops.py`

### `generated_stops` (line 12)
- Filter nodes where `community_route is True` or `connection_route is True`
- Convert to `FeatureCollection` of `PointGeometry` features

### `actual_stops` (line 20)
- Filter nodes where `actual_stop is True`
- Convert to `FeatureCollection` with `actual_route_order` in properties

### `nearest_stop` (line 28)
- From generated stop nodes, find min Euclidean distance to `(lng, lat)` = `(x, y)`
- Return single-feature `FeatureCollection`
- Add `import math` for `math.hypot`

---

## Phase 7: Analysis Module (7 functions)

### `analysis/sampling.py` (2 functions)

**File**: `src/donegal_bus/analysis/sampling.py`

#### `cochran_sample_size` (line 6)
- Port from `testing/helpers/testing_helper.py:42-93`
- z-score lookup dict, Cochran's formula, return `math.ceil(result)`
- Add `import math`

#### `select_sample_nodes` (line 26)
- Get all node IDs excluding stops (community_route, connection_route, actual_stop all falsy)
- `random.sample(all_nodes, min(sample_size, len(all_nodes)))`
- Add `import random`

### `analysis/accessibility.py` (3 functions)

**File**: `src/donegal_bus/analysis/accessibility.py`

#### `test_single_node` (line 8)
- Create 20km ego graph: `nx.ego_graph(G, source, radius=20000, distance=weight)`
- For each destination in ego graph: try `nx.shortest_path`, compute path length via `G.adj[u][v][0][weight]`
- Count reached, compute average path length
- Return `AccessibilityTestResult`

#### `run_accessibility_tests` (line 28)
- Loop `test_single_node` over all sample_nodes, collect results

#### `summarize_results` (line 48)
- Compute averages of `destinations_reached` and `avg_path_length`
- Return `AccessibilitySummary(total_tests=len(results), ...)`

### `analysis/cost.py` (2 functions)

**File**: `src/donegal_bus/analysis/cost.py`

#### `estimate_route_cost` (line 8)
- Sum edge lengths: `sum(float(G.adj[u][v][0]["length"]) for u, v in zip(path, path[1:]))`
- `distance_km = total / 1000`, `fuel = distance_km * consumption`, `cost = fuel * price`
- Return `RouteCost`

#### `compare_costs` (line 30)
- Sum `estimated_cost_eur` for each list
- Return `CostComparison(generated_routes=generated, actual_routes=actual, total_generated_cost=..., total_actual_cost=...)`

---

## Phase 8: API Routers — Analysis (3 endpoints)

**File**: `src/donegal_bus/api/routers/analysis.py`

### `population_summary` (line 29)
- Read `settings.population_csv` with pandas
- Return `PopulationSummary(total_townlands=len(df), total_population=df["population"].sum(), townlands_with_coords=df["lat"].notna().sum())`
- Add imports: `import pandas as pd`, `from donegal_bus.api.dependencies import get_settings`

### `cost_comparison` (line 20)
- Extract community route paths from G (reuse route extraction logic from routes router)
- Call `estimate_route_cost` for each generated route
- Extract actual route paths from G (nodes with `actual_stop is True`, ordered by `actual_route_order`)
- Call `estimate_route_cost` for each actual route
- Return `compare_costs(generated, actual)`
- Add imports from `donegal_bus.analysis.cost`

### `accessibility_summary` (line 13)
- Get generated stop node IDs as destinations
- Compute sample size via `cochran_sample_size(G.number_of_nodes())`
- Select sample nodes via `select_sample_nodes`
- Run `run_accessibility_tests` and `summarize_results`
- Note: This endpoint may be slow (~seconds). Consider adding response caching later.
- Add imports from `donegal_bus.analysis.sampling`, `donegal_bus.analysis.accessibility`

---

## Phase 9: Helper Modules (17 functions)

### `helpers/route_planning.py` (11 functions)

**File**: `src/donegal_bus/helpers/route_planning.py`

Port each function from `route_planning/helpers/route_planning_helper.py`:

| Function | Lines in old file | Key porting notes |
|---|---|---|
| `assign_community_labels` | 8-18 | Direct port using `nx.set_node_attributes` + loop |
| `convert_edge_weights_to_floats` | 21-32 | `nx.get_edge_attributes` + dict comprehension + `nx.set_edge_attributes` |
| `split_into_community_graphs` | 199-217 | Replace `ox.graph_to_gdfs` with `set(d["community"] for _, d in G.nodes(data=True))` |
| `get_n_highest_ranked_nodes_in_community` | 35-58 | Direct port, add `import pandas as pd` at function level |
| `get_top_n_ranked_nodes_per_community` | 83-107 | Cast `int(y["top_n"])` for string comparison |
| `get_node_coordinates` | 110-133 | Direct port |
| `find_furthest_apart_nodes` | 136-169 | Replace `ox.distance.euclidean_dist_vec` with `math.hypot(float(y1)-float(y2), float(x1)-float(x2))` |
| `greatest_distance_between_top_ranked_nodes` | 61-80 | Composition — calls 3 functions above |
| `assign_route_start_end_points` | 172-196 | Direct port |
| `path_weight` | 220-231 | `sum(float(G.adj[u][v][0][weight]) for u, v in zip(path, path[1:]))` |
| `find_highest_weighted_simple_path` | 234-265 | Replace `ox.graph_to_gdfs` with pure nx for finding route_flag nodes. `max(nx.all_simple_paths(...), key=lambda p: path_weight(G, p))` |

### `helpers/testing.py` (3 functions)

**File**: `src/donegal_bus/helpers/testing.py`

| Function | Source | Notes |
|---|---|---|
| `get_path_of_route` | `testing/helpers/testing_helper.py:6-18` | Direct port |
| `find_shortest_path_to_destinations` | `testing/helpers/testing_helper.py:21-39` | Remove `print_all` param, add try/except for `NetworkXNoPath`, use local `path_weight` calc |
| `sample_size` | `testing/helpers/testing_helper.py:42-93` | Direct port of Cochran's formula, returns `float` |

### `helpers/population.py` (3 functions)

**File**: `src/donegal_bus/helpers/population.py`

| Function | Source | Notes |
|---|---|---|
| `extract_county_townlands_from_source_data` | `graph/helpers/population_helper.py:8-29` | Now takes explicit `source_xlsx` and `output_csv` args. Add `from openpyxl import load_workbook` |
| `extract_lat_long_from_nominatim` | `graph/helpers/population_helper.py:32-57` | Direct port, add `from OSMPythonTools.nominatim import Nominatim` |
| `lookup_osm_coordinates` | `graph/helpers/population_helper.py:60-71` | Direct port |

---

## Phase 10: Offline Scripts — TODO Comments Only

**Files**: All 7 files in `src/donegal_bus/offline/`

For each stub function, replace `raise NotImplementedError` with a detailed TODO comment block describing:
- What the function should do
- Which old notebook/cell contains the reference implementation
- Key API changes needed (osmnx v0.15 → v2.1)
- Input/output file paths

Leave the `NotImplementedError` in place below the TODO so callers get a clear error.

---

## Key Gotchas

1. **String node IDs**: `nx.read_graphml` returns string IDs. Must relabel to int in `load_graphml()`.
2. **String attributes**: GraphML stores everything as strings. Cast `int()` / `float()` when reading community, top_n, route_flag, rank, weight, length.
3. **MultiDiGraph edge access**: Always use `G.adj[u][v][0][attr]` (keyed by edge index 0).
4. **GeoJSON coordinate order**: `[x, y]` = `[longitude, latitude]`.
5. **osmnx not needed by API**: The API layer uses only `networkx` for graph operations. No osmnx migration needed for API code.
6. **`find_highest_weighted_simple_path`** is O(exponential) — only used by offline scripts, never called by the API.

---

## Verification (end-to-end)

After all phases:

```bash
# Linting
ruff check src/
pyright src/

# App startup
python -c "from donegal_bus.api.app import create_app; app = create_app()"

# Start server and test all endpoint groups
python main.py &
sleep 2
curl -s http://localhost:8000/graph/summary | python -m json.tool
curl -s http://localhost:8000/communities/ | python -m json.tool
curl -s http://localhost:8000/routes/ | python -m json.tool
curl -s http://localhost:8000/stops/generated | python -c "import json,sys; print(f'Generated: {len(json.load(sys.stdin)[\"features\"])} stops')"
curl -s http://localhost:8000/stops/actual | python -c "import json,sys; print(f'Actual: {len(json.load(sys.stdin)[\"features\"])} stops')"
curl -s "http://localhost:8000/stops/nearest?lat=55.0&lng=-7.5" | python -m json.tool
curl -s http://localhost:8000/analysis/population | python -m json.tool
curl -s http://localhost:8000/analysis/cost | python -m json.tool
curl -s http://localhost:8000/analysis/accessibility | python -m json.tool
kill %1
```

## Implementation Order Summary

| Phase | File(s) | Stubs | Depends on |
|---|---|---|---|
| 1 | `graph_io.py` | 8 | — |
| 2 | `api/dependencies.py` | 1 | Phase 1 |
| 3 | `api/routers/graph.py` | 3 | Phase 2 |
| 4 | `api/routers/communities.py` | 2 | Phase 2 |
| 5 | `api/routers/routes.py` | 4 | Phase 2 |
| 6 | `api/routers/stops.py` | 3 | Phase 2 |
| 7 | `analysis/sampling.py`, `accessibility.py`, `cost.py` | 7 | Phase 1 |
| 8 | `api/routers/analysis.py` | 3 | Phase 7 |
| 9 | `helpers/route_planning.py`, `testing.py`, `population.py` | 17 | Phase 1 |
| 10 | `offline/*.py` | 17 | TODO comments only |

**Total: 65 stubs across 20 files. 48 fully implemented, 17 with TODO comments.**
