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

## If Resuming
Next step: run `ruff check src/` and `pyright src/` to verify linting, then start server and hit endpoints.

## Verification Commands
```bash
ruff check src/
pyright src/
python -c "from donegal_bus.api.app import create_app; app = create_app()"
# Then start server and curl all endpoints (see plan for full list)
```
