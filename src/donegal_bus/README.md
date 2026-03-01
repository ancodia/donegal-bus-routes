# donegal_bus

FastAPI package for rural bus route planning in County Donegal. Serves road network
graphs, community-based routes, bus stops, and cost/accessibility analysis via a REST API.

## Environment Setup

**Requirements**: Python 3.11+, [`uv`](https://docs.astral.sh/uv/)

```bash
# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate  # fish: source .venv/bin/activate.fish

# Install the package in editable mode
pip install -e .

# Install pre-commit hooks (ruff + pyright)
pre-commit install
```

### Configuration

Settings are loaded from environment variables (prefix `DONEGAL_`) or a `.env` file
in the project root. All paths default to subdirectories relative to the working
directory — run the server from the project root.

| Variable | Default | Description |
|---|---|---|
| `DONEGAL_GRAPH_GRAPHML_PATH` | `graph/graphml` | Road network GraphML files |
| `DONEGAL_TESTING_GRAPHML_PATH` | `testing/graphml` | Routes GraphML files |
| `DONEGAL_POPULATION_DATA_PATH` | `data/population` | Population CSV/XLSX |

Example `.env` to override paths:

```ini
DONEGAL_GRAPH_GRAPHML_PATH=/absolute/path/to/graph/graphml
DONEGAL_TESTING_GRAPHML_PATH=/absolute/path/to/testing/graphml
```

### Required data files

The server loads two graphs at startup — these must exist on disk:

| File | Used for |
|---|---|
| `graph/graphml/donegal_osm_weights_applied.graphml` | Road network (nodes/edges endpoints) |
| `testing/graphml/actual_routes_added.graphml` | Routes, communities, stops, analysis |
| `data/population/donegal_townlands_all_coordinates.csv` | Population endpoint |

## Starting the Server

Run from the **project root** (so relative data paths resolve correctly):

```bash
python main.py
```

The server starts at `http://localhost:8000` with hot-reload enabled. On startup,
both graphs are loaded into memory — expect a short pause (~2–5s).

Interactive API docs are available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Graph — `/graph`

#### `GET /graph/summary`
Road network statistics.

```bash
curl -s http://localhost:8000/graph/summary | python -m json.tool
```

```json
{
    "num_nodes": 3579,
    "num_edges": 7195,
    "num_communities": 14,
    "is_connected": false
}
```

#### `GET /graph/nodes`
All road network nodes as a GeoJSON FeatureCollection.

```bash
curl -s http://localhost:8000/graph/nodes | python -c "
import json, sys; fc = json.load(sys.stdin)
print(f'{len(fc[\"features\"])} node features')
"
```

#### `GET /graph/edges`
All road network edges as a GeoJSON FeatureCollection.

```bash
curl -s http://localhost:8000/graph/edges | python -c "
import json, sys; fc = json.load(sys.stdin)
print(f'{len(fc[\"features\"])} edge features')
"
```

---

### Communities — `/communities`

#### `GET /communities/`
All 14 detected communities with node/edge counts.

```bash
curl -s http://localhost:8000/communities/ | python -m json.tool
```

```json
{
    "communities": [
        {"label": 0, "num_nodes": 312, "num_edges": 621},
        ...
    ],
    "total": 14
}
```

#### `GET /communities/{label}`
Detail for a single community (0–13): top-ranked nodes, route start/end.

```bash
curl -s http://localhost:8000/communities/0 | python -m json.tool
```

---

### Routes — `/routes`

#### `GET /routes/`
All 14 community bus routes with stop lists and path weights.

```bash
curl -s http://localhost:8000/routes/ | python -m json.tool
```

#### `GET /routes/geojson`
All community routes as a GeoJSON FeatureCollection (LineStrings).

```bash
curl -s http://localhost:8000/routes/geojson | python -c "
import json, sys; fc = json.load(sys.stdin)
print(f'{len(fc[\"features\"])} route features')
"
```

#### `GET /routes/community/{label}`
Single community route by label (0–13).

```bash
curl -s http://localhost:8000/routes/community/0 | python -m json.tool
```

#### `GET /routes/connection/{label}`
Connection route by label. Accepts `a`, `b`, `c`, or `d`.

```bash
curl -s http://localhost:8000/routes/connection/a | python -m json.tool
curl -s http://localhost:8000/routes/connection/b | python -m json.tool
```

---

### Stops — `/stops`

#### `GET /stops/generated`
All 650 generated stops (community routes + connection routes) as GeoJSON.

```bash
curl -s http://localhost:8000/stops/generated | python -c "
import json, sys; fc = json.load(sys.stdin)
print(f'{len(fc[\"features\"])} generated stops')
"
```

#### `GET /stops/actual`
All 122 existing LocalLink stops as GeoJSON.

```bash
curl -s http://localhost:8000/stops/actual | python -c "
import json, sys; fc = json.load(sys.stdin)
print(f'{len(fc[\"features\"])} actual stops')
"
```

#### `GET /stops/nearest?lat={lat}&lng={lng}`
Nearest generated stop to a coordinate, returned as a single-feature GeoJSON.

```bash
# Near Letterkenny
curl -s "http://localhost:8000/stops/nearest?lat=54.9558&lng=-7.7342" | python -m json.tool
```

---

### Analysis — `/analysis`

> Note: `/analysis/accessibility` runs Dijkstra across a sampled set of nodes
> and may take 10–30 seconds on first call.

#### `GET /analysis/population`
Population summary from townland data.

```bash
curl -s http://localhost:8000/analysis/population | python -m json.tool
```

```json
{
    "total_townlands": 2020,
    "total_population": 157920,
    "townlands_with_coords": 2020
}
```

#### `GET /analysis/cost`
Fuel cost comparison between generated routes and existing LocalLink routes.

```bash
curl -s http://localhost:8000/analysis/cost | python -m json.tool
```

```json
{
    "generated_routes": [...],
    "actual_routes": [...],
    "total_generated_cost": 95.4,
    "total_actual_cost": 142.7
}
```

#### `GET /analysis/accessibility`
Accessibility test: sample nodes and measure how many stops are reachable
within 20 km via the road network.

```bash
curl -s http://localhost:8000/analysis/accessibility | python -m json.tool
```

```json
{
    "total_tests": 348,
    "avg_destinations_reached": 64.2,
    "avg_path_length": 8431.5
}
```

## Linting and Type Checking

```bash
ruff check src/
pyright src/
```

Both should report 0 errors. Run before committing — pre-commit hooks enforce this.
