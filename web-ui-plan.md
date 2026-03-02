# Donegal Bus Routes — Web UI Implementation Plan

## Framework Decision: Why Not Reflex

Reflex is a great framework for Python-first dashboards, but there are three specific reasons it's the wrong fit for **this** project:

1. **Map components are Enterprise-only or broken.** Reflex's built-in Leaflet components require the paid Enterprise tier. The open-source alternative — wrapping `react-leaflet` via `NoSSRComponent` — has open bugs (GitHub issue #4836, `render is not a function` errors) and limited support for GeoJSON layers, layer toggling, and custom styling. Your project is fundamentally a map visualisation. Fighting the framework on its core feature is not a good trade.

2. **Architectural redundancy.** Reflex runs its own server (compiled Next.js frontend + FastAPI-based backend with WebSocket state sync). You already have a working FastAPI backend with well-designed GeoJSON endpoints. Using Reflex means either running two servers with HTTP calls between them, or migrating your API logic into Reflex's state management — which throws away the API work and defeats the portfolio goal of showcasing it.

3. **Portfolio signal.** For GIS-adjacent roles, demonstrating you can consume a spatial API and render interactive maps with industry-standard tools (Leaflet, Mapbox GL) carries more weight than using a Python-only abstraction layer. It also shows you can work across the stack rather than only in Python.

### Recommendation: Vite + TypeScript + Leaflet

This aligns with your earlier thinking and hits the sweet spot:

- **Vite + vanilla TypeScript**: fast build tooling, no heavy framework, matches your stated goal of "nice looking frontend but not interested in working as frontend dev"
- **Leaflet**: the standard open-source mapping library, native GeoJSON support, lightweight, huge plugin ecosystem
- **Your existing FastAPI backend**: consumed via `fetch()` calls, showcasing the API design

The frontend is a thin presentation layer. The intelligence lives in your API and graph pipeline. The UI's job is to make that visible.

---

## Architecture Overview

```
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│  Frontend (Vite + TS + Leaflet) │────▶│  FastAPI Backend (:8000)     │
│  localhost:5173                  │     │                              │
│                                  │     │  /graph/*     → GeoJSON      │
│  ┌────────────┐  ┌────────────┐ │     │  /communities/* → JSON       │
│  │  Map View   │  │  Sidebar   │ │     │  /routes/*    → GeoJSON      │
│  │  (Leaflet)  │  │  (controls │ │     │  /stops/*     → GeoJSON      │
│  │             │  │   + info)  │ │     │  /analysis/*  → JSON         │
│  └────────────┘  └────────────┘ │     │  /health      → status       │
└─────────────────────────────────┘     └──────────────────────────────┘
```

Single-page application. No client-side routing needed — the "pages" are different map states controlled by a step navigator.

---

## UI Concept: Guided Pipeline Walkthrough

The core UX metaphor is a **step-by-step guided tour** through the dissertation pipeline, with the map as the centrepiece and a sidebar providing narrative context, controls, and results at each step.

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Header: "Rural Bus Route Planning — County Donegal"         │
├──────────────────────┬───────────────────────────────────────┤
│                      │                                       │
│   Sidebar            │          Map (Leaflet)                │
│   ~350px             │          fills remaining space        │
│                      │                                       │
│  ┌────────────────┐  │   OSM tiles + GeoJSON overlays        │
│  │ Step Navigator  │  │                                       │
│  │ (1) (2) (3)... │  │                                       │
│  └────────────────┘  │                                       │
│                      │                                       │
│  Step title          │                                       │
│  Narrative text      │                                       │
│  Controls/filters    │                                       │
│  Key metrics         │                                       │
│                      │                                       │
│  ┌────────────────┐  │                                       │
│  │ Charts/stats   │  │                                       │
│  │ (when needed)  │  │                                       │
│  └────────────────┘  │                                       │
│                      │                                       │
├──────────────────────┴───────────────────────────────────────┤
│  Footer (optional): data source attribution, GitHub link     │
└──────────────────────────────────────────────────────────────┘
```

### Pipeline Steps (mapped to dissertation stages)

Each step corresponds to a stage from the original notebook sequence and loads different data layers onto the map.

| Step | Title | Map Layers | Sidebar Content | API Endpoints Used |
|------|-------|------------|----------------|--------------------|
| 0 | Introduction | Donegal outline / base map | Project abstract, aims | — (static) |
| 1 | Road Network | All nodes + edges (road graph) | Graph summary stats | `GET /graph/summary`, `GET /graph/nodes`, `GET /graph/edges` |
| 2 | Population Data | Edges coloured by weight + townland markers | Population summary, townland count | `GET /analysis/population`, `GET /graph/edges` (weight property) |
| 3 | Community Detection | Nodes coloured by community (14 colours) | Community list, node/edge counts | `GET /communities/`, `GET /graph/nodes` (community property) |
| 4 | Node Ranking | Top-ranked nodes highlighted per community | Top-ranked node lists, route endpoints | `GET /communities/{label}` |
| 5 | Generated Routes | Route LineStrings + generated stops | Route details, stop counts | `GET /routes/geojson`, `GET /stops/generated` |
| 6 | Existing Routes | Actual LocalLink stops overlaid | Comparison narrative | `GET /stops/actual` |
| 7 | Accessibility | Heatmap/radius visualisation | Accessibility test results | `GET /analysis/accessibility` |
| 8 | Cost Comparison | Side-by-side route display | Cost breakdown table, bar chart | `GET /analysis/cost` |
| 9 | Nearest Stop Finder | Click-to-query interactive mode | Nearest stop result | `GET /stops/nearest?lat=&lng=` |

---

## Implementation Phases

### Phase 1: Project Scaffolding

**Goal**: Vite project initialised, Leaflet rendering, FastAPI CORS verified.

**Files to create**:
```
web/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── src/
│   ├── main.ts           # entry point, initialise map + sidebar
│   ├── style.css          # global styles
│   ├── api.ts             # fetch wrapper for all API calls
│   ├── map.ts             # Leaflet map initialisation + layer management
│   ├── sidebar.ts         # sidebar DOM rendering
│   └── steps/
│       └── index.ts       # step definitions (exported array)
└── public/
    └── favicon.svg
```

**Dependencies**:
```json
{
  "devDependencies": {
    "typescript": "^5.7",
    "vite": "^6.0"
  },
  "dependencies": {
    "leaflet": "^1.9",
    "@types/leaflet": "^1.9"
  }
}
```

No UI framework. The sidebar is built with plain DOM manipulation. This keeps the bundle tiny and the code straightforward.

**Key implementation details**:

- `api.ts` exports typed fetch functions: `fetchGraphSummary()`, `fetchGraphNodes()`, `fetchRouteGeoJSON()`, etc. Each returns typed interfaces matching your Pydantic models. Base URL configurable via `import.meta.env.VITE_API_URL` (defaults to `http://localhost:8000`).
- `map.ts` creates the Leaflet map centred on Donegal (~54.95, -7.73, zoom 9), uses OSM tiles, and exports functions like `addGeoJSONLayer()`, `clearLayers()`, `fitBoundsToLayer()`.
- Verify CORS works by fetching `/health` on page load.

**Acceptance criteria**: Map renders with OSM tiles centred on Donegal. `/health` fetch succeeds in browser console.

---

### Phase 2: Step Navigation + Road Network (Steps 0–1)

**Goal**: Sidebar step navigator functional. Introduction and road network steps working.

**New files**:
```
src/
├── steps/
│   ├── types.ts           # Step interface definition
│   ├── step0-intro.ts     # Introduction step
│   └── step1-network.ts   # Road network step
├── layers.ts              # layer factory functions (GeoJSON → Leaflet layers)
└── components/
    └── loading.ts         # loading spinner for async data
```

**Step interface**:
```typescript
interface PipelineStep {
  id: number;
  title: string;
  narrativeHtml: string;
  onEnter: (map: L.Map, sidebar: HTMLElement) => Promise<void>;
  onExit: (map: L.Map) => void;
}
```

Each step's `onEnter` fetches data, adds map layers, and populates the sidebar. `onExit` cleans up layers.

**Step 0 (Introduction)**:
- Static content: project abstract, aims, study region description.
- Map shows Donegal boundary outline (can be a static GeoJSON file or just the base map).

**Step 1 (Road Network)**:
- Fetches `/graph/summary` → displays node/edge counts in sidebar.
- Fetches `/graph/nodes` → renders as small circle markers.
- Fetches `/graph/edges` → renders as polylines.
- Use bbox filtering to load visible area only (leverage existing `min_lng/lat/max_lng/lat` params). Load full extent initially, then filter on pan/zoom if performance is an issue.

**Performance note**: The road network has ~3,500 nodes and ~7,200 edges. Leaflet can handle this without virtualisation, but use `L.CircleMarker` (canvas renderer) for nodes, not `L.Marker` (DOM-based). Enable Leaflet's canvas renderer:
```typescript
const map = L.map('map', { preferCanvas: true });
```

**Acceptance criteria**: Can click through steps 0 and 1. Road network draws on map. Graph summary stats display in sidebar.

---

### Phase 3: Population + Communities (Steps 2–3)

**Goal**: Edge weight visualisation and community colouring.

**New files**:
```
src/
├── steps/
│   ├── step2-population.ts
│   └── step3-communities.ts
├── colours.ts             # colour palette for 14 communities
└── components/
    └── legend.ts          # map legend component
```

**Step 2 (Population Data)**:
- Fetches `/graph/edges` and `/analysis/population`.
- Colours edges by `weight` property using a sequential colour scale (e.g., light yellow → dark red).
- Sidebar shows population summary and a colour scale legend.
- Optional: if you have townland coordinates in the population CSV, plot them as small dots sized by population.

**Step 3 (Community Detection)**:
- Fetches `/graph/nodes` (which include `community` property) and `/communities/`.
- Colours nodes by community label using a 14-colour categorical palette.
- Sidebar shows list of communities with node/edge counts.
- Clicking a community in the sidebar highlights it on the map (dim other communities).
- Add a legend mapping colour → community label.

**Colour palette**: Use a perceptually distinct 14-colour palette. A good choice for categorical geo data is a modified Tableau 20 or ColorBrewer Set3. Define in `colours.ts` as a simple label→hex map.

**Acceptance criteria**: Edges coloured by weight on step 2. Nodes coloured by community on step 3. Legend displays on map. Community list is interactive.

---

### Phase 4: Ranking + Routes (Steps 4–5)

**Goal**: Show node ranking results and generated bus routes.

**New files**:
```
src/
├── steps/
│   ├── step4-ranking.ts
│   └── step5-routes.ts
└── components/
    └── route-card.ts      # collapsible route detail card
```

**Step 4 (Node Ranking)**:
- Fetches `/communities/{label}` for each community.
- Highlights `top_ranked_nodes` with larger markers.
- Marks `route_start` (green) and `route_end` (red) with distinct icons.
- Sidebar shows the top-ranked nodes per community (selectable).
- On community select: zooms to community bounds, shows endpoint pair.

**Step 5 (Generated Routes)**:
- Fetches `/routes/geojson` → draws route LineStrings, coloured by community.
- Fetches `/stops/generated` → draws stop markers along routes.
- Fetches `/routes/connection/{a,b,c,d}` → draws connecting routes in a distinct style (dashed lines).
- Sidebar shows list of routes with stop counts and total weights.
- Clicking a route in the sidebar isolates it on the map.

**Acceptance criteria**: Route lines drawn correctly. Stops visible along routes. Connection routes distinguishable. Individual route selection works.

---

### Phase 5: Testing + Analysis (Steps 6–8)

**Goal**: Actual routes comparison, accessibility, and cost analysis.

**New files**:
```
src/
├── steps/
│   ├── step6-actual.ts
│   ├── step7-accessibility.ts
│   └── step8-cost.ts
└── components/
    ├── bar-chart.ts       # simple canvas/SVG bar chart
    └── stats-table.ts     # key-value stat display
```

**Step 6 (Existing Routes)**:
- Fetches `/stops/actual` → draws actual LocalLink stops with a different marker style (e.g., squares vs circles).
- Overlays with generated stops from step 5 for visual comparison.
- Sidebar: narrative about existing service limitations, stop count comparison.

**Step 7 (Accessibility)**:
- Fetches `/analysis/accessibility` (warn user this may take 10–30s on first load).
- Shows loading indicator during computation.
- Results: display avg destinations reached, avg path length.
- Map visualisation: for each test result, draw a circle marker at the source node, coloured by `destinations_reached` (green = high, red = low).
- Sidebar: summary stats, histogram of destinations reached (simple SVG/canvas chart).

**Step 8 (Cost Comparison)**:
- Fetches `/analysis/cost`.
- Sidebar: side-by-side table of generated vs actual route costs.
- Bar chart comparing total costs.
- Highlight the key finding: new routes are more accessible at ~10% higher cost.

**Chart approach**: For the 2–3 simple charts needed, use inline SVG generation in TypeScript. No need for a charting library for bar charts and histograms. If you want something more polished, Chart.js is a lightweight option (~60KB).

**Acceptance criteria**: All three steps render data correctly. Accessibility loading state works. Cost comparison is clear and visually compelling.

---

### Phase 6: Interactive Features (Step 9)

**Goal**: Nearest stop finder, click interactions, polish.

**New files**:
```
src/
├── steps/
│   └── step9-finder.ts
└── components/
    └── popup.ts           # custom Leaflet popup template
```

**Step 9 (Nearest Stop Finder)**:
- Click anywhere on the map → sends coordinates to `GET /stops/nearest?lat=&lng=`.
- Draws a line from click point to nearest stop.
- Popup shows stop details (osmid, community, distance).
- Sidebar: instructions, last query result, distance in human-readable units (convert `distance_deg` to approximate km using the latitude-appropriate conversion factor).

**Additional interactivity polish across all steps**:
- Tooltips on hover for nodes/edges/stops showing key properties.
- Smooth transitions when switching steps (fade layers in/out).
- Responsive sidebar (collapsible on narrow screens).
- Map attribution for OSM data source.

**Acceptance criteria**: Click-to-find-stop works. Popups display correctly. All 10 steps navigable end-to-end.

---

### Phase 7: Styling + Polish

**Goal**: Production-quality visual design.

**New files**:
```
src/
├── style.css              # (expanded)
└── fonts/                 # self-hosted web fonts if desired
```

**Design direction**: Cartographic / editorial. The map is the hero element. The sidebar uses clean typography, subtle borders, and generous whitespace. Colour palette derived from the community colours but with a neutral UI chrome.

**Specific polish items**:
- Custom map tile layer: consider Stamen Toner Lite or CartoDB Positron for a cleaner base map that lets your data layers stand out.
- Loading skeleton states for sidebar content.
- Smooth CSS transitions on step changes.
- Print/export: add a "Screenshot" button using `leaflet-image` or `html2canvas` for the map view.
- Mobile: at narrow widths, sidebar becomes a bottom drawer.

---

### Phase 8: Build + Deployment

**Goal**: Production build, Docker integration, deployment-ready.

**Changes**:
```
web/
├── Dockerfile             # multi-stage: node build → nginx serve
├── nginx.conf             # SPA routing + API proxy
└── vite.config.ts         # (add proxy for dev)

docker-compose.yml         # (project root) — API + web services
```

**Dev setup** (`vite.config.ts` proxy):
```typescript
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
```

This means frontend code fetches from `/api/graph/summary` in dev, which proxies to `http://localhost:8000/graph/summary`. In production, nginx does the same routing.

**Docker compose**:
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./graph:/app/graph:ro
      - ./testing:/app/testing:ro
      - ./data:/app/data:ro

  web:
    build: ./web
    ports: ["80:80"]
    depends_on: [api]
```

**Acceptance criteria**: `docker compose up` serves the full application. Frontend loads, fetches data from API, all steps work.

---

## File Structure Summary

```
donegal-bus-routes/
├── src/donegal_bus/         # existing FastAPI backend
├── main.py                   # existing
├── web/                      # NEW — frontend
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── public/
│   │   └── favicon.svg
│   └── src/
│       ├── main.ts
│       ├── style.css
│       ├── api.ts
│       ├── map.ts
│       ├── sidebar.ts
│       ├── layers.ts
│       ├── colours.ts
│       ├── steps/
│       │   ├── types.ts
│       │   ├── index.ts
│       │   ├── step0-intro.ts
│       │   ├── step1-network.ts
│       │   ├── step2-population.ts
│       │   ├── step3-communities.ts
│       │   ├── step4-ranking.ts
│       │   ├── step5-routes.ts
│       │   ├── step6-actual.ts
│       │   ├── step7-accessibility.ts
│       │   ├── step8-cost.ts
│       │   └── step9-finder.ts
│       └── components/
│           ├── loading.ts
│           ├── legend.ts
│           ├── route-card.ts
│           ├── bar-chart.ts
│           ├── stats-table.ts
│           └── popup.ts
├── docker-compose.yml        # NEW — orchestration
└── ...
```

---

## API Changes Needed Before Starting

Review your existing endpoints against what the UI needs. A few small additions would help:

1. **`GET /graph/edges` — expose `weight` in GeoJSON properties.** Already done — `weight` and `length` are in the edge feature properties. Confirmed in `graph_io.py`.

2. **`GET /graph/nodes` — ensure `community` is always present.** Currently included if present. For steps 2–3, the frontend needs this reliably. Already handled by the routes graph; road network graph may not have community labels — that's fine since step 3 uses routes graph data via `/communities/`.

3. **`GET /communities/{label}` — add node coordinates to response.** Currently returns `top_ranked_nodes` as a list of osmid ints. The frontend needs coordinates to place markers. Two options:
   - Add `top_ranked_coords: list[{osmid, x, y}]` to `CommunityDetail`
   - Or have the frontend cross-reference with `/graph/nodes` data (already loaded)
   
   Recommend option (a) for cleaner API design — the frontend shouldn't have to join datasets client-side.

4. **`GET /routes/connection/{label}` — add GeoJSON variant.** Currently returns `RouteDetail` (JSON with stops). For map rendering, a GeoJSON LineString would be more convenient. Consider adding a `/routes/connections/geojson` endpoint similar to `/routes/geojson`.

5. **Population townland coordinates.** Step 2 benefits from showing townland locations on the map. Consider adding `GET /analysis/population/townlands` returning a GeoJSON FeatureCollection of townlands with lat/lng/population. This is a simple read from the CSV.

These are all small additions, not refactors. They can be done incrementally as each phase requires them.

---

## Estimated Effort

| Phase | Description | Rough Estimate |
|-------|-------------|---------------|
| 1 | Scaffolding + map + CORS | 2–3 hours |
| 2 | Step nav + road network | 3–4 hours |
| 3 | Population + communities | 3–4 hours |
| 4 | Ranking + routes | 4–5 hours |
| 5 | Testing + analysis | 4–5 hours |
| 6 | Interactive features | 2–3 hours |
| 7 | Styling + polish | 3–4 hours |
| 8 | Build + Docker | 2–3 hours |
| **Total** | | **~24–30 hours** |

These assume the API endpoints are working (which they are) and that you're comfortable with basic TypeScript (which, given you've been working with Python typing and Pydantic, you largely already are — TypeScript's type system will feel familiar).

---

## Suggested Build Order

Start with Phase 1, then work through phases 2–6 sequentially since each step builds on the previous map state. Phase 7 (styling) can be done incrementally throughout. Phase 8 (Docker) can be done early or late — doing it early means you can test the full stack from the start.

The single most impactful milestone is **Phase 4 complete** — at that point you have the full road network → communities → routes pipeline visible on the map, which tells the core story of the dissertation.
