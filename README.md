# Graph Analytics for Rural Bus Route Planning — County Donegal

A GIS pipeline that generates optimised rural bus routes using community detection and population-weighted graph analysis, with an interactive web UI for exploring the results.

For the original dissertation abstract and Jupyter notebook sequence, see [README.legacy.md](README.legacy.md).

---

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────────┐
│  Web UI  (Vite + TS)    │────▶│  FastAPI  (:8000)            │
│  localhost:5173 / :80   │     │                              │
│                         │     │  /graph/*     → GeoJSON      │
│  Leaflet map            │     │  /communities/* → JSON       │
│  Guided pipeline steps  │     │  /routes/*    → GeoJSON      │
│                         │     │  /stops/*     → GeoJSON      │
└─────────────────────────┘     │  /analysis/*  → JSON         │
                                │  /health      → status       │
                                └──────────────────────────────┘
```

---

## Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose (for containerised setup)
- **Or**, for local development:
  - [uv](https://docs.astral.sh/uv/) (Python package manager)
  - [Node.js](https://nodejs.org/) 22+

---

## Quick start — Docker (recommended)

```sh
git clone https://github.com/ancodia/donegal-bus-routes.git
cd donegal-bus-routes
docker compose up --build
```

| URL | What |
|-----|------|
| `http://localhost` | Web UI |
| `http://localhost:8000/docs` | API — interactive Swagger docs |
| `http://localhost:8000/health` | API health check |

The web container waits for the API to pass its healthcheck before starting. First-time builds take a few minutes while Python wheels for the geospatial stack are compiled.

To stop:
```sh
docker compose down
```

---

## Quick start — local development

```sh
# 1. Install Python dependencies
uv sync

# 2. Start both servers with a single command
./dev.sh
```

| URL | What |
|-----|------|
| `http://localhost:5173` | Web UI with hot-module reload |
| `http://localhost:8000/docs` | API — interactive Swagger docs |

`dev.sh` auto-installs frontend dependencies (`npm install`) if `web/node_modules/` is missing.

---

## Project structure

```
donegal-bus-routes/
├── src/donegal_bus/        # FastAPI application
│   ├── api/                #   routers, dependencies, app factory
│   ├── models/             #   Pydantic response models
│   ├── analysis/           #   accessibility + cost analysis
│   ├── helpers/            #   graph/population helpers
│   └── offline/            #   data preparation scripts (run once)
├── web/                    # Frontend (Vite + TypeScript + Leaflet)
│   ├── src/
│   │   ├── api.ts          #   typed fetch wrappers for all endpoints
│   │   ├── map.ts          #   Leaflet map init + layer management
│   │   ├── sidebar.ts      #   sidebar DOM helpers
│   │   └── steps/          #   pipeline step definitions
│   ├── Dockerfile          #   multi-stage: node build → nginx serve
│   └── nginx.conf          #   SPA routing + /api proxy
├── main.py                 # uvicorn entrypoint
├── Dockerfile.api          # FastAPI container (Python 3.11 + uv)
├── Dockerfile              # legacy Jupyter image (Python 3.8)
├── docker-compose.yml      # orchestrates api + web services
├── dev.sh                  # local development launcher
├── graph/                  # road network GraphML files
├── testing/                # test graph GraphML files
└── data/                   # population CSV and LocalLink data
```

---

## Docker details

### Services

| Service | Dockerfile | Build context | Exposed port |
|---------|-----------|---------------|--------------|
| `api` | `Dockerfile.api` | project root | `8000` |
| `web` | `web/Dockerfile` | `web/` | `80` |

### Data volumes (API)

The graph and data files are **not baked into the image** — they are mounted read-only at runtime:

| Host path | Container path |
|-----------|---------------|
| `./graph/` | `/app/graph/` |
| `./testing/` | `/app/testing/` |
| `./data/` | `/app/data/` |

### Useful commands

```sh
# Rebuild a single service after a code change
docker compose up --build api

# View logs
docker compose logs -f api
docker compose logs -f web

# Open a shell in the running API container
docker compose exec api bash

# Tear down (keeps volumes)
docker compose down

# Tear down and remove all images
docker compose down --rmi all
```

### API healthcheck

The `api` service exposes a `HEALTHCHECK` on `GET /health`. The `web` service uses `depends_on: condition: service_healthy` so nginx only starts once the API is ready. Initial graph loading can take up to 60 seconds — this is normal.

---

## Legacy Jupyter setup

The original `Dockerfile` (Python 3.8, Jupyter) is preserved for running the original notebooks. See [README.legacy.md](README.legacy.md) for instructions.
