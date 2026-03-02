"""FastAPI application factory with lifespan and router includes."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from donegal_bus.api.dependencies import get_settings, load_graphs
from donegal_bus.api.routers import analysis, communities, graph, routes, stops


def _health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load graphs into memory at startup."""
    load_graphs()
    yield


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="Donegal Bus Routes API",
        description="Rural bus route planning for County Donegal.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_api_route("/health", _health, methods=["GET"], tags=["health"])

    app.include_router(graph.router, prefix="/graph", tags=["graph"])
    app.include_router(communities.router, prefix="/communities", tags=["communities"])
    app.include_router(routes.router, prefix="/routes", tags=["routes"])
    app.include_router(stops.router, prefix="/stops", tags=["stops"])
    app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])

    return app
