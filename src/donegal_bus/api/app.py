"""FastAPI application factory with lifespan and router includes."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from donegal_bus.api.dependencies import load_graphs
from donegal_bus.api.routers import analysis, communities, graph, routes, stops


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load graphs into memory at startup."""
    load_graphs()
    yield


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="Donegal Bus Routes API",
        description="Rural bus route planning for County Donegal.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(graph.router, prefix="/graph", tags=["graph"])
    app.include_router(communities.router, prefix="/communities", tags=["communities"])
    app.include_router(routes.router, prefix="/routes", tags=["routes"])
    app.include_router(stops.router, prefix="/stops", tags=["stops"])
    app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])

    return app
