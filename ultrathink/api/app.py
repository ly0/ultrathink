"""FastAPI application for ultrathink web API.

Provides a LangGraph-compatible HTTP API and serves the frontend.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultrathink.api.routes import (
    health_router,
    threads_router,
    assistants_router,
    runs_router,
)


def create_app(
    serve_frontend: bool = True,
    frontend_path: Optional[Path] = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        serve_frontend: Whether to serve the static frontend.
        frontend_path: Path to the frontend build directory.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(
        title="Ultrathink API",
        description="LangGraph-compatible API for the Ultrathink AI coding assistant",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Next.js dev server
            "http://localhost:8000",  # Ultrathink server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(threads_router, prefix="/api/threads", tags=["threads"])
    app.include_router(assistants_router, prefix="/api/assistants", tags=["assistants"])
    app.include_router(runs_router, prefix="/api/runs", tags=["runs"])

    # Serve frontend if enabled
    if serve_frontend:
        if frontend_path is None:
            # Default to web/out in the package directory
            frontend_path = Path(__file__).parent.parent / "web" / "out"

        if frontend_path.exists():
            # Mount static files at root (must be last)
            app.mount(
                "/",
                StaticFiles(directory=frontend_path, html=True),
                name="frontend",
            )

    return app


# Default app instance for uvicorn
app = create_app()
