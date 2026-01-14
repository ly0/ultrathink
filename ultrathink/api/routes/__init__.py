"""API routes for ultrathink web API."""

from ultrathink.api.routes.health import router as health_router
from ultrathink.api.routes.threads import router as threads_router
from ultrathink.api.routes.assistants import router as assistants_router
from ultrathink.api.routes.runs import router as runs_router

__all__ = ["health_router", "threads_router", "assistants_router", "runs_router"]
