"""Health check endpoint."""

from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Health status information.
    """
    return {
        "status": "healthy",
        "service": "ultrathink",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/info")
async def info():
    """Get API information.

    Returns:
        API version and capabilities.
    """
    return {
        "name": "Ultrathink API",
        "version": "0.1.0",
        "langgraph_compatible": True,
        "features": [
            "threads",
            "assistants",
            "streaming",
            "tool_approval",
        ],
    }
