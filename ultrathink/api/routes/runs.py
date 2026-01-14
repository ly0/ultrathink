"""Run and streaming endpoints.

These endpoints provide SSE streaming for agent execution,
matching the LangGraph SDK run API.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from ultrathink.api.models.run import Command, RunConfig, StreamInput
from ultrathink.api.services.stream_manager import get_stream_manager

router = APIRouter()


@router.post("/stream")
async def stream_run(
    body: StreamInput,
    thread_id: Optional[str] = Query(default=None),
    assistant_id: str = Query(default="default"),
):
    """Stream a run execution.

    This endpoint provides Server-Sent Events (SSE) for streaming
    agent execution results.

    Args:
        body: Stream input with messages, config, and commands.
        thread_id: Optional thread ID. Creates new thread if not provided.
        assistant_id: The assistant to use.

    Returns:
        SSE stream of execution events.
    """
    stream_manager = get_stream_manager()

    async def event_generator():
        async for event in stream_manager.start_stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input_data=body.input,
            config=body.config or RunConfig(),
            checkpoint=body.checkpoint.model_dump() if body.checkpoint else None,
            interrupt_before=body.interrupt_before,
            interrupt_after=body.interrupt_after,
            command=body.command,
        ):
            yield {
                "event": event.event.value,
                "data": event.model_dump_json(),
            }

    return EventSourceResponse(event_generator())


@router.post("/wait")
async def wait_run(
    body: StreamInput,
    thread_id: Optional[str] = Query(default=None),
    assistant_id: str = Query(default="default"),
) -> Dict[str, Any]:
    """Execute a run and wait for completion.

    This is a non-streaming version that returns the final result.

    Args:
        body: Stream input with messages, config, and commands.
        thread_id: Optional thread ID.
        assistant_id: The assistant to use.

    Returns:
        Final execution result.
    """
    stream_manager = get_stream_manager()

    result = {
        "thread_id": thread_id,
        "values": {},
        "interrupt": None,
        "error": None,
    }

    async for event in stream_manager.start_stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input_data=body.input,
        config=body.config or RunConfig(),
        checkpoint=body.checkpoint.model_dump() if body.checkpoint else None,
        interrupt_before=body.interrupt_before,
        interrupt_after=body.interrupt_after,
        command=body.command,
    ):
        if event.event.value == "thread_created":
            result["thread_id"] = event.data.get("thread_id")
        elif event.event.value == "values":
            result["values"] = event.data
        elif event.event.value == "interrupt":
            result["interrupt"] = event.data
        elif event.event.value == "error":
            result["error"] = event.data

    return result
