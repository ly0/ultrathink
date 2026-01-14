"""Thread management endpoints.

These endpoints match the LangGraph SDK thread API.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from ultrathink.api.models.thread import Thread, ThreadSearchParams, StateSnapshot
from ultrathink.api.models.run import RunConfig, StreamInput
from ultrathink.api.services.thread_store import get_thread_store
from ultrathink.api.services.stream_manager import get_stream_manager

router = APIRouter()


@router.post("/search")
async def search_threads(
    params: Optional[ThreadSearchParams] = None,
) -> List[Thread]:
    """Search for threads.

    Args:
        params: Search parameters.

    Returns:
        List of matching threads.
    """
    thread_store = get_thread_store()
    search_params = params or ThreadSearchParams()
    return await thread_store.search_threads(search_params)


@router.post("")
async def create_thread(
    metadata: Optional[Dict[str, Any]] = None,
) -> Thread:
    """Create a new thread.

    Args:
        metadata: Optional thread metadata.

    Returns:
        The created thread.
    """
    thread_store = get_thread_store()
    return await thread_store.create_thread(metadata=metadata)


@router.get("/{thread_id}")
async def get_thread(thread_id: str) -> Thread:
    """Get a thread by ID.

    Args:
        thread_id: The thread ID.

    Returns:
        The thread.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()
    thread = await thread_store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.patch("/{thread_id}")
async def update_thread(
    thread_id: str,
    updates: Dict[str, Any],
) -> Thread:
    """Update a thread.

    Args:
        thread_id: The thread ID.
        updates: Fields to update.

    Returns:
        The updated thread.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()
    thread = await thread_store.update_thread(thread_id, updates)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str) -> Dict[str, bool]:
    """Delete a thread.

    Args:
        thread_id: The thread ID.

    Returns:
        Deletion status.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()
    deleted = await thread_store.delete_thread(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"deleted": True}


@router.get("/{thread_id}/state")
async def get_thread_state(thread_id: str) -> Dict[str, Any]:
    """Get thread state values.

    Args:
        thread_id: The thread ID.

    Returns:
        Thread state values.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()
    thread = await thread_store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"values": thread.values}


@router.post("/{thread_id}/state")
async def update_thread_state(
    thread_id: str,
    body: Dict[str, Any],
) -> Thread:
    """Update thread state values.

    Args:
        thread_id: The thread ID.
        body: Request body with 'values' to update.

    Returns:
        The updated thread.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()
    values = body.get("values", {})
    thread = await thread_store.update_state(thread_id, values)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.put("/{thread_id}/state")
async def put_thread_state(
    thread_id: str,
    body: Dict[str, Any],
) -> Thread:
    """Update thread state values (PUT method).

    Same as POST but using PUT for LangGraph SDK compatibility.

    Args:
        thread_id: The thread ID.
        body: Request body with 'values' to update.

    Returns:
        The updated thread.

    Raises:
        HTTPException: If thread not found.
    """
    return await update_thread_state(thread_id, body)


@router.get("/{thread_id}/state/history")
async def get_state_history(
    thread_id: str,
    limit: int = Query(default=10, ge=1, le=100),
) -> List[StateSnapshot]:
    """Get thread state history.

    Args:
        thread_id: The thread ID.
        limit: Maximum number of snapshots to return.

    Returns:
        List of state snapshots.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()

    # Verify thread exists
    thread = await thread_store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    history = await thread_store.get_state_history(thread_id)
    return history[:limit]


@router.post("/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Get thread history (LangGraph SDK compatible).

    This endpoint matches the LangGraph SDK thread history API.
    Returns the state history for a thread.

    IMPORTANT: The LangGraph SDK uses this endpoint to load messages when
    selecting a thread. It expects the current thread values (messages) to
    be included as the first history entry.

    Args:
        thread_id: The thread ID.
        body: Optional request body with filter parameters.

    Returns:
        List of history entries with values and metadata.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()

    # Get thread (need current values for messages)
    thread = await thread_store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Parse filter parameters
    limit = 10
    if body:
        limit = body.get("limit", 10)

    result = []

    # IMPORTANT: Add current thread state as the first history entry
    # This is what the SDK needs to load messages when selecting a thread
    result.append({
        "values": thread.values,  # Contains messages, todos, files
        "next": [],
        "config": {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "current",
            }
        },
        "metadata": thread.metadata or {},
        "created_at": thread.updated_at.isoformat() if thread.updated_at else None,
        "parent_config": None,
    })

    # Also include any saved checkpoints as additional history
    checkpoints = await thread_store.get_state_history(thread_id)
    for snapshot in checkpoints[:limit - 1]:
        result.append({
            "values": snapshot.values,
            "next": snapshot.next or [],
            "config": {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": snapshot.checkpoint_id,
                }
            },
            "metadata": snapshot.metadata or {},
            "created_at": snapshot.created_at,
            "parent_config": snapshot.parent_config,
        })

    return result[:limit]


@router.post("/{thread_id}/state/checkpoint")
async def create_checkpoint(thread_id: str) -> StateSnapshot:
    """Create a checkpoint of the current thread state.

    Args:
        thread_id: The thread ID.

    Returns:
        The created checkpoint.

    Raises:
        HTTPException: If thread not found.
    """
    thread_store = get_thread_store()
    thread = await thread_store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    return await thread_store.save_checkpoint(thread)


@router.get("/{thread_id}/state/checkpoint/{checkpoint_id}")
async def get_checkpoint(
    thread_id: str,
    checkpoint_id: str,
) -> StateSnapshot:
    """Get a specific checkpoint.

    Args:
        thread_id: The thread ID.
        checkpoint_id: The checkpoint ID.

    Returns:
        The checkpoint.

    Raises:
        HTTPException: If not found.
    """
    thread_store = get_thread_store()
    checkpoint = await thread_store.get_checkpoint(thread_id, checkpoint_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return checkpoint


@router.post("/{thread_id}/runs/stream")
async def stream_thread_run(
    thread_id: str,
    body: StreamInput,
    assistant_id: str = Query(default="default"),
):
    """Stream a run execution on a specific thread.

    This endpoint provides Server-Sent Events (SSE) for streaming
    agent execution results. This matches the LangGraph SDK API format.

    Args:
        thread_id: The thread ID to run on.
        body: Stream input with messages, config, and commands.
        assistant_id: The assistant to use.

    Returns:
        SSE stream of execution events.
    """
    stream_manager = get_stream_manager()

    async def event_generator():
        import json
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
            # Format data according to LangGraph SDK expectations
            # The SDK expects just the data payload, not wrapped in StreamEvent
            data_payload = event.data if event.data is not None else {}
            yield {
                "event": event.event.value,
                "data": json.dumps(data_payload),
            }

    return EventSourceResponse(event_generator())


@router.post("/{thread_id}/runs/wait")
async def wait_thread_run(
    thread_id: str,
    body: StreamInput,
    assistant_id: str = Query(default="default"),
) -> Dict[str, Any]:
    """Execute a run on a specific thread and wait for completion.

    This is a non-streaming version that returns the final result.

    Args:
        thread_id: The thread ID to run on.
        body: Stream input with messages, config, and commands.
        assistant_id: The assistant to use.

    Returns:
        Final execution result.
    """
    stream_manager = get_stream_manager()

    result: Dict[str, Any] = {
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
        if event.event.value == "values":
            result["values"] = event.data
        elif event.event.value == "interrupt":
            result["interrupt"] = event.data
        elif event.event.value == "error":
            result["error"] = event.data

    return result
