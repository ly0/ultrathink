"""Thread persistence service.

Stores threads as JSON files in ~/.ultrathink/threads/.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiofiles

from ultrathink.api.models.thread import Thread, ThreadSearchParams, StateSnapshot, ThreadStatus


class ThreadStore:
    """File-based thread storage.

    Threads are stored as JSON files in the base directory.
    Each thread has its own file: {thread_id}.json
    Checkpoints are stored in: {thread_id}/checkpoints/{checkpoint_id}.json
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the thread store.

        Args:
            base_path: Base directory for thread storage.
                      Defaults to ~/.ultrathink/threads/
        """
        self.base_path = base_path or Path.home() / ".ultrathink" / "threads"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_thread_path(self, thread_id: str) -> Path:
        """Get the file path for a thread."""
        return self.base_path / f"{thread_id}.json"

    def _get_checkpoint_dir(self, thread_id: str) -> Path:
        """Get the checkpoint directory for a thread."""
        return self.base_path / thread_id / "checkpoints"

    async def create_thread(
        self,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Thread:
        """Create a new thread.

        Args:
            thread_id: Optional ID for the thread. Generated if not provided.
            metadata: Optional metadata for the thread.

        Returns:
            The created thread.
        """
        if thread_id is None:
            thread_id = str(uuid4())

        now = datetime.utcnow()
        thread = Thread(
            thread_id=thread_id,
            created_at=now,
            updated_at=now,
            status="idle",
            values={"messages": [], "todos": [], "files": {}},
            metadata=metadata or {},
        )

        await self._save_thread(thread)
        return thread

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID.

        Args:
            thread_id: The thread ID.

        Returns:
            The thread, or None if not found.
        """
        path = self._get_thread_path(thread_id)
        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, "r") as f:
                data = await f.read()
                return Thread.model_validate_json(data)
        except (json.JSONDecodeError, ValueError):
            return None

    async def update_thread(
        self,
        thread_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Thread]:
        """Update a thread.

        Args:
            thread_id: The thread ID.
            updates: Fields to update (status, metadata, config).

        Returns:
            The updated thread, or None if not found.
        """
        thread = await self.get_thread(thread_id)
        if thread is None:
            return None

        for key, value in updates.items():
            if hasattr(thread, key):
                setattr(thread, key, value)

        thread.updated_at = datetime.utcnow()
        await self._save_thread(thread)
        return thread

    async def update_state(
        self,
        thread_id: str,
        values: Dict[str, Any],
    ) -> Optional[Thread]:
        """Update thread state values.

        Args:
            thread_id: The thread ID.
            values: State values to update.

        Returns:
            The updated thread, or None if not found.
        """
        thread = await self.get_thread(thread_id)
        if thread is None:
            return None

        thread.update_state(values)
        await self._save_thread(thread)
        return thread

    async def search_threads(
        self,
        params: ThreadSearchParams,
    ) -> List[Thread]:
        """Search for threads.

        Args:
            params: Search parameters.

        Returns:
            List of matching threads.
        """
        threads: List[Thread] = []

        # Load all threads
        for path in self.base_path.glob("*.json"):
            try:
                async with aiofiles.open(path, "r") as f:
                    data = await f.read()
                    thread = Thread.model_validate_json(data)
                    threads.append(thread)
            except (json.JSONDecodeError, ValueError):
                continue

        # Filter by status
        if params.status:
            threads = [t for t in threads if t.status == params.status]

        # Filter by metadata
        if params.metadata:
            def matches_metadata(thread: Thread) -> bool:
                for key, value in params.metadata.items():  # type: ignore
                    if thread.metadata.get(key) != value:
                        return False
                return True

            threads = [t for t in threads if matches_metadata(t)]

        # Sort
        reverse = params.sort_order == "desc"
        if params.sort_by == "updated_at":
            threads.sort(key=lambda t: t.updated_at, reverse=reverse)
        elif params.sort_by == "created_at":
            threads.sort(key=lambda t: t.created_at, reverse=reverse)

        # Paginate
        start = params.offset
        end = start + params.limit
        return threads[start:end]

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its checkpoints.

        Args:
            thread_id: The thread ID.

        Returns:
            True if deleted, False if not found.
        """
        path = self._get_thread_path(thread_id)
        if not path.exists():
            return False

        path.unlink()

        # Delete checkpoints
        checkpoint_dir = self._get_checkpoint_dir(thread_id)
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)

        return True

    async def save_checkpoint(self, thread: Thread) -> StateSnapshot:
        """Save a checkpoint of the current thread state.

        Args:
            thread: The thread to checkpoint.

        Returns:
            The created checkpoint.
        """
        checkpoint_id = str(uuid4())
        snapshot = StateSnapshot(
            checkpoint_id=checkpoint_id,
            thread_id=thread.thread_id,
            timestamp=datetime.utcnow(),
            values=thread.values.copy(),
        )

        checkpoint_dir = self._get_checkpoint_dir(thread.thread_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{checkpoint_id}.json"
        async with aiofiles.open(checkpoint_path, "w") as f:
            await f.write(snapshot.model_dump_json(indent=2))

        return snapshot

    async def get_state_history(self, thread_id: str) -> List[StateSnapshot]:
        """Get the state history for a thread.

        Args:
            thread_id: The thread ID.

        Returns:
            List of state snapshots, ordered by timestamp descending.
        """
        checkpoint_dir = self._get_checkpoint_dir(thread_id)
        if not checkpoint_dir.exists():
            return []

        snapshots: List[StateSnapshot] = []
        for path in checkpoint_dir.glob("*.json"):
            try:
                async with aiofiles.open(path, "r") as f:
                    data = await f.read()
                    snapshot = StateSnapshot.model_validate_json(data)
                    snapshots.append(snapshot)
            except (json.JSONDecodeError, ValueError):
                continue

        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        return snapshots

    async def get_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> Optional[StateSnapshot]:
        """Get a specific checkpoint.

        Args:
            thread_id: The thread ID.
            checkpoint_id: The checkpoint ID.

        Returns:
            The checkpoint, or None if not found.
        """
        checkpoint_path = self._get_checkpoint_dir(thread_id) / f"{checkpoint_id}.json"
        if not checkpoint_path.exists():
            return None

        try:
            async with aiofiles.open(checkpoint_path, "r") as f:
                data = await f.read()
                return StateSnapshot.model_validate_json(data)
        except (json.JSONDecodeError, ValueError):
            return None

    async def _save_thread(self, thread: Thread) -> None:
        """Save a thread to disk."""
        path = self._get_thread_path(thread.thread_id)
        async with aiofiles.open(path, "w") as f:
            await f.write(thread.model_dump_json(indent=2))


# Global instance
_thread_store: Optional[ThreadStore] = None


def get_thread_store() -> ThreadStore:
    """Get the global thread store instance."""
    global _thread_store
    if _thread_store is None:
        _thread_store = ThreadStore()
    return _thread_store
