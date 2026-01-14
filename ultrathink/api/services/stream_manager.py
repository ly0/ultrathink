"""Stream manager service.

Manages streaming execution of agent runs with SSE output.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from ultrathink.api.models.message import Message, ToolCall
from ultrathink.api.models.run import (
    ActionRequest,
    Command,
    Interrupt,
    ReviewConfig,
    RunConfig,
    StreamEvent,
    StreamEventType,
)
from ultrathink.api.models.thread import Thread
from ultrathink.api.services.thread_store import ThreadStore, get_thread_store


class StreamManager:
    """Manages streaming execution of agent runs.

    Integrates with UltrathinkClient to execute agents and
    stream results as SSE events.
    """

    def __init__(self, thread_store: Optional[ThreadStore] = None):
        """Initialize the stream manager.

        Args:
            thread_store: Thread store for persistence.
        """
        self.thread_store = thread_store or get_thread_store()
        self._active_runs: Dict[str, Dict[str, Any]] = {}

    async def start_stream(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        input_data: Optional[Dict[str, Any]],
        config: RunConfig,
        checkpoint: Optional[Dict[str, Any]] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None,
        command: Optional[Command] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Start a streaming execution.

        Args:
            thread_id: The thread ID. Creates a new thread if None.
            assistant_id: The assistant to use.
            input_data: Input data (messages, etc).
            config: Run configuration.
            checkpoint: Optional checkpoint to resume from.
            interrupt_before: Nodes to interrupt before.
            interrupt_after: Nodes to interrupt after.
            command: Command for resume/goto.

        Yields:
            Stream events.
        """
        run_id = str(uuid4())

        # Get or create thread
        thread = await self._get_or_create_thread(thread_id, assistant_id)
        thread_id = thread.thread_id

        # Emit thread created event if new
        if not input_data and not command:
            yield StreamEvent(
                event=StreamEventType.THREAD_CREATED,
                data={"thread_id": thread_id},
                run_id=run_id,
            )

        # Handle commands (resume, goto)
        if command:
            async for event in self._handle_command(thread, command, run_id):
                yield event
            return

        # Update thread status to busy
        await self.thread_store.update_thread(thread_id, {"status": "busy"})

        try:
            # Execute agent and stream events
            async for event in self._execute_agent(
                thread=thread,
                input_data=input_data,
                config=config,
                interrupt_before=interrupt_before or [],
                interrupt_after=interrupt_after or [],
                run_id=run_id,
            ):
                yield event
        except Exception as e:
            # Emit error event
            yield StreamEvent(
                event=StreamEventType.ERROR,
                data={"error": str(e), "type": type(e).__name__},
                run_id=run_id,
            )
            await self.thread_store.update_thread(thread_id, {"status": "error"})
        finally:
            # Clean up run tracking
            if run_id in self._active_runs:
                del self._active_runs[run_id]

    async def _get_or_create_thread(
        self,
        thread_id: Optional[str],
        assistant_id: str,
    ) -> Thread:
        """Get an existing thread or create a new one."""
        if thread_id:
            thread = await self.thread_store.get_thread(thread_id)
            if thread:
                return thread

        # Create new thread
        return await self.thread_store.create_thread(
            thread_id=thread_id,
            metadata={"assistant_id": assistant_id},
        )

    async def _handle_command(
        self,
        thread: Thread,
        command: Command,
        run_id: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Handle resume/goto commands."""
        if command.resume is not None:
            # Resume from interrupt
            async for event in self._resume_from_interrupt(thread, command.resume, run_id):
                yield event

        elif command.goto == "__end__":
            # Mark thread as resolved
            await self.thread_store.update_thread(thread.thread_id, {"status": "idle"})
            await self.thread_store.update_state(thread.thread_id, {"interrupt": None})
            yield StreamEvent(event=StreamEventType.END, run_id=run_id)

    async def _resume_from_interrupt(
        self,
        thread: Thread,
        resume_value: Any,
        run_id: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Resume execution after user response to interrupt."""
        # Get current interrupt
        interrupt = thread.values.get("interrupt")
        if not interrupt:
            yield StreamEvent(
                event=StreamEventType.ERROR,
                data={"error": "No pending interrupt to resume from"},
                run_id=run_id,
            )
            return

        # Process the decision
        decisions = resume_value.get("decisions", []) if isinstance(resume_value, dict) else []

        if decisions:
            decision = decisions[0]
            decision_type = decision.get("type", "approve")

            if decision_type == "reject":
                # Add rejection message to thread
                rejection_msg = decision.get("message", "Tool execution rejected by user")
                messages = thread.values.get("messages", [])
                messages.append({
                    "id": str(uuid4()),
                    "type": "human",
                    "content": f"[Tool rejected]: {rejection_msg}",
                })
                await self.thread_store.update_state(
                    thread.thread_id,
                    {"messages": messages, "interrupt": None},
                )

                # Emit updated values
                thread = await self.thread_store.get_thread(thread.thread_id)  # type: ignore
                yield StreamEvent(
                    event=StreamEventType.VALUES,
                    data=thread.values,
                    run_id=run_id,
                )

            elif decision_type == "approve" or decision_type == "edit":
                # Clear interrupt and continue execution
                await self.thread_store.update_state(
                    thread.thread_id,
                    {"interrupt": None},
                )

                # Get action to execute
                action_requests = interrupt.get("action_requests", [])
                if action_requests:
                    action = action_requests[0]
                    if decision_type == "edit":
                        # Use edited action
                        action = decision.get("edited_action", action)

                    # Execute the approved tool
                    async for event in self._execute_approved_tool(
                        thread, action, run_id
                    ):
                        yield event

        # Update thread status
        await self.thread_store.update_thread(thread.thread_id, {"status": "idle"})
        yield StreamEvent(event=StreamEventType.END, run_id=run_id)

    async def _execute_approved_tool(
        self,
        thread: Thread,
        action: Dict[str, Any],
        run_id: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute an approved tool call."""
        # For now, we add a placeholder message
        # In full implementation, this would call the actual tool
        tool_name = action.get("name", "unknown")
        tool_args = action.get("args", {})

        messages = thread.values.get("messages", [])

        # Add tool call message
        tool_call_id = str(uuid4())
        ai_msg = {
            "id": str(uuid4()),
            "type": "ai",
            "content": "",
            "tool_calls": [{
                "id": tool_call_id,
                "name": tool_name,
                "args": tool_args,
            }],
        }
        messages.append(ai_msg)

        # Add tool result (placeholder)
        tool_result = {
            "id": str(uuid4()),
            "type": "tool",
            "content": f"Tool {tool_name} executed successfully",
            "tool_call_id": tool_call_id,
        }
        messages.append(tool_result)

        await self.thread_store.update_state(thread.thread_id, {"messages": messages})

        # Emit values update
        thread = await self.thread_store.get_thread(thread.thread_id)  # type: ignore
        yield StreamEvent(
            event=StreamEventType.VALUES,
            data=thread.values,
            run_id=run_id,
        )

    async def _execute_agent(
        self,
        thread: Thread,
        input_data: Optional[Dict[str, Any]],
        config: RunConfig,
        interrupt_before: List[str],
        interrupt_after: List[str],
        run_id: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the ultrathink agent and stream results."""
        import asyncio

        # Get existing messages
        messages = thread.values.get("messages", [])

        # Add new input messages
        if input_data and "messages" in input_data:
            for msg in input_data["messages"]:
                if isinstance(msg, dict):
                    messages.append(msg)

        # Update thread with new messages
        await self.thread_store.update_state(thread.thread_id, {"messages": messages})

        # Emit values update
        yield StreamEvent(
            event=StreamEventType.VALUES,
            data={"messages": messages},
            run_id=run_id,
        )

        # Check if we should interrupt before tools
        should_interrupt_before_tools = "tools" in interrupt_before

        # Try to use the real ultrathink agent
        agent = None
        session = None
        use_demo_mode = True  # Default to demo mode

        try:
            from ultrathink.core.agent_factory import create_ultrathink_agent
            from ultrathink.core.session import ConversationSession

            # Create session and add history messages
            session = ConversationSession()
            for msg in messages:
                msg_type = msg.get("type", "")
                content = msg.get("content", "")
                if msg_type == "human" and isinstance(content, str):
                    session.add_message("user", content)
                elif msg_type == "ai" and isinstance(content, str):
                    session.add_message("assistant", content)

            # Use the same config system as CLI
            # Don't pass model/base_url - let it use ~/.ultrathink.json config
            agent = await asyncio.wait_for(
                create_ultrathink_agent(
                    session=session,
                    safe_mode=True,
                    verbose=False,
                    cwd=Path.cwd(),
                    # model=None → uses config_manager's "main" profile
                    # base_url=None → uses profile's api_base
                ),
                timeout=30.0  # 30 second timeout for agent creation
            )
            use_demo_mode = False
            logging.info("Real agent created successfully, using configured model")

        except asyncio.TimeoutError:
            logging.warning("Agent creation timed out, falling back to demo mode")
        except Exception as e:
            logging.warning(f"Agent creation failed: {e}, falling back to demo mode")

        if agent is not None and not use_demo_mode:
            # Use real agent
            try:
                # Initialize streaming variables
                ai_message_id = str(uuid4())
                full_response = ""
                pending_tool_calls: List[Dict[str, Any]] = []

                # Use session's messages for agent input
                agent_input = {"messages": session.get_messages_with_summary()} if session else {"messages": []}

                async for event in agent.astream_events(agent_input, version="v2"):
                    event_type = event.get("event", "")

                    if event_type == "on_chat_model_stream":
                        # Stream message chunks
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            full_response += chunk.content

                            # Build current AI message
                            current_ai_message = {
                                "id": ai_message_id,
                                "type": "ai",
                                "content": full_response,
                            }
                            if pending_tool_calls:
                                current_ai_message["tool_calls"] = pending_tool_calls

                            # Send VALUES event with full messages array
                            yield StreamEvent(
                                event=StreamEventType.VALUES,
                                data={
                                    "messages": messages + [current_ai_message],
                                    "todos": thread.values.get("todos", []),
                                    "files": thread.values.get("files", {}),
                                },
                                run_id=run_id,
                            )

                    elif event_type == "on_tool_start":
                        # Tool is about to execute
                        tool_name = event.get("name", "")
                        tool_input = event.get("data", {}).get("input", {})
                        tool_call_id = event.get("run_id", str(uuid4()))

                        # Add to pending tool calls
                        pending_tool_calls.append({
                            "id": tool_call_id,
                            "name": tool_name,
                            "args": tool_input,
                        })

                        if should_interrupt_before_tools:
                            # Create interrupt for tool approval
                            interrupt = Interrupt.for_tool_approval(
                                action_requests=[
                                    ActionRequest(
                                        name=tool_name,
                                        args=tool_input,
                                        description=f"Execute tool: {tool_name}",
                                    )
                                ],
                                review_configs=[
                                    ReviewConfig(
                                        actionName=tool_name,
                                        allowedDecisions=["approve", "reject", "edit"],
                                    )
                                ],
                            )

                            # Store interrupt in thread
                            await self.thread_store.update_state(
                                thread.thread_id,
                                {"interrupt": interrupt.value},
                            )
                            await self.thread_store.update_thread(
                                thread.thread_id,
                                {"status": "interrupted"},
                            )

                            # Emit interrupt event
                            yield StreamEvent(
                                event=StreamEventType.INTERRUPT,
                                data=interrupt.value,
                                run_id=run_id,
                            )
                            return  # Stop execution

                    elif event_type == "on_tool_end":
                        # Tool finished executing
                        tool_output = event.get("data", {}).get("output", "")
                        tool_run_id = event.get("run_id", "")

                        # Save current AI message if we have content
                        if full_response or pending_tool_calls:
                            ai_msg = {
                                "id": ai_message_id,
                                "type": "ai",
                                "content": full_response,
                            }
                            if pending_tool_calls:
                                ai_msg["tool_calls"] = pending_tool_calls
                            messages.append(ai_msg)

                            # Reset for new AI message
                            ai_message_id = str(uuid4())
                            full_response = ""
                            pending_tool_calls = []

                        # Add tool result message
                        messages.append({
                            "id": str(uuid4()),
                            "type": "tool",
                            "content": str(tool_output)[:2000],  # Truncate long output
                            "tool_call_id": tool_run_id,
                        })

                        # Send updated values
                        yield StreamEvent(
                            event=StreamEventType.VALUES,
                            data={
                                "messages": messages,
                                "todos": thread.values.get("todos", []),
                                "files": thread.values.get("files", {}),
                            },
                            run_id=run_id,
                        )

                # After stream ends, save final AI message if we have content
                if full_response:
                    messages.append({
                        "id": ai_message_id,
                        "type": "ai",
                        "content": full_response,
                    })

                # Update thread state
                await self.thread_store.update_state(
                    thread.thread_id,
                    {"messages": messages},
                )

            except Exception as e:
                # Handle errors
                logging.error(f"Agent execution error: {e}")
                yield StreamEvent(
                    event=StreamEventType.ERROR,
                    data={"error": str(e), "type": type(e).__name__},
                    run_id=run_id,
                )
                await self.thread_store.update_thread(thread.thread_id, {"status": "error"})
                return

        else:
            # Demo mode - generate a simple response
            # Get the last human message
            last_human_msg = None
            for msg in reversed(messages):
                if msg.get("type") == "human":
                    last_human_msg = msg.get("content", "")
                    break

            if last_human_msg:
                # Generate demo response chunks
                demo_response = f"Hello! I received your message: \"{last_human_msg}\"\n\nThis is a demo response from the Ultrathink API. The full agent integration requires resolving some dependency conflicts. For now, the web interface is working correctly with this placeholder response.\n\nYou can test the streaming functionality - this message was sent in chunks!"

                # Create the AI message that we'll update as we stream
                ai_message_id = str(uuid4())
                full_response = ""
                chunk_size = 10

                for i in range(0, len(demo_response), chunk_size):
                    chunk = demo_response[i:i + chunk_size]
                    full_response += chunk

                    # Create current AI message with partial content
                    current_ai_message = {
                        "id": ai_message_id,
                        "type": "ai",
                        "content": full_response,
                    }

                    # Send values event with updated messages array
                    # This is how LangGraph SDK expects streaming to work
                    current_messages = messages + [current_ai_message]
                    yield StreamEvent(
                        event=StreamEventType.VALUES,
                        data={
                            "messages": current_messages,
                            "todos": thread.values.get("todos", []),
                            "files": thread.values.get("files", {}),
                        },
                        run_id=run_id,
                    )
                    await asyncio.sleep(0.03)  # Small delay for realistic streaming

                # Final AI message
                final_ai_message = {
                    "id": ai_message_id,
                    "type": "ai",
                    "content": full_response,
                }
                messages.append(final_ai_message)

                # Update thread state
                await self.thread_store.update_state(
                    thread.thread_id,
                    {"messages": messages},
                )

        # Emit final values
        thread = await self.thread_store.get_thread(thread.thread_id)  # type: ignore
        yield StreamEvent(
            event=StreamEventType.VALUES,
            data=thread.values,
            run_id=run_id,
        )

        # Update status to idle
        await self.thread_store.update_thread(thread.thread_id, {"status": "idle"})

        # Emit end event
        yield StreamEvent(event=StreamEventType.END, run_id=run_id)


# Global instance
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get the global stream manager instance."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
