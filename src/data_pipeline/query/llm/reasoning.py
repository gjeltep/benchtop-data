"""
Callback handler for capturing and logging reasoning tokens from LLMs.

Reasoning-based LLMs (e.g., DeepSeek R1, Qwen2.5-Reasoning) output reasoning tokens
in the `thinking_delta` field when streaming via Ollama. This handler captures
these tokens and logs them for observability.
"""

import sys
from typing import Optional
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from ...logging_config import get_logger

logger = get_logger(__name__)


class ReasoningTokenHandler(BaseCallbackHandler):
    """
    Callback handler that captures and logs reasoning tokens from LLMs.

    This handler intercepts LLM calls throughout the query process, including:
    - RouterQueryEngine's routing decisions
    - SQL query generation
    - Vector query responses

    Reasoning-based models output thinking tokens via `thinking_delta` in additional_kwargs
    when using streaming. This handler captures these tokens and logs them for any
    model that supports this feature (e.g., via Ollama).
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize reasoning token handler.

        Args:
            verbose: If True, logs reasoning tokens to console in real-time
        """
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[]
        )
        self.verbose = verbose
        self.current_reasoning: str = ""
        self._buffer: str = ""  # Buffer for accumulating tokens before logging
        self._buffer_size: int = 200  # Log when buffer reaches this size
        self._reasoning_started = False

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[EventPayload] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs,
    ) -> str:
        """Called when an event starts."""
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[EventPayload] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs,
    ) -> None:
        """Called when an event ends."""
        # Note: Thinking tokens are captured via ThinkingOllamaWrapper during streaming,
        # not through callbacks, so this method is kept minimal for compatibility.
        pass

    def start_reasoning_log(self) -> None:
        """Starts the reasoning log, printing a header if verbose."""
        if not self._reasoning_started:
            self._reasoning_started = True
            # Reset for new reasoning session
            self.current_reasoning = ""
            self._buffer = ""
            if self.verbose:
                # Write header directly to stderr for cleaner output
                print("\n" + "=" * 70, file=sys.stderr)
                print("MODEL REASONING:", file=sys.stderr)
                print("=" * 70, file=sys.stderr)

    def append_thinking_token(self, token: str) -> None:
        """
        Append a thinking token to the buffer and log when buffer is full.

        Args:
            token: The thinking token to append
        """
        if not token:
            return

        # Add to accumulated reasoning
        self.current_reasoning += token
        self._buffer += token

        # Log buffer when it reaches a certain size or contains newlines
        if len(self._buffer) >= self._buffer_size or '\n' in self._buffer:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush the current buffer to stderr for real-time viewing."""
        if self._buffer and self.verbose:
            print(self._buffer, end="", file=sys.stderr, flush=True)
            self._buffer = ""

    def log_complete_reasoning(self) -> None:
        """
        Mark reasoning as complete and reset for next LLM call.
        Reasoning tokens are still available via get_reasoning() but not logged as blocks.
        """
        # Flush any remaining buffer
        self._flush_buffer()

        # Reset for next LLM call (reasoning still available via get_reasoning())
        self._reasoning_started = False

    def get_reasoning(self) -> str:
        """
        Get accumulated reasoning tokens.

        Returns:
            Complete reasoning text captured so far
        """
        return self.current_reasoning

    def reset(self) -> None:
        """Reset captured reasoning tokens."""
        self.current_reasoning = ""
        self._buffer = ""
        self._reasoning_started = False

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace (required by BaseCallbackHandler)."""
        pass

    def end_trace(self, trace_id: Optional[str] = None, trace_map: Optional[dict] = None) -> None:
        """End a trace (required by BaseCallbackHandler)."""
        pass

