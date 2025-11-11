"""Wrapper for Ollama LLM that captures thinking tokens during streaming."""

import re
from typing import AsyncIterator, AsyncGenerator, List, ClassVar
from collections import deque
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import CompletionResponse, ChatResponse
from llama_index.core.llms import ChatMessage
from ...logging import get_logger

logger = get_logger(__name__)


class ThinkingOllamaWrapper(Ollama):
    """Wrapper around Ollama LLM that captures thinking tokens during streaming.
    Also checks for repetitive output and stops the generation if it detects a loop.
    """

    THINKING_TOKEN_KEYS: ClassVar[List[str]] = ["thinking_delta", "thinking", "reasoning", "think"]
    REPETITION_THRESHOLD: ClassVar[int] = 3  # Number of times a sentence can repeat before stopping
    RECENT_TEXT_WINDOW: ClassVar[int] = 500  # Characters to check for repetition

    def __init__(self, reasoning_handler=None, **kwargs):
        """Initialize wrapper."""
        super().__init__(**kwargs)
        object.__setattr__(self, "reasoning_handler", reasoning_handler)
        object.__setattr__(self, "_recent_text", deque(maxlen=1000))  # Track recent output

    def _check_repetition(self, text: str) -> bool:
        """
        Detect if output is stuck in a repetitive loop.

        Uses a simple heuristic: check if the same sentence appears multiple times
        in recent output, which indicates the model is looping.
        """
        if not text or len(text) < 100:
            return False

        # Get recent text window
        recent = "".join(self._recent_text)[-self.RECENT_TEXT_WINDOW :]
        if not recent or len(recent) < 100:
            return False

        # Split into meaningful sentences
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", recent) if len(s.strip()) > 20]

        if len(sentences) < 3:
            return False

        # Check if the last sentence repeats too many times (indicates loop)
        last_sentence = sentences[-1]
        repetition_count = recent.count(last_sentence)

        if repetition_count >= self.REPETITION_THRESHOLD:
            logger.warning(
                f"Detected repetitive output (sentence repeated {repetition_count} times)"
            )
            return True

        return False

    def _extract_chunk_text(self, chunk) -> str:
        """Extract text content from streaming chunk."""
        for attr in ["delta", "text", "content"]:
            if hasattr(chunk, attr):
                value = getattr(chunk, attr)
                if value:
                    return str(value)
        return ""

    def _extract_thinking_tokens(self, chunk):
        """Extract thinking tokens from a chunk and append to handler."""
        if (
            not self.reasoning_handler
            or not hasattr(chunk, "additional_kwargs")
            or not chunk.additional_kwargs
        ):
            return
        for key in self.THINKING_TOKEN_KEYS:
            if key in chunk.additional_kwargs:
                thinking_value = chunk.additional_kwargs[key]
                if thinking_value:
                    self.reasoning_handler.append_thinking_token(str(thinking_value))

    async def astream_complete(self, prompt: str, **kwargs) -> AsyncIterator[CompletionResponse]:
        """Stream completion and capture thinking tokens."""
        if self.reasoning_handler:
            self.reasoning_handler.start_reasoning_log()
        async for chunk in super().astream_complete(prompt, **kwargs):
            self._extract_thinking_tokens(chunk)
            yield chunk
        if self.reasoning_handler:
            self.reasoning_handler.log_complete_reasoning()

    def _wrap_stream_generator(self, stream: AsyncGenerator) -> AsyncGenerator:
        """Yield chunks while capturing thinking tokens and logging completion."""

        async def generator():
            try:
                accumulated_text = ""
                async for chunk in stream:
                    self._extract_thinking_tokens(chunk)

                    # Extract text from chunk
                    chunk_text = self._extract_chunk_text(chunk)
                    if chunk_text:
                        accumulated_text += chunk_text
                        self._recent_text.append(chunk_text)

                        # Check for repetition every 100 chars
                        if len(accumulated_text) % 100 == 0:
                            if self._check_repetition(accumulated_text):
                                logger.warning("Stopping due to repetitive output pattern")
                                break

                    yield chunk
            finally:
                # Reset repetition tracking
                self._recent_text.clear()
                if self.reasoning_handler:
                    self.reasoning_handler.log_complete_reasoning()

        return generator()

    async def astream_chat(self, messages: List[ChatMessage], **kwargs):
        """Stream chat completion and capture thinking tokens."""
        if self.reasoning_handler:
            self.reasoning_handler.start_reasoning_log()
        parent_stream = await super().astream_chat(messages, **kwargs)
        if not hasattr(parent_stream, "__aiter__"):
            raise TypeError("Unexpected return type from Ollama.astream_chat()")
        return self._wrap_stream_generator(parent_stream)

    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        """Synchronous version of stream_chat."""
        if self.reasoning_handler:
            self.reasoning_handler.start_reasoning_log()
        stream = super().stream_chat(messages, **kwargs)
        for chunk in stream:
            self._extract_thinking_tokens(chunk)
            yield chunk
        if self.reasoning_handler:
            self.reasoning_handler.log_complete_reasoning()

    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Async chat completion - intercepts to capture thinking tokens."""
        if self.reasoning_handler:
            self.reasoning_handler.start_reasoning_log()

        # Reset repetition tracking
        self._recent_text.clear()

        parent_stream = await super().astream_chat(messages, **kwargs)
        if not hasattr(parent_stream, "__aiter__"):
            raise TypeError("Unexpected return type from Ollama.astream_chat()")
        final_response = None
        accumulated_text = ""

        try:
            async for chunk in parent_stream:
                self._extract_thinking_tokens(chunk)

                # Extract text from chunk
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text:
                    accumulated_text += chunk_text
                    self._recent_text.append(chunk_text)

                    # Check for repetition every 100 chars
                    if len(accumulated_text) % 100 == 0:
                        if self._check_repetition(accumulated_text):
                            logger.warning("Stopping due to repetitive output pattern")
                            break

                final_response = chunk
        finally:
            # Reset repetition tracking
            self._recent_text.clear()
            if self.reasoning_handler:
                self.reasoning_handler.log_complete_reasoning()

        return final_response if final_response else await super().achat(messages, **kwargs)
