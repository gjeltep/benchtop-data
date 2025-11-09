"""Wrapper for Ollama LLM that captures thinking tokens during streaming."""

import sys
from typing import AsyncIterator, AsyncGenerator, List, ClassVar
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import CompletionResponse, ChatResponse
from llama_index.core.llms import ChatMessage
from ...logging_config import get_logger

logger = get_logger(__name__)


class ThinkingOllamaWrapper(Ollama):
    """Wrapper around Ollama LLM that captures thinking tokens during streaming."""

    THINKING_TOKEN_KEYS: ClassVar[List[str]] = ["thinking_delta", "thinking", "reasoning", "think"]

    def __init__(self, reasoning_handler=None, **kwargs):
        """Initialize wrapper."""
        super().__init__(**kwargs)
        object.__setattr__(self, 'reasoning_handler', reasoning_handler)

    def _extract_thinking_tokens(self, chunk):
        """Extract thinking tokens from a chunk and append to handler."""
        if not self.reasoning_handler or not hasattr(chunk, "additional_kwargs") or not chunk.additional_kwargs:
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
                async for chunk in stream:
                    self._extract_thinking_tokens(chunk)
                    yield chunk
            finally:
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
        parent_stream = await super().astream_chat(messages, **kwargs)
        if not hasattr(parent_stream, "__aiter__"):
            raise TypeError("Unexpected return type from Ollama.astream_chat()")
        final_response = None
        async for chunk in parent_stream:
            self._extract_thinking_tokens(chunk)
            final_response = chunk
        if self.reasoning_handler:
            self.reasoning_handler.log_complete_reasoning()
        return final_response if final_response else await super().achat(messages, **kwargs)
