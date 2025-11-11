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
    """Wrapper around Ollama LLM that captures thinking tokens during streaming."""

    THINKING_TOKEN_KEYS: ClassVar[List[str]] = ["thinking_delta", "thinking", "reasoning", "think"]
    # TODO: Better way to detect this weird looping behavior in reasoning models?
    REPETITION_THRESHOLD: ClassVar[int] = 3  # Number of times a phrase can repeat before stopping
    REPETITION_WINDOW: ClassVar[int] = 200  # Characters to check for repetition

    def __init__(self, reasoning_handler=None, **kwargs):
        """Initialize wrapper."""
        super().__init__(**kwargs)
        object.__setattr__(self, 'reasoning_handler', reasoning_handler)
        object.__setattr__(self, '_recent_text', deque(maxlen=1000))  # Track recent output
        object.__setattr__(self, '_repetition_count', 0)
        object.__setattr__(self, '_last_phrase', "")

    def _check_repetition(self, text: str) -> bool:
        """Check if text contains repetitive patterns and should be stopped."""
        if not text or len(text) < 100:
            return False

        # Get recent text window (last 500 chars)
        recent = "".join(self._recent_text)[-500:]
        if not recent:
            return False

        # Split into sentences/lines
        sentences = re.split(r'[.!?\n]+', recent)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]  # Only meaningful sentences

        if len(sentences) < 3:
            return False

        # Check if last few sentences are repeating
        # Look at last 3 sentences and see if they're similar
        last_sentences = sentences[-3:]

        # Check if all last sentences are very similar (repetition)
        if len(set(last_sentences)) == 1:
            logger.warning("Detected exact sentence repetition, stopping generation")
            return True

        # Check for phrase repetition (like "But note:" appearing many times)
        common_phrases = ["But note:", "However,", "But note", "However"]
        for phrase in common_phrases:
            count = recent.count(phrase)
            if count >= self.REPETITION_THRESHOLD:
                logger.warning(f"Detected repetitive phrase '{phrase}' ({count} times), stopping generation")
                return True

        # Check if last sentence appears multiple times in recent text
        if len(last_sentences) > 0:
            last_sentence = last_sentences[-1]
            if len(last_sentence) > 20:  # Only check meaningful sentences
                count = recent.count(last_sentence)
                if count >= self.REPETITION_THRESHOLD:
                    logger.warning(f"Detected repetitive sentence (repeated {count} times), stopping generation")
                    return True

        return False

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
                accumulated_text = ""
                async for chunk in stream:
                    self._extract_thinking_tokens(chunk)

                    # Extract text from chunk
                    chunk_text = ""
                    if hasattr(chunk, "delta") and chunk.delta:
                        chunk_text = chunk.delta
                    elif hasattr(chunk, "text") and chunk.text:
                        chunk_text = chunk.text
                    elif hasattr(chunk, "content") and chunk.content:
                        chunk_text = chunk.content

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
                self._repetition_count = 0
                self._last_phrase = ""
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
        self._repetition_count = 0
        self._last_phrase = ""

        parent_stream = await super().astream_chat(messages, **kwargs)
        if not hasattr(parent_stream, "__aiter__"):
            raise TypeError("Unexpected return type from Ollama.astream_chat()")
        final_response = None
        accumulated_text = ""

        try:
            async for chunk in parent_stream:
                self._extract_thinking_tokens(chunk)

                # Extract text from chunk
                chunk_text = ""
                if hasattr(chunk, "delta") and chunk.delta:
                    chunk_text = chunk.delta
                elif hasattr(chunk, "text") and chunk.text:
                    chunk_text = chunk.text
                elif hasattr(chunk, "content") and chunk.content:
                    chunk_text = chunk.content

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
            self._repetition_count = 0
            self._last_phrase = ""
            if self.reasoning_handler:
                self.reasoning_handler.log_complete_reasoning()

        return final_response if final_response else await super().achat(messages, **kwargs)
