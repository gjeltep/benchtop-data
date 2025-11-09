"""LLM factory and engine type definitions."""

from enum import Enum
from typing import Optional
from llama_index.llms.ollama import Ollama
from .wrapper import ThinkingOllamaWrapper
from .reasoning import ReasoningTokenHandler


class EngineType(Enum):
    """Enum for query engine types."""
    VECTOR = 0
    SQL = 1

    @property
    def name_lower(self) -> str:
        return self.name.lower()


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create(
        model_name: str,
        base_url: str,
        reasoning_handler: Optional[ReasoningTokenHandler],
        request_timeout: float,
        context_window: int,
        num_output: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ):
        """Create an LLM instance."""
        llm_config = {
            "model": model_name,
            "base_url": base_url,
            "request_timeout": request_timeout,
            "context_window": context_window,
            "num_output": num_output,
            "temperature": temperature,
            "thinking": reasoning_handler is not None,
            "additional_kwargs": {"num_ctx": context_window},
        }
        if system_prompt:
            llm_config["system_prompt"] = system_prompt

        if reasoning_handler:
            return ThinkingOllamaWrapper(reasoning_handler=reasoning_handler, **llm_config)
        return Ollama(**llm_config)

