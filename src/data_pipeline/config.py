"""
Unified configuration for the data pipeline.

Struct-based configuration using dataclasses. Environment variables are read
at config creation time, not accessed globally throughout the codebase.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_env_bool(key: str, default: str = "false") -> bool:
    """Helper to parse boolean environment variables."""
    return os.getenv(key, default).lower() == "true"


def _get_env_int(key: str, default: str) -> int:
    """Helper to parse integer environment variables."""
    return int(os.getenv(key, default))


def _get_env_float(key: str, default: str) -> float:
    """Helper to parse float environment variables."""
    return float(os.getenv(key, default))


@dataclass(frozen=True)
class Config:
    """
    Unified configuration for the data pipeline.

    All configuration values in one place. Environment variables are read
    at initialization time. Pass config instances explicitly rather than
    using globals.
    """

    # Model Configuration
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "DATA_PIPELINE_LLM_MODEL",
            "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
        )
    )

    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "DATA_PIPELINE_EMBED_MODEL",
            "nomic-embed-text"
        )
    )

    ollama_url: str = field(
        default_factory=lambda: os.getenv(
            "DATA_PIPELINE_OLLAMA_URL",
            "http://localhost:11434"
        )
    )

    # LLM Parameters
    context_window: int = field(
        default_factory=lambda: _get_env_int("DATA_PIPELINE_CONTEXT_WINDOW", "32768")
    )

    temperature: float = field(
        default_factory=lambda: _get_env_float("DATA_PIPELINE_TEMPERATURE", "0.2")
    )

    num_output: int = field(
        default_factory=lambda: _get_env_int("DATA_PIPELINE_NUM_OUTPUT", "1024")
    )

    request_timeout: float = field(
        default_factory=lambda: _get_env_float("DATA_PIPELINE_REQUEST_TIMEOUT", "180.0")
    )

    # Query Engine Parameters
    similarity_top_k: int = field(
        default_factory=lambda: _get_env_int("DATA_PIPELINE_SIMILARITY_TOP_K", "5")
    )

    chat_history_token_limit: int = field(
        default_factory=lambda: _get_env_int("DATA_PIPELINE_CHAT_HISTORY_LIMIT", "3000")
    )

    embed_batch_size: int = field(
        default_factory=lambda: _get_env_int("DATA_PIPELINE_EMBED_BATCH_SIZE", "64")
    )

    # Reasoning/Logging Parameters
    enable_reasoning_logs: bool = field(
        default_factory=lambda: _get_env_bool("DATA_PIPELINE_ENABLE_REASONING_LOGS", "false")
    )

    # Agent Configuration
    use_react_agent: bool = field(
        default_factory=lambda: _get_env_bool("USE_REACT_AGENT", "false")
    )

    # Database Configuration
    db_path: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_PIPELINE_DB_PATH")
    )

    chroma_path: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_PIPELINE_CHROMA_PATH")
    )

    def with_overrides(
        self,
        db_path: Optional[str] = None,
        chroma_path: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        context_window: Optional[int] = None,
        temperature: Optional[float] = None,
        similarity_top_k: Optional[int] = None,
        embed_batch_size: Optional[int] = None,
        enable_reasoning_logs: Optional[bool] = None,
        use_react_agent: Optional[bool] = None,
        embed_model: Optional[str] = None,
        num_output: Optional[int] = None,
        request_timeout: Optional[float] = None,
        chat_history_token_limit: Optional[int] = None,
    ) -> "Config":
        """
        Create a new config with overrides.

        Returns a new Config instance with specified values overridden.
        Useful for creating configs with specific overrides while keeping defaults.
        """
        return Config(
            llm_model=ollama_model or self.llm_model,
            embed_model=embed_model or self.embed_model,
            ollama_url=ollama_base_url or self.ollama_url,
            context_window=context_window if context_window is not None else self.context_window,
            temperature=temperature if temperature is not None else self.temperature,
            num_output=num_output if num_output is not None else self.num_output,
            request_timeout=request_timeout if request_timeout is not None else self.request_timeout,
            similarity_top_k=similarity_top_k if similarity_top_k is not None else self.similarity_top_k,
            chat_history_token_limit=chat_history_token_limit if chat_history_token_limit is not None else self.chat_history_token_limit,
            embed_batch_size=embed_batch_size if embed_batch_size is not None else self.embed_batch_size,
            enable_reasoning_logs=enable_reasoning_logs if enable_reasoning_logs is not None else self.enable_reasoning_logs,
            use_react_agent=use_react_agent if use_react_agent is not None else self.use_react_agent,
            db_path=db_path if db_path is not None else self.db_path,
            chroma_path=chroma_path if chroma_path is not None else self.chroma_path,
        )


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Factory function to create a Config instance. This is the preferred
    way to get config - explicit and testable.
    """
    return Config()
