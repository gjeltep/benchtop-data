"""
Configuration module for data pipeline.

Single source of truth for all default values.
Can be overridden via environment variables or function parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Centralized configuration for the data pipeline."""

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
        default_factory=lambda: int(os.getenv("DATA_PIPELINE_CONTEXT_WINDOW", "32768"))
    )

    temperature: float = field(
        default_factory=lambda: float(os.getenv("DATA_PIPELINE_TEMPERATURE", "0.2"))
    )

    num_output: int = field(
        default_factory=lambda: int(os.getenv("DATA_PIPELINE_NUM_OUTPUT", "1024"))
    )

    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("DATA_PIPELINE_REQUEST_TIMEOUT", "180.0"))
    )

    # Query Engine Parameters
    similarity_top_k: int = field(
        default_factory=lambda: int(os.getenv("DATA_PIPELINE_SIMILARITY_TOP_K", "5"))
    )

    chat_history_token_limit: int = field(
        default_factory=lambda: int(os.getenv("DATA_PIPELINE_CHAT_HISTORY_LIMIT", "3000"))
    )

    embed_batch_size: int = field(
        default_factory=lambda: int(os.getenv("DATA_PIPELINE_EMBED_BATCH_SIZE", "64"))
    )

    # Reasoning/Logging Parameters
    enable_reasoning_logs: bool = field(
        default_factory=lambda: os.getenv("DATA_PIPELINE_ENABLE_REASONING_LOGS", "false").lower() == "true"
    )

    # Database Configuration
    db_path: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_PIPELINE_DB_PATH")
    )

    chroma_path: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_PIPELINE_CHROMA_PATH")
    )


# Global singleton instance
config = Config()

