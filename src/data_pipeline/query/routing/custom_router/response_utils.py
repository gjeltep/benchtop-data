"""Utilities for extracting and processing Response objects."""

from llama_index.core.base.response.schema import Response
from ...llm.factory import EngineType


def extract_response_text(response: Response) -> str:
    """
    Extract text from a Response object (LlamaIndex's response synthesis)
    The response.response field should contain the properly formatted answer
    based on our configured synthesis prompts.

    Args:
        response: Response object from query engine

    Returns:
        Extracted text content
    """
    # Primary: Check response.response (synthesized answer)
    if hasattr(response, "response") and response.response:
        return str(response.response)

    # Fallback: Check response_text attribute
    if hasattr(response, "response_text"):
        return str(response.response_text)

    # Final fallback: string representation
    return str(response)


def get_engine_name(index: int) -> str:
    """
    Get engine name from index (lowercase).

    Args:
        index: Engine index (0 = Vector, 1 = SQL)

    Returns:
        Engine name as lowercase string
    """
    try:
        return EngineType(index).name_lower
    except (ValueError, AttributeError):
        return "unknown"


def get_engine_display_name(index: int) -> str:
    """
    Get human-readable engine name.

    Args:
        index: Engine index (0 = Vector, 1 = SQL)

    Returns:
        Engine name with proper capitalization
    """
    try:
        return EngineType(index).name
    except (ValueError, AttributeError):
        return "Unknown"
