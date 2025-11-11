"""Utilities for extracting and processing Response objects."""

from llama_index.core.base.response.schema import Response
from ...llm.factory import EngineType


# SQL error indicators - extracted for clarity and reusability
_SQL_ERROR_INDICATORS = [
    "contains an error",
    "SQL query contains an error",
    "invalid",
    "cannot be used",
    "Binder Error",
    "Error:",
    "syntax error",
    "cannot include",
]


def _has_sql_error(text: str) -> bool:
    """Check if text contains SQL error indicators."""
    text_lower = text.lower()
    return any(indicator.lower() in text_lower for indicator in _SQL_ERROR_INDICATORS)


def _is_valid_data_tuple(text: str) -> bool:
    """
    Check if text looks like valid data tuples (not errors).

    Simple heuristic: contains digits and tuple-like structures.
    """
    if not text:
        return False

    # Check for error markers
    if any(err in text for err in ["Error:", "Binder Error"]):
        return False

    # Check for data-like patterns (digits and tuple indicators)
    preview = text[:200]
    return any(c.isdigit() for c in preview) and ('(' in text or ',' in text)


def _extract_source_node_texts(source_nodes) -> str:
    """Extract text from source nodes, combining into a single string."""
    if not source_nodes:
        return ""

    node_texts = [
        node.text if hasattr(node, 'text') else
        node.get_content() if hasattr(node, 'get_content') else
        str(node)
        for node in source_nodes
    ]

    return "\n".join(node_texts) if node_texts else ""


def extract_response_text(response: Response) -> str:
    """
    Extract text from a Response object following LlamaIndex best practices.

    With NLSQLTableQueryEngine.response_synthesis_prompt configured, response.response
    should contain the properly synthesized answer. However, if the SQL query failed,
    response.response may contain an error explanation instead of data. In that case,
    we check source_nodes for raw data tuples.

    Args:
        response: Response object from query engine

    Returns:
        Extracted text content
    """
    # Primary: Check response.response (synthesized answer)
    if hasattr(response, 'response') and response.response:
        response_text = str(response.response)

        # If response looks like an error, try source_nodes for actual data
        if _has_sql_error(response_text) and response.source_nodes:
            source_text = _extract_source_node_texts(response.source_nodes)
            if source_text and _is_valid_data_tuple(source_text):
                return source_text

        # No error indicators - trust the synthesized response
        return response_text

    # Fallback: Check response_text attribute
    if hasattr(response, 'response_text'):
        return str(response.response_text)

    # Fallback: Check source_nodes for raw data
    if response.source_nodes:
        source_text = _extract_source_node_texts(response.source_nodes)
        if source_text and _is_valid_data_tuple(source_text):
            return source_text

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

