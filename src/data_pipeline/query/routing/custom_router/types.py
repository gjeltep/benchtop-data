"""Type definitions for query routing workflow."""

from typing import TypedDict, List
from pydantic import BaseModel, Field


class SubQuestionDict(TypedDict, total=False):
    """Type-safe dictionary for sub-question data."""

    question: str
    requires_sql: bool
    requires_semantic: bool
    reasoning: str  # Optional


class QueryMetadata(TypedDict, total=False):
    """Type-safe metadata for query execution."""

    engines_used: List[str]
    has_vector_db: bool
    has_sql: bool
    source_node_count: int
    vector_source_count: int
    error: str  # Optional
