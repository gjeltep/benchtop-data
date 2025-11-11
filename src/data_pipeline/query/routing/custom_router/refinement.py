"""Refinement context utilities for query enhancement."""

from typing import Optional
from llama_index.core.workflow import Context
from ...agents.config import AgenticConfig
from ...agents.reflection_agent import ReflectionResult
from .context_keys import ContextKeys
from ....logging import get_logger

logger = get_logger(__name__)


def _build_refinement_context(
    query: str,
    missing_info: list[str],
    improvements: list[str],
    previous_response: Optional[str],
    config: AgenticConfig,
) -> str:
    """
    Build enhanced query with refinement context.

    Simple function approach - no builder pattern needed for MVP.

    Args:
        query: Original query
        missing_info: Missing information from reflection
        improvements: Suggested improvements
        previous_response: Previous response (if available)
        config: Configuration

    Returns:
        Enhanced query string, or original query if no enhancements
    """
    enhancements = []

    # Add missing info (limit to first 2 items)
    if missing_info:
        info_str = ', '.join(missing_info[:2])
        enhancements.append(f"Missing: {info_str}")

    # Add improvements (first item only)
    if improvements:
        enhancements.append(f"Improve: {improvements[0]}")

    # Add previous response if short enough
    if previous_response and len(previous_response) < 200:
        truncated = previous_response[:config.refinement_context_length]
        enhancements.append(f"Previous: {truncated}")

    if not enhancements:
        return query

    context = '; '.join(enhancements)
    return f"{query} [Refinement context: {context}]"


async def enhance_query_with_refinement_context(
    ctx: Context,
    query: str,
    config: Optional[AgenticConfig] = None
) -> str:
    """
    Enhance a query with refinement context from workflow store.

    This is the single source of truth for refinement context enhancement,
    eliminating duplication across the codebase.

    Args:
        ctx: Workflow context
        query: Query to enhance
        config: Optional configuration

    Returns:
        Enhanced query if in refinement iteration, otherwise original query
    """
    config = config or AgenticConfig()

    refinement_iteration = await ctx.store.get(ContextKeys.REFINEMENT_ITERATION, default=0)
    if refinement_iteration == 0:
        return query

    # Get reflection result from previous iteration
    reflection_data = await ctx.store.get(ContextKeys.REFLECTION_FEEDBACK, default=None)
    if not reflection_data:
        return query

    try:
        reflection_result = ReflectionResult(**reflection_data)
    except Exception as e:
        logger.warning(f"Failed to parse reflection result: {e}")
        return query

    previous_response = await ctx.store.get(ContextKeys.PREVIOUS_RESPONSE, default=None)

    enhanced = _build_refinement_context(
        query=query,
        missing_info=reflection_result.missing_information,
        improvements=reflection_result.suggested_improvements,
        previous_response=previous_response,
        config=config,
    )

    if enhanced != query:
        logger.debug("Enhanced query with refinement context")

    return enhanced

