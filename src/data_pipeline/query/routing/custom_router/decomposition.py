"""Decomposition utilities for sub-question execution."""

from typing import List
import asyncio
from llama_index.core.workflow import Context
from llama_index.core.base.response.schema import AsyncStreamingResponse
from ....logging import get_logger
from .refinement import enhance_query_with_refinement_context
from ...agents.config import AgenticConfig

logger = get_logger(__name__)


async def execute_sub_questions(
    sub_questions: List,
    original_query: str,
    query_engine_tools,
    selector_step_fn,
    coerce_to_response_fn,
    ctx: Context,
    reasoning_handler=None,
    config: AgenticConfig = None,
):
    """
    Execute all sub-questions in parallel using decomposition hints for routing.

    Uses requires_sql/requires_semantic hints to route directly without LLM selector.

    Args:
        sub_questions: List of sub-questions to execute
        original_query: Original user query
        query_engine_tools: Query engine tools for execution
        selector_step_fn: Function to call for LLM-based routing
        coerce_to_response_fn: Function to coerce responses to Response type
        ctx: Workflow context
        reasoning_handler: Optional reasoning token handler
        config: Agentic configuration

    Returns:
        Dict with sub_question_results, sub_questions, and original_query
    """
    config = config or AgenticConfig()
    logger.info(f"Executing {len(sub_questions)} sub-questions in parallel...")

    if not query_engine_tools:
        raise ValueError("query_engine_tools must be provided")

    query_engines = [tool.query_engine for tool in query_engine_tools]
    # Engine 0 = vector, Engine 1 = SQL

    async def execute_sub_question(sub_q, index):
        """Execute a single sub-question using hints for routing."""
        # Extract question and hints from Pydantic SubQuestion model
        question = sub_q.question
        requires_sql = sub_q.requires_sql
        requires_semantic = sub_q.requires_semantic

        # Enhance query with refinement context if in refinement iteration
        question = await enhance_query_with_refinement_context(ctx, question, config)

        # Route using hints (skip LLM selector when clear)
        if requires_sql and not requires_semantic:
            engine_index = 1
            logger.debug(f"Sub-Q {index + 1}: routing to SQL (hint-based)")
        elif requires_semantic and not requires_sql:
            engine_index = 0
            logger.debug(f"Sub-Q {index + 1}: routing to Vector (hint-based)")
        else:
            logger.debug(f"Sub-Q {index + 1}: using LLM selector for routing")
            selection_event = await selector_step_fn(ctx, question)
            engine_index = selection_event.selected_query_engines.selections[0].index

        # Execute and process response
        response = await query_engines[engine_index].aquery(question)
        if isinstance(response, AsyncStreamingResponse):
            async for _ in response.async_response_gen():
                pass
            if reasoning_handler:
                reasoning_handler.log_complete_reasoning()

        logger.debug(f"Sub-Q {index + 1} completed")
        return await coerce_to_response_fn(response)

    # Execute all sub-questions in parallel
    sub_question_results = await asyncio.gather(
        *[execute_sub_question(sq, i) for i, sq in enumerate(sub_questions)]
    )

    logger.info(f"Completed execution of {len(sub_question_results)} sub-questions")

    # Convert Pydantic SubQuestion models to dict format
    sub_questions_dict = [sq.model_dump() for sq in sub_questions]

    return {
        "sub_question_results": sub_question_results,
        "sub_questions": sub_questions_dict,
        "original_query": original_query,
    }
