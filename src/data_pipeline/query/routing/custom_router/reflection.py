"""Reflection utilities for quality assessment and query refinement."""

from typing import Optional
from llama_index.core.workflow import Context, StartEvent
from llama_index.core.base.response.schema import Response
from ....logging import get_logger
from ...agents import ReflectionAgent
from ...agents.config import AgenticConfig
from .context_keys import ContextKeys
from .types import QueryMetadata, ReflectionFeedback
from ...agents.refinement_agent import RefinementAgent

logger = get_logger(__name__)


async def reflect_and_refine(
    reflection_agent: ReflectionAgent,
    refinement_agent: RefinementAgent,
    decompose_query_fn,
    ctx: Context,
    query: str,
    response: str,
    iteration: int = 0,
    response_obj: Optional[Response] = None,
    config: AgenticConfig = None,
):
    """
    Reflect on response and potentially refine query.

    Args:
        reflection_agent: Agent for reflection analysis
        refinement_agent: Agent for query refinement
        decompose_query_fn: Function to trigger query decomposition
        ctx: Workflow context
        query: Query being evaluated
        response: Response to evaluate
        iteration: Current refinement iteration
        response_obj: Optional response object
        config: Agentic configuration

    Returns:
        ReflectionCompleteEvent, QueryEngineSelectionEvent (for refinement), or StopEvent
    """
    from .router import ReflectionCompleteEvent

    config = config or AgenticConfig()
    logger.info(f"Evaluating response quality (iteration {iteration + 1})...")

    # Get metadata for reflection context
    engines_used = await ctx.store.get(ContextKeys.ENGINES_USED, default=[])
    metadata: QueryMetadata = {
        "engines_used": engines_used,
        "has_vector_db": "Vector" in engines_used,
        "has_sql": "SQL" in engines_used,
    }

    # Include source node count if available
    if response_obj and hasattr(response_obj, 'source_nodes'):
        metadata["source_node_count"] = len(response_obj.source_nodes)
        vector_sources = sum(
            1 for node in response_obj.source_nodes
            if hasattr(node, 'score') and node.score is not None
        )
        metadata["vector_source_count"] = vector_sources

    reflection_result = await reflection_agent.reflect(query, response, metadata)

    # Store reflection for potential use in refinement
    await ctx.store.set(ContextKeys.REFLECTION, reflection_result.model_dump())

    logger.info(f"Reflection complete: complete={reflection_result.is_complete}, "
               f"accurate={reflection_result.is_accurate}, "
               f"confidence={reflection_result.confidence_score:.2f}, "
               f"should_refine={reflection_result.should_refine}")

    # If reflection suggests refinement and we haven't exceeded max iterations
    if reflection_result.should_refine and iteration < config.max_refinement_iterations:
        logger.info(f"Generating refined query based on reflection feedback (iteration {iteration + 1})...")

        if reflection_result.suggested_improvements:
            logger.info(f"Suggested improvements: {', '.join(reflection_result.suggested_improvements)}")
        if reflection_result.missing_information:
            logger.info(f"Missing information: {', '.join(reflection_result.missing_information)}")

        # Generate refined query using refinement agent
        refined_query = await refinement_agent.refine_query(
            query, response, reflection_result
        )

        logger.info(f"Refined query: {refined_query}")
        await ctx.store.set(ContextKeys.REFINED_QUERY, refined_query)
        await ctx.store.set(ContextKeys.REFINEMENT_ITERATION, iteration + 1)
        await ctx.store.set(ContextKeys.QUERY, refined_query)

        # Store previous response and reflection feedback
        await ctx.store.set(ContextKeys.PREVIOUS_RESPONSE, response)
        await ctx.store.set(ContextKeys.REFLECTION_FEEDBACK, ReflectionFeedback(
            missing_information=reflection_result.missing_information or [],
            suggested_improvements=reflection_result.suggested_improvements or [],
            confidence_score=reflection_result.confidence_score,
        ).model_dump())

        # Re-execute with refined query
        return await decompose_query_fn(ctx, StartEvent(query=refined_query))

    elif reflection_result.should_refine:
        logger.info(
            f"Refinement suggested but max iterations ({config.max_refinement_iterations}) "
            "reached, returning current response"
        )

    # Get Response object from context
    response_obj = await ctx.store.get(ContextKeys.FINAL_RESPONSE_OBJ, default=None)
    if not response_obj:
        response_obj = Response(response=response)

    return ReflectionCompleteEvent(
        reflection_result=reflection_result,
        original_query=query,
        response=response,
        final_response=response_obj,
    )
