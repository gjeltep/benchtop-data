"""Simplified utilities for query routing workflows."""

from typing import Optional, List
import asyncio
from .context_keys import ContextKeys
from llama_index.core.base.response.schema import (
    Response,
    AsyncStreamingResponse,
    PydanticResponse,
    RESPONSE_TYPE,
)
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import QueryBundle


async def coerce_response(response: RESPONSE_TYPE) -> Response:
    """
    Convert any response type to standard Response.
    """
    if isinstance(response, AsyncStreamingResponse):
        return await response.get_response()
    elif isinstance(response, PydanticResponse):
        return await response.get_response()
    elif isinstance(response, Response):
        return response
    else:
        raise TypeError(f"Unexpected response type: {type(response)}")


async def synthesize_responses(
    query: str,
    response_strs: List[str],
    source_nodes: list,
    summarizer: Optional[TreeSummarize] = None,
) -> Response:
    """
    Synthesize multiple responses into one.

    Uses TreeSummarize if available, otherwise simple concatenation.
    Direct approach - no strategy pattern needed for binary choice.
    """
    if summarizer:
        # Use TreeSummarize for intelligent synthesis
        query_bundle = QueryBundle(query_str=query)
        summary = await summarizer.aget_response(query_bundle.query_str, response_strs)
        if isinstance(summary, str):
            return Response(response=summary, source_nodes=source_nodes)
        return summary
    else:
        # Simple concatenation
        combined_text = "\n\n".join(response_strs)
        return Response(response=combined_text, source_nodes=source_nodes)


async def execute_queries(
    selections: List,
    query: str,
    query_engines: List,
    process_response,
) -> List[Response]:
    """
    Execute queries on selected engines.

    Executes in parallel if multiple engines, sequential if single.
    Simple, direct approach - no enum-based dispatch needed.
    """
    if len(selections) > 1:
        # Multiple engines - execute in parallel
        async def query_and_process(selection):
            query_engine = query_engines[selection.index]
            response = await query_engine.aquery(query)
            return await process_response(response)

        tasks = [query_and_process(selection) for selection in selections]
        return await asyncio.gather(*tasks)
    else:
        # Single engine - execute directly
        selection = selections[0]
        query_engine = query_engines[selection.index]
        response = await query_engine.aquery(query)
        processed_response = await process_response(response)
        return [processed_response]


def is_single_response(responses: List[Response]) -> bool:
    """Check if we have a single response."""
    return len(responses) == 1


async def track_engine_selection(
    selected_indices: List[int], ctx, workflow_instance
) -> None:
    """Track which engine(s) were selected."""
    if len(selected_indices) == 1:
        # Single engine selected
        await ctx.store.set(ContextKeys.SELECTED_ENGINE_INDEX, selected_indices[0])
        workflow_instance.selected_engine_index = selected_indices[0]
    else:
        # Multiple engines selected
        await ctx.store.set(ContextKeys.SELECTED_ENGINE_INDICES, selected_indices)
        workflow_instance.selected_engine_index = selected_indices[0] if selected_indices else None
