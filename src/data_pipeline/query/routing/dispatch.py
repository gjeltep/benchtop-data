"""Enum-based dispatch patterns for query routing workflows."""

from enum import Enum
from typing import Callable, Dict, Optional, List
import asyncio
from llama_index.core.base.response.schema import (
    Response,
    AsyncStreamingResponse,
    PydanticResponse,
    RESPONSE_TYPE,
)
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import QueryBundle


class ResponseType(Enum):
    """Types of responses from query engines."""
    STREAMING = "streaming"
    PYDANTIC = "pydantic"
    STANDARD = "standard"


class SynthesisMode(Enum):
    """How to combine multiple responses."""
    SUMMARIZE = "summarize"
    COMBINE = "combine"


class ResponseCount(Enum):
    """Number of responses from query engines."""
    SINGLE = "single"
    MULTIPLE = "multiple"
    NONE = "none"


class ExecutionMode(Enum):
    """How to execute query engines."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class ResponseCoercer:
    """Coerces different response types to standard Response using enum-based dispatch."""

    @staticmethod
    def _get_response_type(response: RESPONSE_TYPE) -> ResponseType:
        """Detect the type of response."""
        if isinstance(response, AsyncStreamingResponse):
            return ResponseType.STREAMING
        if isinstance(response, PydanticResponse):
            return ResponseType.PYDANTIC
        return ResponseType.STANDARD

    @staticmethod
    async def coerce(response: RESPONSE_TYPE) -> Response:
        """Coerce response to standard Response type."""
        response_type = ResponseCoercer._get_response_type(response)

        coercers: Dict[ResponseType, Callable] = {
            ResponseType.STREAMING: lambda r: r.get_response(),
            ResponseType.PYDANTIC: lambda r: r.get_response(),
            ResponseType.STANDARD: lambda r: r,
        }

        coercer = coercers[response_type]
        result = coercer(response)
        coerced = await result if asyncio.iscoroutine(result) else result
        # Ensure we always return Response (STANDARD case already is Response)
        if not isinstance(coerced, Response):
            raise TypeError(f"Expected Response, got {type(coerced)}")
        return coerced


class SynthesisStrategy:
    """Strategy pattern for response synthesis using enum-based dispatch."""

    def __init__(self, summarizer: Optional[TreeSummarize]):
        self.summarizer = summarizer
        self.mode = SynthesisMode.SUMMARIZE if summarizer else SynthesisMode.COMBINE

    async def synthesize(
        self, query: str, response_strs: list[str], source_nodes: list
    ) -> Response:
        """Synthesize multiple responses based on mode."""
        strategies: Dict[SynthesisMode, Callable] = {
            SynthesisMode.SUMMARIZE: self._summarize,
            SynthesisMode.COMBINE: self._combine,
        }
        handler = strategies[self.mode]
        result = await handler(query, response_strs, source_nodes) if asyncio.iscoroutinefunction(handler) else handler(query, response_strs, source_nodes)
        # Ensure we always return Response
        if isinstance(result, str):
            return Response(response=result, source_nodes=source_nodes)
        return result

    async def _summarize(self, query: str, response_strs: list[str], source_nodes: list) -> Response:
        """Synthesize using summarizer."""
        query_bundle = QueryBundle(query_str=query)
        summary = await self.summarizer.aget_response(query_bundle.query_str, response_strs)
        if isinstance(summary, str):
            return Response(response=summary, source_nodes=source_nodes)
        return summary

    @staticmethod
    def _combine(query: str, response_strs: list[str], source_nodes: list) -> Response:
        """Combine responses without summarization."""
        combined_text = "\n\n".join(response_strs)
        return Response(response=combined_text, source_nodes=source_nodes)


class ResponseCountHandler:
    """Handles responses based on count using enum-based dispatch."""

    @staticmethod
    def get_count(responses: List[Response]) -> ResponseCount:
        """Determine response count."""
        if not responses:
            return ResponseCount.NONE
        return ResponseCount.SINGLE if len(responses) == 1 else ResponseCount.MULTIPLE

    @staticmethod
    async def track_selection(
        selected_indices: List[int], ctx, workflow_instance
    ) -> None:
        """Track engine selection based on count."""
        count = ResponseCount.SINGLE if len(selected_indices) == 1 else ResponseCount.MULTIPLE

        handlers: Dict[ResponseCount, Callable] = {
            ResponseCount.SINGLE: lambda: ResponseCountHandler._track_single(
                selected_indices, ctx, workflow_instance
            ),
            ResponseCount.MULTIPLE: lambda: ResponseCountHandler._track_multiple(
                selected_indices, ctx, workflow_instance
            ),
        }
        await handlers[count]()

    @staticmethod
    async def _track_single(selected_indices: List[int], ctx, workflow_instance) -> None:
        """Track single engine selection."""
        await ctx.store.set("selected_engine_index", selected_indices[0])
        workflow_instance.selected_engine_index = selected_indices[0]

    @staticmethod
    async def _track_multiple(selected_indices: List[int], ctx, workflow_instance) -> None:
        """Track multiple engine selection."""
        await ctx.store.set("selected_engine_indices", selected_indices)
        workflow_instance.selected_engine_index = selected_indices[0] if selected_indices else None


class QueryExecutor:
    """Executes queries based on execution mode using enum-based dispatch."""

    @staticmethod
    def get_mode(selection_count: int) -> ExecutionMode:
        """Determine execution mode based on selection count."""
        return ExecutionMode.PARALLEL if selection_count > 1 else ExecutionMode.SEQUENTIAL

    @staticmethod
    async def execute(
        selections: List,
        query: str,
        query_engines: List,
        process_response: Callable,
    ) -> List[Response]:
        """Execute queries based on mode."""
        mode = QueryExecutor.get_mode(len(selections))

        executors: Dict[ExecutionMode, Callable] = {
            ExecutionMode.PARALLEL: QueryExecutor._execute_parallel,
            ExecutionMode.SEQUENTIAL: QueryExecutor._execute_sequential,
        }

        return await executors[mode](selections, query, query_engines, process_response)

    @staticmethod
    async def _execute_parallel(
        selections: List, query: str, query_engines: List, process_response: Callable
    ) -> List[Response]:
        """Execute multiple engines in parallel."""
        async def query_and_process(selection):
            query_engine = query_engines[selection.index]
            response = await query_engine.aquery(query)
            return await process_response(response)

        tasks = [query_and_process(selection) for selection in selections]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def _execute_sequential(
        selections: List, query: str, query_engines: List, process_response: Callable
    ) -> List[Response]:
        """Execute single engine sequentially."""
        selection = selections[0]
        query_engine = query_engines[selection.index]
        response = await query_engine.aquery(query)
        processed_response = await process_response(response)
        return [processed_response]
