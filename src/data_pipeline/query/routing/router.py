"""
Router for query engines using LlamaIndex Workflows.

This implements a native LlamaIndex Workflow-based router that:
- Routes between SQL and vector query engines
- Supports streaming events for reasoning token capture
- Maintains the same interface as RouterQueryEngine

Based on: https://developers.llamaindex.ai/python/llamaagents/workflows/
"""

from typing import List, Optional, Dict, Any
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.core.base.base_selector import SelectorResult
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.base.response.schema import Response, AsyncStreamingResponse
import asyncio
from ...logging_config import get_logger
from .dispatch import ResponseCoercer, SynthesisStrategy, ResponseCountHandler, QueryExecutor, ResponseCount

logger = get_logger(__name__)


class QueryEngineSelectionEvent(Event):
    """Result of selecting the query engine tools."""
    selected_query_engines: SelectorResult


class SynthesizeEvent(Event):
    """Event for synthesizing the response from different query engines."""
    result: List[Response]
    selected_query_engines: SelectorResult


class RouterQueryEngineWorkflow(Workflow):
    """
    Router Query Engine implemented as a LlamaIndex Workflow.

    This provides native LlamaIndex routing with support for:
    - Streaming events (for reasoning token capture)
    - Explicit step-by-step execution
    - Better observability

    Steps:
    1. selector: Routes query to appropriate engine(s)
    2. generate_responses: Executes selected engine(s)
    3. synthesize: Combines multiple responses if needed
    """

    def __init__(self, reasoning_handler=None, llm=None, summarizer=None, query_engine_tools=None, **kwargs):
        """
        Initialize router workflow.

        Args:
            reasoning_handler: Optional handler for capturing reasoning tokens
            llm: LLM instance (stored as instance var to avoid deepcopy issues)
            summarizer: Response synthesizer (stored as instance var to avoid deepcopy issues)
            query_engine_tools: Query engine tools (stored as instance var to avoid deepcopy issues)
            **kwargs: Workflow configuration (timeout, verbose, etc.)
        """
        super().__init__(**kwargs)
        self.reasoning_handler = reasoning_handler
        self.llm = llm  # Store LLM as instance var, not in context (avoids deepcopy issues)
        self.summarizer = summarizer  # Store summarizer as instance var (it contains LLM)
        self.query_engine_tools = query_engine_tools  # Store tools as instance var (they contain LLMs)
        self.selected_engine_index = None  # Track which engine was selected (0 = vector, 1 = SQL)

    @step
    async def selector(
        self, ctx: Context, ev: StartEvent
    ) -> QueryEngineSelectionEvent:
        """
        Selects query engine(s) based on the query.

        This step uses an LLM to decide which engine(s) to use.
        Reasoning tokens from this decision are captured if reasoning_handler is set.
        """
        # Store query in context (strings are safe to deepcopy)
        # NOTE: Do NOT store LLM, summarizer, or query_engine_tools in context
        # They contain locks that can't be deepcopied by workflow runtime
        query = ev.get("query")
        # Use instance variables for LLM and tools (set in constructor)
        if not self.llm:
            raise ValueError("LLM must be provided to workflow constructor")
        if not self.query_engine_tools:
            raise ValueError("query_engine_tools must be provided to workflow constructor")

        await ctx.store.set("query", query)

        # Use LLMSingleSelector which uses prompts instead of function calling
        # This works with models that don't support tools (like DeepSeek)
        selector = LLMSingleSelector.from_defaults(llm=self.llm)

        # Get metadata for routing decision
        query_engines_metadata = [
            tool.metadata for tool in self.query_engine_tools
        ]

        # Select engine(s) - this LLM call may produce reasoning tokens
        # The selector internally uses the LLM, so reasoning tokens will be captured
        # by the LLM's streaming mechanism if enabled
        selected_query_engines = await selector.aselect(
            query_engines_metadata, query
        )

        logger.info(f"Router selected {len(selected_query_engines.selections)} engine(s)")
        for selection in selected_query_engines.selections:
            logger.info(f"  - Engine {selection.index}: {selection.reason}")

        return QueryEngineSelectionEvent(
            selected_query_engines=selected_query_engines
        )

    async def _coerce_to_response(self, response) -> Response:
        """Normalize different response types into a standard Response."""
        return await ResponseCoercer.coerce(response)

    @step
    async def generate_responses(
        self, ctx: Context, ev: QueryEngineSelectionEvent
    ) -> SynthesizeEvent:
        """
        Generate responses from the selected query engine(s).

        This step executes the actual queries. If streaming is enabled,
        reasoning tokens are captured directly from streaming responses.
        """
        query = await ctx.store.get("query")
        selected_query_engines = ev.selected_query_engines
        # Use instance variable for query_engine_tools (they contain LLMs, can't be deepcopied)
        query_engine_tools = self.query_engine_tools
        if not query_engine_tools:
            raise ValueError("query_engine_tools must be provided to workflow constructor")

        query_engines = [tool.query_engine for tool in query_engine_tools]

        # Helper to process streaming response and capture reasoning tokens
        async def process_response(response):
            """Process response (streaming or standard) and capture reasoning tokens."""
            if isinstance(response, AsyncStreamingResponse):
                # Stream the response
                # Note: Thinking tokens are captured via ThinkingOllamaWrapper at the LLM level
                async for chunk in response.async_response_gen():
                    # Collect response text
                    if hasattr(chunk, "delta") and chunk.delta:
                        pass  # Chunk processed, delta available if needed
                    elif hasattr(chunk, "text") and chunk.text:
                        pass  # Chunk processed, text available if needed

                # Ensure reasoning is logged after stream completes
                if self.reasoning_handler:
                    self.reasoning_handler.log_complete_reasoning()

            # Note: Thinking tokens are captured via ThinkingOllamaWrapper at the LLM level
            return await self._coerce_to_response(response)

        # Execute selected engine(s) using enum-based dispatch
        responses = await QueryExecutor.execute(
            selected_query_engines.selections,
            query,
            query_engines,
            process_response,
        )

        return SynthesizeEvent(
            result=responses,
            selected_query_engines=selected_query_engines,
        )

    @step
    async def synthesize(
        self, ctx: Context, ev: SynthesizeEvent
    ) -> StopEvent:
        """Synthesize final response from query engine results."""
        responses = ev.result
        selected_query_engines = ev.selected_query_engines
        query = await ctx.store.get("query")

        # Track engine selection for metadata using enum-based dispatch
        selected_indices = [s.index for s in selected_query_engines.selections]
        await ResponseCountHandler.track_selection(selected_indices, ctx, self)

        # Handle responses based on count using enum-based dispatch
        response_count = ResponseCountHandler.get_count(responses)

        if response_count == ResponseCount.SINGLE:
            # Responses are already coerced to Response in process_response
            return StopEvent(result=str(responses[0]))

        # Multiple responses - synthesize or combine
        response_strs = []
        source_nodes = []
        for response in responses:
            # Responses are already coerced to Response in process_response
            source_nodes.extend(response.source_nodes)
            response_strs.append(str(response))

        strategy = SynthesisStrategy(self.summarizer)
        result = await strategy.synthesize(query, response_strs, source_nodes)
        # result is always Response now
        return StopEvent(result=str(result))

