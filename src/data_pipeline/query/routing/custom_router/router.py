"""
Router for query engines using LlamaIndex Workflows.

This implements a native LlamaIndex Workflow-based router that:
- Routes between SQL and vector query engines
- Supports streaming events for reasoning token capture
- Decomposes complex queries into sub-questions (Sub-Question Query Engine pattern)
- Reflects on responses to improve quality (Reflection pattern)
- Maintains the same interface as RouterQueryEngine

Based on: https://developers.llamaindex.ai/python/llamaagents/workflows/
"""

from typing import List
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.core.base.base_selector import SelectorResult, SingleSelection
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.base.response.schema import Response, AsyncStreamingResponse
from ....logging import get_logger
from .dispatch import coerce_response, synthesize_responses, execute_queries, is_single_response
from .response_utils import extract_response_text, get_engine_display_name
from .refinement import enhance_query_with_refinement_context
from .decomposition import execute_sub_questions
from .reflection import reflect_and_refine
from .context_keys import ContextKeys
from .types import SubQuestionDict
from ...agents import SubQuestionDecomposer, ReflectionAgent, ReflectionEvent
from ...agents.reflection_agent import ReflectionResult
from ...agents.refinement_agent import RefinementAgent
from ...agents.config import AgenticConfig

logger = get_logger(__name__)


class QueryEngineSelectionEvent(Event):
    """Result of selecting the query engine tools."""
    selected_query_engines: SelectorResult


class SynthesizeEvent(Event):
    """Event for synthesizing the response from different query engines."""
    result: List[Response]
    selected_query_engines: SelectorResult


class ReflectionCompleteEvent(Event):
    """Event after reflection is complete."""
    reflection_result: ReflectionResult  # Store ReflectionResult directly, not ReflectionEvent
    original_query: str
    response: str
    final_response: Response  # Store Response object for consistency


class SubQuestionExecutionEvent(Event):
    """Event containing results from executing all sub-questions."""
    sub_question_results: List[Response]
    sub_questions: List[SubQuestionDict]
    original_query: str


class RefinementEvent(Event):
    """Event for query refinement based on reflection."""
    refined_query: str
    original_query: str
    refinement_reason: str


class RouterQueryEngineWorkflow(Workflow):
    """
    Custom Router Query Engine implemented as a LlamaIndex Workflow.

    This is the advanced workflow with built-in agentic patterns:
    - Sub-question decomposition for complex queries (always enabled)
    - Reflection & refinement for quality improvement (always enabled)
    - Streaming events for reasoning token capture
    - Explicit step-by-step execution
    - Better observability

    For simpler routing without these features, use react_agent workflow.

    Workflow Steps:
    1. decompose_query: Decomposes complex queries into sub-questions
    2. selector: Routes query to appropriate engine(s)
    3. generate_responses: Executes selected engine(s)
    4. synthesize_and_reflect: Combines multiple responses and validates quality (with optional refinement)
    5. finalize: Returns the final response after reflection
    """

    def __init__(
        self,
        reasoning_handler=None,
        llm=None,
        summarizer=None,
        query_engine_tools=None,
        reflection_threshold=0.7,
        **kwargs
    ):
        """
        Initialize router workflow.

        Custom router always includes sub-question decomposition and reflection.
        For simpler routing without these features, use react_agent workflow instead.

        Args:
            reasoning_handler: Optional handler for capturing reasoning tokens
            llm: LLM instance (stored as instance var to avoid deepcopy issues)
            summarizer: Response synthesizer (stored as instance var to avoid deepcopy issues)
            query_engine_tools: Query engine tools (stored as instance var to avoid deepcopy issues)
            reflection_threshold: Confidence threshold for reflection (0-1, default: 0.7)
            **kwargs: Workflow configuration (timeout, verbose, etc.)
        """
        super().__init__(**kwargs)
        self.reasoning_handler = reasoning_handler
        self.llm = llm  # Store LLM as instance var, not in context (avoids deepcopy issues)
        self.summarizer = summarizer  # Store summarizer as instance var (it contains LLM)
        self.query_engine_tools = query_engine_tools  # Store tools as instance var (they contain LLMs)
        self.selected_engine_index = None  # Track which engine was selected (0 = vector, 1 = SQL)

        # Initialize config
        self.config = AgenticConfig(
            reflection_threshold=reflection_threshold,
            max_refinement_iterations=kwargs.get("max_refinement_iterations", 2),
        )

        # Initialize agentic components (required for custom router)
        if not llm:
            raise ValueError("LLM must be provided for custom router workflow")

        self.sub_question_decomposer = SubQuestionDecomposer(llm, reasoning_handler)
        self.reflection_agent = ReflectionAgent(llm, reasoning_handler, reflection_threshold)
        self.refinement_agent = RefinementAgent(llm, self.config)

    def _log_step(self, title: str, message: str = ""):
        """Helper for consistent step logging."""
        logger.info("=" * 70)
        logger.info(title)
        logger.info("=" * 70)
        if message:
            logger.info(message)

    @step
    async def decompose_query(
        self, ctx: Context, ev: StartEvent
    ) -> QueryEngineSelectionEvent | SubQuestionExecutionEvent:
        """
        Optionally decompose complex queries into sub-questions.

        If sub-question decomposition is enabled and the query is complex,
        this step breaks it down into simpler sub-questions that can be
        executed independently and then synthesized.
        """
        query = ev.get("query")
        await ctx.store.set(ContextKeys.QUERY, query)

        # Set original query only if not already set
        existing = await ctx.store.get(ContextKeys.ORIGINAL_QUERY, default=None)
        if existing is None:
            await ctx.store.set(ContextKeys.ORIGINAL_QUERY, query)

        self._log_step("STEP 1: QUERY DECOMPOSITION", f"Analyzing query: {query[:100]}...")

        # Decompose the query
        decomposition = await self.sub_question_decomposer.decompose(query)
        await ctx.store.set(ContextKeys.DECOMPOSITION, decomposition.model_dump())

        # If decomposition is not needed, proceed with original query
        if not decomposition.needs_decomposition or not decomposition.sub_questions:
            logger.info("Query does not need decomposition, proceeding with original query")
            return await self._selector_step(ctx, query)

        # Store sub-questions for later execution
        sub_questions_data = [sq.model_dump() for sq in decomposition.sub_questions]
        await ctx.store.set(ContextKeys.SUB_QUESTIONS, sub_questions_data)
        await ctx.store.set(ContextKeys.HAS_SUB_QUESTIONS, True)

        # Execute all sub-questions in parallel using decomposition hints
        logger.info(f"✓ Decomposed into {len(decomposition.sub_questions)} sub-questions")
        logger.info("=" * 70)

        self._log_step("STEP 2: PARALLEL SUB-QUESTION EXECUTION",
                      f"Executing {len(decomposition.sub_questions)} sub-questions...")
        result = await execute_sub_questions(
            sub_questions=decomposition.sub_questions,
            original_query=query,
            query_engine_tools=self.query_engine_tools,
            selector_step_fn=self._selector_step,
            coerce_to_response_fn=self._coerce_to_response,
            ctx=ctx,
            reasoning_handler=self.reasoning_handler,
            config=self.config,
        )

        return SubQuestionExecutionEvent(
            sub_question_results=result["sub_question_results"],
            sub_questions=result["sub_questions"],
            original_query=result["original_query"],
        )

    async def _selector_step(self, ctx: Context, query: str) -> QueryEngineSelectionEvent:
        """Internal selector step that can be called from decompose or directly."""
        self._log_step("STEP: QUERY ROUTING", f"Routing query: {query[:100]}...")

        if not self.llm:
            raise ValueError("LLM must be provided to workflow constructor")
        if not self.query_engine_tools:
            raise ValueError("query_engine_tools must be provided to workflow constructor")

        # Enhance query with refinement context if in refinement iteration
        query = await enhance_query_with_refinement_context(ctx, query, self.config)

        # Use LLMSingleSelector which uses prompts instead of function calling
        # This works with models that don't support tools (like DeepSeek)
        selector = LLMSingleSelector.from_defaults(llm=self.llm)

        # Get metadata for routing decision
        query_engines_metadata = [
            tool.metadata for tool in self.query_engine_tools
        ]

        selected_query_engines = await selector.aselect(
            query_engines_metadata, query
        )

        logger.info(f"✓ Router selected {len(selected_query_engines.selections)} engine(s)")
        for selection in selected_query_engines.selections:
            engine_name = get_engine_display_name(selection.index)
            logger.info(f"  → {engine_name} Engine: {selection.reason[:80]}...")
        logger.info("=" * 70)

        return QueryEngineSelectionEvent(
            selected_query_engines=selected_query_engines
        )

    async def _coerce_to_response(self, response) -> Response:
        """Normalize different response types into a standard Response."""
        return await coerce_response(response)

    async def _acombine_responses(
        self, responses: List[Response], query: str
    ) -> Response:
        """
        Combine multiple responses following LlamaIndex best practices.

        This method follows the official RouterQueryEngineWorkflow pattern:
        - Extracts response text and source nodes
        - Uses TreeSummarize if available, otherwise combines
        - Returns a Response object with all source nodes

        Args:
            responses: List of Response objects (already coerced)
            query: Original query string

        Returns:
            Combined Response object
        """
        response_strs = []
        source_nodes = []

        for response in responses:
            # Responses are already coerced to Response in process_response
            source_nodes.extend(response.source_nodes)
            response_strs.append(str(response))

        # Use synthesize function (TreeSummarize if available, else combine)
        return await synthesize_responses(query, response_strs, source_nodes, self.summarizer)

    @step
    async def synthesize_sub_questions(
        self, ctx: Context, ev: SubQuestionExecutionEvent
    ) -> SynthesizeEvent:
        """
        Synthesize results from multiple sub-questions into a coherent answer.

        This step combines all sub-question responses using the original query context.
        """
        original_query = ev.original_query
        sub_question_results = ev.sub_question_results
        sub_questions = ev.sub_questions

        self._log_step("STEP 3: SUB-QUESTION SYNTHESIS",
                       f"Synthesizing {len(sub_question_results)} sub-question results into final answer...")

        # Build context for synthesis
        response_strs = []
        source_nodes = []

        for i, (result, sub_q) in enumerate(zip(sub_question_results, sub_questions), 1):
            question = sub_q.get("question", f"Sub-question {i}")
            response_text = extract_response_text(result)

            response_strs.append(f"Sub-question {i}: {question}\nAnswer: {response_text}")
            source_nodes.extend(result.source_nodes)
            logger.info(f"  ✓ Sub-Q {i} result: {len(response_text)} chars")
            logger.info(f"  Sub-Q {i} preview: {response_text[:500]}...")

        # Synthesize results
        if self.summarizer:
            logger.info("Using TreeSummarize to synthesize results...")
        else:
            logger.info("Using simple concatenation (no summarizer available)")

        synthesized = await synthesize_responses(
            original_query, response_strs, source_nodes, self.summarizer
        )

        logger.info(f"✓ Synthesis complete: {len(str(synthesized))} chars")
        logger.info("=" * 70)

        return SynthesizeEvent(
            result=[synthesized],
            selected_query_engines=SelectorResult(selections=[SingleSelection(index=0, reason="Sub-question synthesis")]),
        )

    @step
    async def generate_responses(
        self, ctx: Context, ev: QueryEngineSelectionEvent
    ) -> SynthesizeEvent:
        """
        Generate responses from the selected query engine(s).

        This step executes the actual queries. If streaming is enabled,
        reasoning tokens are captured directly from streaming responses.
        """
        query = await ctx.store.get(ContextKeys.QUERY, default="")
        selected_query_engines = ev.selected_query_engines

        # Enhance query with refinement context if in refinement iteration
        query = await enhance_query_with_refinement_context(ctx, query, self.config)

        self._log_step("STEP: QUERY EXECUTION", f"Executing query: {query[:100]}...")

        # Use instance variable for query_engine_tools (they contain LLMs, can't be deepcopied)
        query_engine_tools = self.query_engine_tools
        if not query_engine_tools:
            raise ValueError("query_engine_tools must be provided to workflow constructor")

        query_engines = [tool.query_engine for tool in query_engine_tools]

        async def process_response(response):
            """Process response and capture reasoning tokens."""
            if isinstance(response, AsyncStreamingResponse):
                async for _ in response.async_response_gen():
                    pass  # Consume stream (reasoning tokens captured by wrapper)
                if self.reasoning_handler:
                    self.reasoning_handler.log_complete_reasoning()
            return await self._coerce_to_response(response)

        # Execute selected engine(s)
        responses = await execute_queries(
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
    async def synthesize_and_reflect(
        self, ctx: Context, ev: SynthesizeEvent
    ) -> StopEvent | ReflectionCompleteEvent | QueryEngineSelectionEvent:
        """Synthesize final response from query engine results and reflect on quality."""
        responses = ev.result
        selected_query_engines = ev.selected_query_engines
        query = await ctx.store.get(ContextKeys.QUERY, default="")
        original_query = await ctx.store.get(ContextKeys.ORIGINAL_QUERY, default=None) or query

        self._log_step("STEP: RESPONSE SYNTHESIS", f"Synthesizing {len(responses)} response(s)...")

        # Track engine selection for metadata
        selected_indices = [s.index for s in selected_query_engines.selections]
        self.selected_engine_index = selected_indices[0] if selected_indices else None

        # Synthesize responses if needed
        if is_single_response(responses):
            # Single response - no synthesis needed
            final_response = responses[0]
            logger.info("Single response - no synthesis needed")
        else:
            # Multiple responses - synthesize or combine
            logger.info("Multiple responses - synthesizing...")
            final_response = await self._acombine_responses(
                responses, query
            )
            logger.info(f"✓ Synthesis complete: {len(str(final_response))} chars")

        # Add metadata following LlamaIndex best practices
        final_response.metadata = final_response.metadata or {}
        final_response.metadata["selector_result"] = selected_query_engines

        logger.info("=" * 70)

        # Store final response text for reflection (reflection expects string)
        final_response_text = str(final_response)
        await ctx.store.set(ContextKeys.FINAL_RESPONSE_OBJ, final_response)

        # Store engine selection info for reflection
        engine_names = [
            get_engine_display_name(selection.index)
            for selection in selected_query_engines.selections
        ]
        await ctx.store.set(ContextKeys.ENGINES_USED, engine_names)

        # Reflect on the response quality
        refinement_iteration = await ctx.store.get(ContextKeys.REFINEMENT_ITERATION, default=0)
        self._log_step(
            f"STEP: REFLECTION & QUALITY ASSESSMENT (Iteration {refinement_iteration + 1})",
            "Evaluating response quality..."
        )
        reflection_result = await reflect_and_refine(
            reflection_agent=self.reflection_agent,
            refinement_agent=self.refinement_agent,
            decompose_query_fn=self.decompose_query,
            ctx=ctx,
            query=original_query,
            response=final_response_text,
            iteration=refinement_iteration,
            response_obj=final_response,
            config=self.config,
        )

        # If reflection returns QueryEngineSelectionEvent, it means we're refining
        if isinstance(reflection_result, QueryEngineSelectionEvent):
            return reflection_result

        return reflection_result


    @step
    async def finalize(
        self, ctx: Context, ev: ReflectionCompleteEvent
    ) -> StopEvent:
        """
        Final step after reflection - returns the response.

        This step receives the reflection result and emits the terminating event
        with the final response.
        """
        self._log_step("STEP: FINAL RESPONSE", "Workflow complete - returning final answer")
        return StopEvent(result=ev.final_response)

