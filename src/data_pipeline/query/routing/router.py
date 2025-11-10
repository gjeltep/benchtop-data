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

from typing import List, Dict, Any
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
from llama_index.core.llms import ChatMessage, MessageRole
import asyncio
from ...logging_config import get_logger
from .dispatch import ResponseCoercer, SynthesisStrategy, ResponseCountHandler, QueryExecutor, ResponseCount
from .response_utils import extract_response_text
from ..agents import SubQuestionDecomposer, ReflectionAgent, ReflectionEvent

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
    reflection: ReflectionEvent
    final_response: Response  # Store Response object for consistency


class SubQuestionExecutionEvent(Event):
    """Event containing results from executing all sub-questions."""
    sub_question_results: List[Response]
    sub_questions: List[Dict[str, Any]]
    original_query: str


class RefinementEvent(Event):
    """Event for query refinement based on reflection."""
    refined_query: str
    original_query: str
    refinement_reason: str


class RouterQueryEngineWorkflow(Workflow):
    """
    Router Query Engine implemented as a LlamaIndex Workflow.

    This provides native LlamaIndex routing with support for:
    - Streaming events (for reasoning token capture)
    - Explicit step-by-step execution
    - Sub-question decomposition for complex queries
    - Reflection for quality improvement
    - Better observability

    Steps:
    1. decompose_query: Optionally decomposes complex queries into sub-questions
    2. selector: Routes query to appropriate engine(s)
    3. generate_responses: Executes selected engine(s)
    4. synthesize: Combines multiple responses if needed
    5. reflect: Validates and improves response quality
    """

    def __init__(
        self,
        reasoning_handler=None,
        llm=None,
        summarizer=None,
        query_engine_tools=None,
        enable_sub_questions=True,
        enable_reflection=True,
        reflection_threshold=0.7,
        **kwargs
    ):
        """
        Initialize router workflow.

        Args:
            reasoning_handler: Optional handler for capturing reasoning tokens
            llm: LLM instance (stored as instance var to avoid deepcopy issues)
            summarizer: Response synthesizer (stored as instance var to avoid deepcopy issues)
            query_engine_tools: Query engine tools (stored as instance var to avoid deepcopy issues)
            enable_sub_questions: Enable sub-question decomposition (default: True)
            enable_reflection: Enable reflection for quality improvement (default: True)
            reflection_threshold: Confidence threshold for reflection (0-1, default: 0.7)
            **kwargs: Workflow configuration (timeout, verbose, etc.)
        """
        super().__init__(**kwargs)
        self.reasoning_handler = reasoning_handler
        self.llm = llm  # Store LLM as instance var, not in context (avoids deepcopy issues)
        self.summarizer = summarizer  # Store summarizer as instance var (it contains LLM)
        self.query_engine_tools = query_engine_tools  # Store tools as instance var (they contain LLMs)
        self.selected_engine_index = None  # Track which engine was selected (0 = vector, 1 = SQL)

        # Agentic pattern components
        self.enable_sub_questions = enable_sub_questions
        self.enable_reflection = enable_reflection
        self.sub_question_decomposer = None
        self.reflection_agent = None

        if enable_sub_questions and llm:
            self.sub_question_decomposer = SubQuestionDecomposer(llm, reasoning_handler)
        if enable_reflection and llm:
            self.reflection_agent = ReflectionAgent(llm, reasoning_handler, reflection_threshold)

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
        await ctx.store.set("query", query)
        # Only set original_query if it doesn't exist (preserve true original across refinements)
        existing_original = await ctx.store.get("original_query", default=None)
        if existing_original is None:
            await ctx.store.set("original_query", query)

        self._log_step("STEP 1: QUERY DECOMPOSITION", f"Analyzing query: {query[:100]}...")

        # If sub-question decomposition is disabled, skip to selector
        if not self.enable_sub_questions or not self.sub_question_decomposer:
            logger.info("Sub-question decomposition disabled, proceeding to routing")
            return await self._selector_step(ctx, query)

        # Decompose the query
        decomposition = await self.sub_question_decomposer.decompose(query)
        await ctx.store.set("decomposition", decomposition.model_dump())

        # If decomposition is not needed, proceed with original query
        if not decomposition.needs_decomposition or not decomposition.sub_questions:
            logger.info("Query does not need decomposition, proceeding with original query")
            return await self._selector_step(ctx, query)

        # Store sub-questions for later execution
        sub_questions_data = [sq.model_dump() for sq in decomposition.sub_questions]
        await ctx.store.set("sub_questions", sub_questions_data)
        await ctx.store.set("has_sub_questions", True)

        # Execute all sub-questions in parallel using decomposition hints
        logger.info(f"✓ Decomposed into {len(decomposition.sub_questions)} sub-questions")
        logger.info("=" * 70)
        return await self._execute_sub_questions(ctx, decomposition.sub_questions, query)

    async def _execute_sub_questions(
        self, ctx: Context, sub_questions: List, original_query: str
    ) -> SubQuestionExecutionEvent:
        """
        Execute all sub-questions in parallel using decomposition hints for routing.

        Uses requires_sql/requires_semantic hints to route directly without LLM selector.
        """
        self._log_step("STEP 2: PARALLEL SUB-QUESTION EXECUTION",
                       f"Executing {len(sub_questions)} sub-questions in parallel...")

        if not self.query_engine_tools:
            raise ValueError("query_engine_tools must be provided to workflow constructor")

        query_engines = [tool.query_engine for tool in self.query_engine_tools]
        # Engine 0 = vector, Engine 1 = SQL

        async def execute_sub_question(sub_q, index):
            """Execute a single sub-question using hints for routing."""
            # Extract question and hints (handle both Pydantic model and dict)
            if hasattr(sub_q, 'question'):
                question = sub_q.question
                requires_sql = sub_q.requires_sql
                requires_semantic = sub_q.requires_semantic
            else:
                question = sub_q.get("question", "")
                requires_sql = sub_q.get("requires_sql", False)
                requires_semantic = sub_q.get("requires_semantic", False)

            # If this is a refinement iteration, enhance the question following LlamaIndex reflection pattern
            # This ensures the model can use the previous output and feedback to generate an improved result
            refinement_iteration = await ctx.store.get("refinement_iteration", default=0)
            if refinement_iteration > 0:
                reflection_feedback = await ctx.store.get("reflection_feedback", default=None)
                previous_response = await ctx.store.get("previous_response", default=None)

                if reflection_feedback:
                    missing_info = reflection_feedback.get("missing_information", [])
                    improvements = reflection_feedback.get("suggested_improvements", [])

                    # Build enhancement following LlamaIndex reflection pattern: show what was wrong
                    enhancement_parts = []
                    if missing_info:
                        enhancement_parts.append(f"Missing: {', '.join(missing_info[:2])}")
                    if improvements:
                        enhancement_parts.append(f"Improve: {improvements[0]}")
                    if previous_response and len(previous_response) < 200:
                        # Include short previous response for context (like LlamaIndex pattern)
                        enhancement_parts.append(f"Previous: {previous_response[:150]}")

                    if enhancement_parts:
                        question = f"{question} [Refinement context: {'; '.join(enhancement_parts)}]"
                        logger.debug(f"Enhanced sub-question {index+1} with reflection context")

            # Route using hints (skip LLM selector when clear)
            if requires_sql and not requires_semantic:
                engine_index = 1
                logger.info(f"  → Sub-Q {index+1}: '{question[:60]}...' → SQL (hint-based)")
            elif requires_semantic and not requires_sql:
                engine_index = 0
                logger.info(f"  → Sub-Q {index+1}: '{question[:60]}...' → Vector (hint-based)")
            else:
                logger.info(f"  → Sub-Q {index+1}: '{question[:60]}...' → Using LLM selector")
                selection_event = await self._selector_step(ctx, question)
                engine_index = selection_event.selected_query_engines.selections[0].index

            # Execute and process response
            response = await query_engines[engine_index].aquery(question)
            if isinstance(response, AsyncStreamingResponse):
                async for _ in response.async_response_gen():
                    pass
                if self.reasoning_handler:
                    self.reasoning_handler.log_complete_reasoning()

            logger.info(f"  ✓ Sub-Q {index+1} completed")
            return await self._coerce_to_response(response)

        # Execute all sub-questions in parallel
        sub_question_results = await asyncio.gather(*[
            execute_sub_question(sq, i) for i, sq in enumerate(sub_questions)
        ])

        logger.info(f"✓ Completed execution of {len(sub_question_results)} sub-questions")
        logger.info("=" * 70)

        # Convert sub_questions to dict format
        sub_questions_dict = [
            sq.model_dump() if hasattr(sq, 'model_dump') else
            sq if isinstance(sq, dict) else
            {"question": getattr(sq, 'question', str(sq)),
             "requires_sql": getattr(sq, 'requires_sql', False),
             "requires_semantic": getattr(sq, 'requires_semantic', False)}
            for sq in sub_questions
        ]

        return SubQuestionExecutionEvent(
            sub_question_results=sub_question_results,
            sub_questions=sub_questions_dict,
            original_query=original_query,
        )

    async def _selector_step(self, ctx: Context, query: str) -> QueryEngineSelectionEvent:
        """Internal selector step that can be called from decompose or directly."""
        self._log_step("STEP: QUERY ROUTING", f"Routing query: {query[:100]}...")

        if not self.llm:
            raise ValueError("LLM must be provided to workflow constructor")
        if not self.query_engine_tools:
            raise ValueError("query_engine_tools must be provided to workflow constructor")

        # If this is a refinement iteration, enhance the query with context following LlamaIndex reflection pattern
        # This ensures the model can use the previous output and feedback to generate an improved result
        refinement_iteration = await ctx.store.get("refinement_iteration", default=0)
        if refinement_iteration > 0:
            reflection_feedback = await ctx.store.get("reflection_feedback", default=None)
            previous_response = await ctx.store.get("previous_response", default=None)

            if reflection_feedback:
                missing_info = reflection_feedback.get("missing_information", [])
                improvements = reflection_feedback.get("suggested_improvements", [])

                # Build enhancement following LlamaIndex reflection pattern: show what was wrong
                enhancement_parts = []
                if missing_info:
                    enhancement_parts.append(f"Missing: {', '.join(missing_info[:2])}")
                if improvements:
                    enhancement_parts.append(f"Improve: {improvements[0]}")
                if previous_response and len(previous_response) < 200:
                    # Include short previous response for context (like LlamaIndex pattern)
                    enhancement_parts.append(f"Previous: {previous_response[:150]}")

                if enhancement_parts:
                    query = f"{query} [Refinement context: {'; '.join(enhancement_parts)}]"
                    logger.debug(f"Enhanced query with reflection context")

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

        logger.info(f"✓ Router selected {len(selected_query_engines.selections)} engine(s)")
        for selection in selected_query_engines.selections:
            engine_name = "SQL" if selection.index == 1 else "Vector"
            logger.info(f"  → {engine_name} Engine: {selection.reason[:80]}...")
        logger.info("=" * 70)

        return QueryEngineSelectionEvent(
            selected_query_engines=selected_query_engines
        )

    # Note: selector is no longer a @step entry point since decompose_query handles entry
    # The _selector_step method is called directly from decompose_query

    async def _coerce_to_response(self, response) -> Response:
        """Normalize different response types into a standard Response."""
        return await ResponseCoercer.coerce(response)

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

        # Use synthesis strategy (TreeSummarize if available, else combine)
        strategy = SynthesisStrategy(self.summarizer)
        return await strategy.synthesize(query, response_strs, source_nodes)

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
            synthesized = await SynthesisStrategy(self.summarizer).synthesize(
                original_query, response_strs, source_nodes
            )
        else:
            logger.info("Using simple concatenation (no summarizer available)")
            synthesized = Response(response="\n\n".join(response_strs), source_nodes=source_nodes)

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
        query = await ctx.store.get("query")
        selected_query_engines = ev.selected_query_engines

        # If this is a refinement iteration, enhance the query with context following LlamaIndex reflection pattern
        # This ensures the model can use the previous output and feedback to generate an improved result
        refinement_iteration = await ctx.store.get("refinement_iteration", default=0)
        if refinement_iteration > 0:
            reflection_feedback = await ctx.store.get("reflection_feedback", default=None)
            previous_response = await ctx.store.get("previous_response", default=None)

            if reflection_feedback:
                missing_info = reflection_feedback.get("missing_information", [])
                improvements = reflection_feedback.get("suggested_improvements", [])

                # Build enhancement following LlamaIndex reflection pattern: show what was wrong
                enhancement_parts = []
                if missing_info:
                    enhancement_parts.append(f"Missing: {', '.join(missing_info[:2])}")
                if improvements:
                    enhancement_parts.append(f"Improve: {improvements[0]}")
                if previous_response and len(previous_response) < 200:
                    # Include short previous response for context (like LlamaIndex pattern)
                    enhancement_parts.append(f"Previous: {previous_response[:150]}")

                if enhancement_parts:
                    query = f"{query} [Refinement context: {'; '.join(enhancement_parts)}]"
                    logger.debug(f"Enhanced query with reflection context for execution")

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
    ) -> StopEvent | ReflectionCompleteEvent | QueryEngineSelectionEvent:
        """Synthesize final response from query engine results."""
        responses = ev.result
        selected_query_engines = ev.selected_query_engines
        query = await ctx.store.get("query")
        original_query = await ctx.store.get("original_query", default=query)

        self._log_step("STEP: RESPONSE SYNTHESIS", f"Synthesizing {len(responses)} response(s)...")

        # Track engine selection for metadata using enum-based dispatch
        selected_indices = [s.index for s in selected_query_engines.selections]
        await ResponseCountHandler.track_selection(selected_indices, ctx, self)

        # Handle responses based on count using enum-based dispatch
        response_count = ResponseCountHandler.get_count(responses)

        if response_count == ResponseCount.SINGLE:
            # Responses are already coerced to Response in process_response
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
        await ctx.store.set("final_response", final_response_text)
        await ctx.store.set("final_response_obj", final_response)  # Store Response object for ReflectionCompleteEvent

        # Store engine selection info for reflection
        engine_names = []
        for selection in selected_query_engines.selections:
            engine_name = "SQL" if selection.index == 1 else "Vector"
            engine_names.append(engine_name)
        await ctx.store.set("engines_used", engine_names)

        # If reflection is enabled, reflect on the response
        if self.enable_reflection and self.reflection_agent:
            refinement_iteration = await ctx.store.get("refinement_iteration", default=0)
            reflection_result = await self._reflect_step(ctx, original_query, final_response_text, refinement_iteration, final_response)

            # If reflection returns QueryEngineSelectionEvent, it means we're refining
            if isinstance(reflection_result, QueryEngineSelectionEvent):
                return reflection_result

            return reflection_result

        # Return Response object following LlamaIndex best practices
        return StopEvent(result=final_response)

    async def _reflect_step(
        self, ctx: Context, query: str, response: str, iteration: int = 0, response_obj=None
    ) -> ReflectionCompleteEvent | QueryEngineSelectionEvent | StopEvent:
        """
        Internal reflection step with iterative refinement support.

        Args:
            ctx: Workflow context
            query: Original query
            response: Current response
            iteration: Current refinement iteration (max 2 to prevent loops)
            response_obj: Optional Response object with source nodes and metadata
        """
        self._log_step(f"STEP: REFLECTION & QUALITY ASSESSMENT (Iteration {iteration + 1})",
                       "Evaluating response quality...")

        # Get metadata for reflection context - include which engines were used
        engines_used = await ctx.store.get("engines_used", default=[])
        metadata = {
            "engines_used": engines_used,
            "has_vector_db": "Vector" in engines_used,
            "has_sql": "SQL" in engines_used,
        }

        # Include source node count if available (helps reflection understand data sources)
        if response_obj and hasattr(response_obj, 'source_nodes'):
            metadata["source_node_count"] = len(response_obj.source_nodes)
            # Check if source nodes contain vector DB results (they'll have similarity scores)
            vector_sources = sum(1 for node in response_obj.source_nodes
                               if hasattr(node, 'score') and node.score is not None)
            metadata["vector_source_count"] = vector_sources

        reflection_result = await self.reflection_agent.reflect(query, response, metadata)

        reflection_event = ReflectionEvent(
            reflection=reflection_result,
            original_query=query,
            response=response,
        )

        # Store reflection for potential use in refinement
        await ctx.store.set("reflection", reflection_result.model_dump())

        logger.info("✓ Reflection complete:")
        logger.info(f"  - Complete: {reflection_result.is_complete}")
        logger.info(f"  - Accurate: {reflection_result.is_accurate}")
        logger.info(f"  - Confidence: {reflection_result.confidence_score:.2f}")
        logger.info(f"  - Should refine: {reflection_result.should_refine}")

        # If reflection suggests refinement and we haven't exceeded max iterations
        # iteration 0 = first reflection, iteration 1 = first refinement, iteration 2 = second refinement (max)
        if reflection_result.should_refine and iteration < 2:
            self._log_step(f"STEP: QUERY REFINEMENT (Iteration {iteration + 1})",
                          "Generating refined query based on reflection feedback...")

            if reflection_result.suggested_improvements:
                logger.info(f"Suggested improvements: {', '.join(reflection_result.suggested_improvements)}")
            if reflection_result.missing_information:
                logger.info(f"Missing information: {', '.join(reflection_result.missing_information)}")

            # Generate refined query based on missing information and suggestions
            refined_query = await self._generate_refined_query(
                query, response, reflection_result, iteration
            )

            logger.info(f"✓ Refined query: {refined_query}")
            logger.info("=" * 70)
            await ctx.store.set("refined_query", refined_query)
            await ctx.store.set("refinement_iteration", iteration + 1)
            await ctx.store.set("query", refined_query)  # Update query for re-execution

            # Store previous response and reflection feedback for use during refinement execution
            # This ensures sub-questions can be enhanced with context about what was missing
            await ctx.store.set("previous_response", response)
            await ctx.store.set("reflection_feedback", {
                "missing_information": reflection_result.missing_information or [],
                "suggested_improvements": reflection_result.suggested_improvements or [],
                "confidence_score": reflection_result.confidence_score,
            })

            # Re-execute with refined query - go back to decomposition to potentially re-decompose
            # Create a new StartEvent to trigger decomposition
            from llama_index.core.workflow import StartEvent
            return await self.decompose_query(ctx, StartEvent(query=refined_query))

        elif reflection_result.should_refine:
            logger.info("Refinement suggested but max iterations (2) reached, returning current response")

        logger.info("=" * 70)

        # Get the Response object from context (stored during synthesis)
        # If not available, create a Response from the string
        response_obj = await ctx.store.get("final_response_obj", default=None)
        if not response_obj:
            # Fallback: create Response from string (loses source_nodes but maintains compatibility)
            from llama_index.core.base.response.schema import Response
            response_obj = Response(response=response)

        return ReflectionCompleteEvent(
            reflection=reflection_event,
            final_response=response_obj,
        )

    async def _generate_refined_query(
        self, original_query: str, current_response: str, reflection_result, iteration: int
    ) -> str:
        """
        Generate a refined query based on reflection feedback.

        Follows LlamaIndex reflection pattern: explicitly shows what was wrong
        and asks the model to improve based on the previous attempt.
        """
        if not self.llm:
            return original_query

        missing_info = ", ".join(reflection_result.missing_information or [])
        improvements = ", ".join(reflection_result.suggested_improvements or [])

        # Reflection prompt pattern following LlamaIndex best practices
        # Shows the previous response and what was wrong, then asks for improvement
        reflection_section = f"""
You already generated this response to the query:

---------------------
{current_response[:800]}
---------------------

This response was evaluated and found to be:
- Missing information: {missing_info or 'None identified'}
- Suggested improvements: {improvements or 'None identified'}
- Confidence score: {reflection_result.confidence_score:.2f}

The original query was: {original_query}

Try again with a refined query that addresses the missing information and improvements.
CRITICAL: Only refine based on the ORIGINAL query. Do NOT add categories, products, or entities that weren't in the original query.
Stay within the scope of the original query - do not add new categories or entities.
Be specific and focused. Return ONLY the refined query, nothing else."""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a query refinement assistant. Generate concise, focused queries that stay within the original query's scope."),
            ChatMessage(role=MessageRole.USER, content=reflection_section),
        ]

        response_obj = await self.llm.achat(messages, stop=["\n\n", "---"])
        refined = response_obj.message.content.strip()

        # Validate: if refined query is significantly longer or adds many new terms not in original, be cautious
        # This is a general heuristic, not dataset-specific
        original_words = set(original_query.lower().split())
        refined_words = set(refined.lower().split())
        new_words = refined_words - original_words

        # If refined query adds many new significant words (more than 3), it might be hallucinating
        # Filter out common stop words and short words
        significant_new_words = {w for w in new_words if len(w) > 4 and w not in ["that", "this", "with", "from", "about", "which", "their", "there"]}
        if len(significant_new_words) > 3:
            logger.warning(f"Refined query adds many new terms ({len(significant_new_words)}) - may be hallucinating. Using original query.")
            return original_query

        return refined if refined and len(refined) >= 10 else original_query

    @step
    async def reflect(
        self, ctx: Context, ev: ReflectionCompleteEvent
    ) -> StopEvent:
        """
        Final step after reflection - returns the response.

        This step allows for potential refinement loops in the future.
        """
        self._log_step("STEP: FINAL RESPONSE", "Workflow complete - returning final answer")
        # Return Response object following LlamaIndex best practices
        return StopEvent(result=ev.final_response)

