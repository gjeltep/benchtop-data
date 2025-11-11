"""
Sub-Question Query Engine for decomposing complex queries.

This module implements the Sub-Question Query Engine pattern, which breaks down
complex analytical queries into simpler sub-questions that can be executed
in parallel and then synthesized into a comprehensive answer.
"""

from typing import List
from pydantic import BaseModel, Field
from llama_index.core.workflow import Event
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from ...logging import get_logger
from .config import AgenticConfig

logger = get_logger(__name__)


class SubQuestion(BaseModel):
    """A single sub-question decomposed from a complex query."""

    question: str = Field(description="The sub-question to execute")
    reasoning: str = Field(description="Why this sub-question is needed")
    requires_sql: bool = Field(description="Whether this sub-question requires SQL analysis")
    requires_semantic: bool = Field(
        description="Whether this sub-question requires semantic search"
    )


class QueryDecomposition(BaseModel):
    """Result of decomposing a complex query."""

    needs_decomposition: bool = Field(description="Whether the query should be decomposed")
    original_query: str = Field(description="The original query")
    sub_questions: List[SubQuestion] = Field(
        default_factory=list, description="List of sub-questions"
    )
    reasoning: str = Field(description="Explanation of why decomposition was or wasn't needed")


class SubQuestionEvent(Event):
    """Event containing decomposed sub-questions."""

    decomposition: QueryDecomposition
    original_query: str


class SubQuestionDecomposer:
    """
    Decomposes complex queries into sub-questions using a reasoning model.

    This enables handling multi-part analytical queries by breaking them down
    into simpler questions that can be executed independently and then synthesized.
    """

    DECOMPOSITION_PROMPT_TEMPLATE = PromptTemplate(
        """Analyze this query and determine if it needs decomposition into sub-questions.

Query: {query}

Provide a structured analysis with these fields:
- needs_decomposition: Whether the query should be broken down (true/false)
- reasoning: One sentence explanation of your decision
- original_query: The query being analyzed
- sub_questions: List of sub-questions (empty list if not needed)

For each sub-question, provide:
- question: The sub-question text
- reasoning: Why this sub-question is needed
- requires_sql: Whether it needs SQL for aggregations/filtering (true/false)
- requires_semantic: Whether it needs semantic search for descriptions (true/false)

Be concise. Only decompose if the query truly has multiple distinct parts that benefit from parallel execution.
"""
    )

    def __init__(self, llm, reasoning_handler=None):
        """
        Initialize the sub-question decomposer.

        Args:
            llm: LLM instance for decomposition reasoning
            reasoning_handler: Optional handler for capturing reasoning tokens
        """
        self.llm = llm
        self.reasoning_handler = reasoning_handler
        self.config = AgenticConfig()

    async def decompose(self, query: str) -> QueryDecomposition:
        """
        Decompose a query into sub-questions if needed.

        Uses LlamaIndex structured outputs to get validated Pydantic responses.

        Args:
            query: The original user query

        Returns:
            QueryDecomposition with sub-questions if needed
        """
        if self.reasoning_handler:
            self.reasoning_handler.start_reasoning_log()

        # Use structured prediction with LlamaIndex
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=QueryDecomposition,
            llm=self.llm,
            prompt_template_str=str(self.DECOMPOSITION_PROMPT_TEMPLATE.template),
            verbose=False,
        )

        # Get structured output directly
        decomposition = await program.acall(query=query)

        if self.reasoning_handler:
            self.reasoning_handler.log_complete_reasoning()

        logger.info(f"Query decomposition: needs_decomposition={decomposition.needs_decomposition}")
        if decomposition.needs_decomposition:
            logger.info(f"  Decomposed into {len(decomposition.sub_questions)} sub-questions")
            for i, sq in enumerate(decomposition.sub_questions, 1):
                logger.info(
                    f"    {i}. {sq.question} (SQL: {sq.requires_sql}, Semantic: {sq.requires_semantic})"
                )

        return decomposition
