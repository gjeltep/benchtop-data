"""
Sub-Question Query Engine for decomposing complex queries.

This module implements the Sub-Question Query Engine pattern, which breaks down
complex analytical queries into simpler sub-questions that can be executed
in parallel and then synthesized into a comprehensive answer.
"""

from typing import List
from pydantic import BaseModel, Field
from llama_index.core.workflow import Event
from llama_index.core.llms import ChatMessage, MessageRole
from ...logging_config import get_logger
from .parsing import extract_field

logger = get_logger(__name__)


class SubQuestion(BaseModel):
    """A single sub-question decomposed from a complex query."""
    question: str = Field(description="The sub-question to execute")
    reasoning: str = Field(description="Why this sub-question is needed")
    requires_sql: bool = Field(description="Whether this sub-question requires SQL analysis")
    requires_semantic: bool = Field(description="Whether this sub-question requires semantic search")


class QueryDecomposition(BaseModel):
    """Result of decomposing a complex query."""
    needs_decomposition: bool = Field(description="Whether the query should be decomposed")
    original_query: str = Field(description="The original query")
    sub_questions: List[SubQuestion] = Field(default_factory=list, description="List of sub-questions")
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

    DECOMPOSITION_PROMPT = """Analyze this query and determine if it needs decomposition into sub-questions.

Query: {query}

Respond ONLY in this exact format (be concise, stop after providing the answer):

needs_decomposition: [true/false]
reasoning: [one sentence explanation]

If true, also provide:
sub_questions:
1. question: [sub-question text]
   requires_sql: [true/false]
   requires_semantic: [true/false]
2. question: [sub-question text]
   requires_sql: [true/false]
   requires_semantic: [true/false]

END"""

    def __init__(self, llm, reasoning_handler=None):
        """
        Initialize the sub-question decomposer.

        Args:
            llm: LLM instance for decomposition reasoning
            reasoning_handler: Optional handler for capturing reasoning tokens
        """
        self.llm = llm
        self.reasoning_handler = reasoning_handler

    async def decompose(self, query: str) -> QueryDecomposition:
        """
        Decompose a query into sub-questions if needed.

        Args:
            query: The original user query

        Returns:
            QueryDecomposition with sub-questions if needed
        """
        if self.reasoning_handler:
            self.reasoning_handler.start_reasoning_log()

        prompt = self.DECOMPOSITION_PROMPT.format(query=query)

        # Use LLM to analyze and decompose
        # Create a temporary LLM with shorter output limit for decomposition
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a concise query analysis assistant. Provide brief, structured responses only."),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        # Use achat with stop sequences to prevent infinite loops
        response = await self.llm.achat(
            messages,
            stop=["END", "\n\n\n", "---"],
        )
        response_text = response.message.content

        # Truncate if too long (safety check)
        if len(response_text) > 2000:
            logger.warning("Decomposition response was very long, truncating")
            response_text = response_text[:2000]

        if self.reasoning_handler:
            self.reasoning_handler.log_complete_reasoning()

        # Parse the response into structured format
        decomposition = self._parse_decomposition(query, response_text)

        logger.info(f"Query decomposition: needs_decomposition={decomposition.needs_decomposition}")
        if decomposition.needs_decomposition:
            logger.info(f"  Decomposed into {len(decomposition.sub_questions)} sub-questions")
            for i, sq in enumerate(decomposition.sub_questions, 1):
                logger.info(f"    {i}. {sq.question} (SQL: {sq.requires_sql}, Semantic: {sq.requires_semantic})")

        return decomposition

    def _parse_decomposition(self, original_query: str, response_text: str) -> QueryDecomposition:
        """
        Parse LLM response into structured QueryDecomposition.

        This uses a simple parsing approach that works with reasoning models.
        For more complex cases, consider using structured outputs or function calling.
        """
        response_lower = response_text.lower()

        # Determine if decomposition is needed
        needs_decomposition = (
            "needs_decomposition: true" in response_text.lower() or
            "should be decomposed" in response_lower or
            "sub_questions" in response_lower or
            "sub-question" in response_lower
        )

        # Extract reasoning
        reasoning = extract_field(response_text, "reasoning", "No reasoning provided",
                                stop_patterns=["\n-", "\n*", "\nneeds_", "\nsub_", "\nif "])

        # Extract sub-questions if decomposition is needed
        sub_questions = []
        if needs_decomposition:
            sub_questions = self._extract_sub_questions(response_text)

        # If no sub-questions were extracted but decomposition was indicated,
        # create a single sub-question from the original query
        if needs_decomposition and not sub_questions:
            # Try to infer requirements from the query
            query_lower = original_query.lower()
            requires_sql = any(keyword in query_lower for keyword in [
                "count", "sum", "average", "total", "max", "min", "group by",
                "how many", "what is the total", "calculate", "aggregate"
            ])
            requires_semantic = any(keyword in query_lower for keyword in [
                "what is", "describe", "similar", "like", "example", "explain"
            ])

            sub_questions = [SubQuestion(
                question=original_query,
                reasoning="Original query as single sub-question",
                requires_sql=requires_sql,
                requires_semantic=requires_semantic or not requires_sql,
            )]

        return QueryDecomposition(
            needs_decomposition=needs_decomposition,
            original_query=original_query,
            sub_questions=sub_questions,
            reasoning=reasoning,
        )

    def _extract_sub_questions(self, text: str) -> List[SubQuestion]:
        """Extract sub-questions from the response text."""
        sub_questions = []

        # Look for numbered or bulleted sub-questions
        lines = text.split("\n")
        current_question = None
        current_reasoning = ""
        current_requires_sql = False
        current_requires_semantic = False

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line starts a new sub-question
            if any(marker in line_lower for marker in ["question:", "sub-question:", "1.", "2.", "3.", "4.", "5."]):
                # Save previous question if exists
                if current_question:
                    sub_questions.append(SubQuestion(
                        question=current_question,
                        reasoning=current_reasoning or "Extracted from decomposition",
                        requires_sql=current_requires_sql,
                        requires_semantic=current_requires_semantic,
                    ))

                # Extract question text
                for marker in ["question:", "sub-question:"]:
                    if marker in line_lower:
                        current_question = line.split(":", 1)[1].strip()
                        break
                else:
                    # Numbered list
                    parts = line.split(".", 1)
                    if len(parts) > 1:
                        current_question = parts[1].strip()

                current_reasoning = ""
                current_requires_sql = False
                current_requires_semantic = False

            # Extract metadata
            elif "reasoning:" in line_lower:
                current_reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            elif "requires_sql:" in line_lower or "sql:" in line_lower:
                current_requires_sql = "true" in line_lower
            elif "requires_semantic:" in line_lower or "semantic:" in line_lower:
                current_requires_semantic = "true" in line_lower

        # Add final question
        if current_question:
            sub_questions.append(SubQuestion(
                question=current_question,
                reasoning=current_reasoning or "Extracted from decomposition",
                requires_sql=current_requires_sql,
                requires_semantic=current_requires_semantic,
            ))

        return sub_questions

