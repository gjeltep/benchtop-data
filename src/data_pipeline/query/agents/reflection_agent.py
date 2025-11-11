"""
Reflection Agent for validating and improving query results.

This module implements the Reflection pattern, where the system self-critiques
its responses to improve quality, catch errors, and suggest improvements.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from llama_index.core.workflow import Event
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from ...logging import get_logger
from .config import AgenticConfig

logger = get_logger(__name__)


# Type alias for query metadata (moved here to avoid circular import)
QueryMetadata = Dict[str, Any]


class ReflectionResult(BaseModel):
    """Result of reflection on a query response."""

    is_complete: bool = Field(description="Whether the response fully answers the query")
    is_accurate: bool = Field(description="Whether the response appears accurate")
    missing_information: List[str] = Field(
        default_factory=list, description="Information that seems missing"
    )
    suggested_improvements: List[str] = Field(
        default_factory=list, description="Suggestions for improving the response"
    )
    confidence_score: float = Field(description="Confidence in the response quality (0-1)")
    reasoning: str = Field(description="Explanation of the reflection analysis")
    should_refine: bool = Field(description="Whether the query should be refined and re-executed")


class ReflectionEvent(Event):
    """Event containing reflection results."""

    reflection: ReflectionResult
    original_query: str
    response: str


class ReflectionAgent:
    """
    Reflection agent that validates and critiques query responses.

    This agent analyzes responses for completeness, accuracy, and quality,
    and can suggest improvements or trigger query refinement.
    """

    REFLECTION_PROMPT_TEMPLATE = PromptTemplate(
        """Evaluate this query response. Be concise and provide structured assessment.

Query: {query}
Response: {response}
{engine_context}

VALIDATION RULES:
1. Verify the response matches the query requirements (within dataset limitations) exactly
2. If the query requirements are not met, explain what's missing or could be improved

Provide your evaluation as a structured response with these fields:
- is_complete: Whether the response fully answers the query
- is_accurate: Whether the response appears accurate based on the data
- confidence_score: Your confidence in the response quality (0.0 to 1.0)
- should_refine: Whether the query should be refined and re-executed
- reasoning: One sentence explanation of your assessment
- missing_information: List of information missing from the response (empty list if none)
- suggested_improvements: List of suggestions to improve the response (empty list if none)

Note: Only suggest improvements that stay within the scope of the original query.
"""
    )

    def __init__(self, llm, reasoning_handler=None, reflection_threshold: float = 0.7):
        """
        Initialize the reflection agent.

        Args:
            llm: LLM instance for reflection reasoning
            reasoning_handler: Optional handler for capturing reasoning tokens
            reflection_threshold: Confidence threshold below which refinement is suggested (0-1)
        """
        self.llm = llm
        self.reasoning_handler = reasoning_handler
        self.reflection_threshold = reflection_threshold
        self.config = AgenticConfig()

    async def reflect(
        self,
        query: str,
        response: str,
        metadata: Optional[QueryMetadata] = None,
    ) -> ReflectionResult:
        """
        Reflect on a query response to evaluate its quality.

        Uses LlamaIndex structured outputs to get validated Pydantic responses.

        Args:
            query: The original user query
            response: The generated response
            metadata: Optional metadata about the query execution

        Returns:
            ReflectionResult with evaluation and suggestions
        """
        if self.reasoning_handler:
            self.reasoning_handler.start_reasoning_log()

        # Build engine context for reflection prompt
        engine_context = ""
        if metadata:
            engines_used = metadata.get("engines_used", [])
            if engines_used:
                engine_context = f"\n\nEngines used: {', '.join(engines_used)}"
                if metadata.get("has_vector_db"):
                    vector_count = metadata.get("vector_source_count", 0)
                    if vector_count > 0:
                        engine_context += f" (Vector DB returned {vector_count} sources)"
                if metadata.get("has_sql"):
                    engine_context += " (SQL database queried)"
                if metadata.get("error"):
                    engine_context += f"\nError occurred: {metadata['error']}"

        # Use structured prediction with LlamaIndex
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=ReflectionResult,
            llm=self.llm,
            prompt_template_str=str(self.REFLECTION_PROMPT_TEMPLATE.template),
            verbose=False,
        )

        # Get structured output directly
        reflection = await program.acall(
            query=query,
            response=response,
            engine_context=engine_context,
        )

        if self.reasoning_handler:
            self.reasoning_handler.log_complete_reasoning()

        # Auto-adjust should_refine based on confidence threshold
        if not reflection.should_refine and reflection.confidence_score < self.reflection_threshold:
            reflection.should_refine = True
            if not reflection.reasoning or "confidence" not in reflection.reasoning.lower():
                reflection.reasoning += f" Low confidence score ({reflection.confidence_score:.2f}) suggests refinement may be needed."

        logger.info(
            f"Reflection: complete={reflection.is_complete}, accurate={reflection.is_accurate}, "
            f"confidence={reflection.confidence_score:.2f}, refine={reflection.should_refine}"
        )
        if reflection.missing_information:
            logger.info(f"  Missing: {', '.join(reflection.missing_information)}")
        if reflection.suggested_improvements:
            logger.info(f"  Improvements: {', '.join(reflection.suggested_improvements)}")

        return reflection
