"""
Reflection Agent for validating and improving query results.

This module implements the Reflection pattern, where the system self-critiques
its responses to improve quality, catch errors, and suggest improvements.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from llama_index.core.workflow import Event
from llama_index.core.llms import ChatMessage, MessageRole
from ...logging import get_logger
from .parsing import extract_field, extract_boolean, extract_float, extract_list
from .config import AgenticConfig

logger = get_logger(__name__)


# Type alias for query metadata (moved here to avoid circular import)
QueryMetadata = Dict[str, Any]


class ReflectionResult(BaseModel):
    """Result of reflection on a query response."""
    is_complete: bool = Field(description="Whether the response fully answers the query")
    is_accurate: bool = Field(description="Whether the response appears accurate")
    missing_information: List[str] = Field(default_factory=list, description="Information that seems missing")
    suggested_improvements: List[str] = Field(default_factory=list, description="Suggestions for improving the response")
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

    REFLECTION_PROMPT = """Evaluate this query response. Be concise.

Query: {query}
Response: {response}
{engine_context}

VALIDATION RULES:
1. Verify the response matches the query requirements (within dataset limitations) exactly
2. If the query requirements are not met, add context as to how the response completeness, accuracy, and/ or quality can be improved.

Respond ONLY in this format (stop after providing the answer):

is_complete: [true/false]
is_accurate: [true/false]
confidence_score: [0.0-1.0]
should_refine: [true/false]
reasoning: [one sentence]
missing_information: [list if any - only include if information is truly missing from the dataset]
suggested_improvements: [list if any - do not suggest adding categories or entities not in the original query]

END"""

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

        prompt = self.REFLECTION_PROMPT.format(
            query=query,
            response=response,
            engine_context=engine_context,
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a concise quality assurance assistant. Provide brief, structured responses only."),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        # Use achat with stop sequences to prevent infinite loops
        response_obj = await self.llm.achat(
            messages,
            stop=["END", "\n\n\n", "---"],
        )
        response_text = response_obj.message.content

        # Truncate if too long (safety check)
        if len(response_text) > self.config.max_response_length:
            logger.warning("Reflection response was very long, truncating")
            response_text = response_text[:self.config.max_response_length]

        if self.reasoning_handler:
            self.reasoning_handler.log_complete_reasoning()

        # Parse reflection result
        reflection = self._parse_reflection(response_text)

        logger.info(f"Reflection: complete={reflection.is_complete}, accurate={reflection.is_accurate}, "
                   f"confidence={reflection.confidence_score:.2f}, refine={reflection.should_refine}")
        if reflection.missing_information:
            logger.info(f"  Missing: {', '.join(reflection.missing_information)}")
        if reflection.suggested_improvements:
            logger.info(f"  Improvements: {', '.join(reflection.suggested_improvements)}")

        return reflection

    def _parse_reflection(self, text: str) -> ReflectionResult:
        """Parse LLM response into structured ReflectionResult."""
        # Extract fields using shared parsing utilities
        is_complete = extract_boolean(text, "is_complete", True)
        is_accurate = extract_boolean(text, "is_accurate", True)
        should_refine = extract_boolean(text, "should_refine", False)
        confidence_score = extract_float(text, "confidence_score", 0.8)
        reasoning = extract_field(text, "reasoning", "No reasoning provided",
                                 stop_patterns=["\n-", "\n*", "\nmissing_", "\nsuggested_", "\nis_", "\nconfidence_", "\nshould_"])
        missing_info = extract_list(text, "missing_information")
        improvements = extract_list(text, "suggested_improvements")

        # Auto-adjust should_refine based on confidence if not explicitly set
        if not should_refine and confidence_score < self.reflection_threshold:
            should_refine = True
            if not reasoning or "confidence" not in reasoning.lower():
                reasoning += f" Low confidence score ({confidence_score:.2f}) suggests refinement may be needed."

        return ReflectionResult(
            is_complete=is_complete,
            is_accurate=is_accurate,
            missing_information=missing_info,
            suggested_improvements=improvements,
            confidence_score=confidence_score,
            reasoning=reasoning,
            should_refine=should_refine,
        )
