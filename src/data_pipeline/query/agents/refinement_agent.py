"""Query refinement agent for improving queries based on reflection feedback."""

from typing import Optional
from llama_index.core.llms import ChatMessage, MessageRole
from ...logging import get_logger
from .reflection_agent import ReflectionResult
from .config import AgenticConfig

logger = get_logger(__name__)


class RefinementAgent:
    """
    Agent that refines queries based on reflection feedback.

    Uses LLM to generate improved queries that address missing information
    and suggested improvements from reflection analysis.

    This is an agent definition - performs a specific task using LLM reasoning.
    """

    def __init__(self, llm, config: Optional[AgenticConfig] = None):
        """
        Initialize query refiner.

        Args:
            llm: LLM instance for query refinement
            config: Optional configuration (uses defaults if not provided)
        """
        self.llm = llm
        self.config = config or AgenticConfig()

    async def refine_query(
        self,
        original_query: str,
        current_response: str,
        reflection_result: ReflectionResult,
    ) -> str:
        """
        Generate a refined query based on reflection feedback.

        Args:
            original_query: The original user query
            current_response: The current response that was evaluated
            reflection_result: Reflection analysis result

        Returns:
            Refined query string, or original query if refinement fails
        """
        if not self.llm:
            return original_query

        missing_info = ", ".join(reflection_result.missing_information or [])
        improvements = ", ".join(reflection_result.suggested_improvements or [])

        # Reflection prompt pattern following LlamaIndex best practices
        response_preview = current_response[: self.config.refinement_context_length]
        reflection_section = f"""
You already generated this response to the query:

---------------------
{response_preview}
---------------------

This response was evaluated and found to be:
- Missing information: {missing_info or "None identified"}
- Suggested improvements: {improvements or "None identified"}
- Confidence score: {reflection_result.confidence_score:.2f}

The original query was: {original_query}

Try again with a refined query that addresses the missing information and improvements.
CRITICAL: Only refine based on the ORIGINAL query. Do NOT add categories, products, or entities that weren't in the original query.
Stay within the scope of the original query - do not add new categories or entities.
Be specific and focused. Return ONLY the refined query, nothing else."""

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a query refinement assistant. Generate concise, focused queries that stay within the original query's scope.",
            ),
            ChatMessage(role=MessageRole.USER, content=reflection_section),
        ]

        response_obj = await self.llm.achat(messages, stop=["\n\n", "---"])
        refined = response_obj.message.content.strip()

        # Validate: if refined query adds many new terms, it might be hallucinating
        if self._has_excessive_new_terms(original_query, refined):
            logger.warning(
                f"Refined query adds many new terms - may be hallucinating. Using original query."
            )
            return original_query

        return refined if refined and len(refined) >= 10 else original_query

    def _has_excessive_new_terms(self, original_query: str, refined_query: str) -> bool:
        """
        Check if refined query adds excessive new terms (potential hallucination).

        Simple heuristic: count significant new words beyond threshold.
        """
        original_words = set(original_query.lower().split())
        refined_words = set(refined_query.lower().split())
        new_words = refined_words - original_words

        # Filter out common stop words and short words
        stop_words = {"that", "this", "with", "from", "about", "which", "their", "there"}
        significant_new_words = {w for w in new_words if len(w) > 4 and w not in stop_words}

        return len(significant_new_words) > self.config.hallucination_detection_threshold
