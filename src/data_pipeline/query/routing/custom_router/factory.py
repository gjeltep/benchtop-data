"""Custom router workflow wrapper and factory."""

from typing import List, Optional
from llama_index.core.tools import QueryEngineTool
from llama_index.core.base.response.schema import Response
from llama_index.core.response_synthesizers import TreeSummarize
from .router import RouterQueryEngineWorkflow
from ..workflow_protocol import QueryWorkflow
from ....logging import get_logger

logger = get_logger(__name__)


class CustomRouterWorkflow:
    """
    Custom router workflow implementation.

    Wrapper around RouterQueryEngineWorkflow to match QueryWorkflow protocol.
    Uses custom LlamaIndex Workflow with sub-question decomposition and reflection.
    """

    def __init__(self, workflow: RouterQueryEngineWorkflow):
        """
        Initialize custom router workflow wrapper.

        Args:
            workflow: RouterQueryEngineWorkflow instance
        """
        self.workflow = workflow

    async def run(self, query: str) -> Response:
        """
        Execute query using custom router workflow.

        Args:
            query: User query string

        Returns:
            Response object with answer and metadata
        """
        result = await self.workflow.run(query=query, select_multi=False)

        # Extract Response from StopEvent
        if hasattr(result, 'result'):
            response = result.result
            if isinstance(response, Response):
                return response
            return Response(response=str(response))

        return Response(response=str(result))

    @property
    def selected_engine_index(self) -> Optional[int]:
        """Get selected engine index from underlying workflow."""
        return self.workflow.selected_engine_index


def create_custom_router_workflow(
    tools: List[QueryEngineTool],
    llm,
    reasoning_handler,
    summarizer: TreeSummarize,
    reflection_threshold: float = 0.7,
    request_timeout: float = 180.0,
) -> CustomRouterWorkflow:
    """
    Factory function to create custom router workflow.

    Creates advanced workflow with sub-question decomposition and reflection
    always enabled. For simpler routing, use react_agent workflow instead.

    Args:
        tools: Query engine tools
        llm: LLM instance
        reasoning_handler: Optional reasoning token handler
        summarizer: Response synthesizer
        reflection_threshold: Confidence threshold for reflection
        request_timeout: Request timeout in seconds

    Returns:
        CustomRouterWorkflow instance
    """
    workflow = RouterQueryEngineWorkflow(
        reasoning_handler=reasoning_handler,
        llm=llm,
        summarizer=summarizer,
        query_engine_tools=tools,
        reflection_threshold=reflection_threshold,
        timeout=request_timeout,
        verbose=True,
    )

    logger.info(
        "Custom router workflow created (workflow-based, native LlamaIndex "
        "with sub-question decomposition, reflection)"
    )

    return CustomRouterWorkflow(workflow)

