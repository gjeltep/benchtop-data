from typing import Optional
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.base.response.schema import Response
from ..workflow_protocol import QueryWorkflow
from ....logging import get_logger

logger = get_logger(__name__)


class ReActAgentWorkflow:
    """
    ReActAgent workflow implementation.

    Wrapper around LlamaIndex ReActAgent to match QueryWorkflow protocol.
    Uses LlamaIndex's ReActAgent for query execution.
    """

    def __init__(self, agent: ReActAgent):
        """
        Initialize ReActAgent workflow wrapper.

        Args:
            agent: ReActAgent instance
        """
        self.agent = agent
        self._selected_engine_index = None  # ReActAgent doesn't track this

    async def run(self, query: str) -> Response:
        """
        Execute query using ReActAgent.

        Args:
            query: User query string

        Returns:
            Response object with answer
        """
        ctx = Context(self.agent)
        handler = self.agent.run(query, ctx=ctx)
        result = await handler

        # Extract response from AgentOutput
        if hasattr(result, "response"):
            content = (
                result.response.content
                if hasattr(result.response, "content")
                else str(result.response)
            )
        else:
            content = str(result)

        return Response(response=content)

    @property
    def selected_engine_index(self) -> Optional[int]:
        """ReActAgent doesn't track selected engine index."""
        return self._selected_engine_index
