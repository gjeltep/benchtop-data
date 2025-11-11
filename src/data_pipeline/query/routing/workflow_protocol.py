"""Protocol definition for query workflows."""

from typing import Protocol, Optional
from llama_index.core.base.response.schema import Response


class QueryWorkflow(Protocol):
    """
    Protocol defining the common interface for all query workflows.

    This allows different workflow implementations (ReActAgent, custom router, etc.)
    to be used interchangeably without branching logic.
    """

    async def run(self, query: str) -> Response:
        """
        Execute a query and return a Response.

        Args:
            query: The user query string

        Returns:
            Response object with answer and metadata
        """
        ...

    @property
    def selected_engine_index(self) -> Optional[int]:
        """
        Get the index of the selected engine (0=vector, 1=SQL).

        Returns:
            Engine index if available, None otherwise
        """
        ...
