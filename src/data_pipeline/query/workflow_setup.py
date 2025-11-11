from typing import List, Optional, Tuple, Literal
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import TreeSummarize
from ..exceptions import QueryError
from ..logging import get_logger
from ..config import Config
from .routing.react_agent import create_react_agent_workflow
from .routing.custom_router import create_custom_router_workflow
from .routing.workflow_protocol import QueryWorkflow
from .agents.config import AgenticConfig
from .prompts import CUSTOM_SYNTHESIS_PROMPT

WorkflowType = Literal["react_agent", "custom_router"]

logger = get_logger(__name__)


def _create_query_engine_tools(
    index: VectorStoreIndex,
    sql_query_engine: NLSQLTableQueryEngine,
    llm,
    similarity_top_k: int,
) -> List[QueryEngineTool]:
    """
    Create query engine tools for routing.

    Shared between all workflow types.

    Args:
        index: Vector store index for semantic queries
        sql_query_engine: SQL query engine for analytical queries
        llm: LLM instance
        similarity_top_k: Number of similar chunks to retrieve

    Returns:
        List of QueryEngineTool instances
    """
    # Create vector query engine
    vector_engine = index.as_query_engine(
        llm=llm, similarity_top_k=similarity_top_k, streaming=True
    )

    # Create tools for routing with enhanced descriptions
    return [
        QueryEngineTool.from_defaults(
            query_engine=vector_engine,
            description=(
                "Useful for semantic queries, similarity searches, and descriptive questions. "
                "Use for: similar items, descriptions, examples, finding related concepts, "
                "questions like 'what is X', 'describe X', 'explain X', or 'tell me about X'. "
                "Also use for: finding items by description, understanding concepts, "
                "exploring relationships between entities. "
                "NOT for: counting, aggregations, distinct values, numerical calculations, "
                "filtering by exact values, or precise data retrieval."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description=(
                "Useful for analytical queries involving numbers, filtering, or precise data retrieval. "
                "Use for: counting, aggregating (totals, averages, min/max, sums), "
                "filtering by values (e.g., 'out of stock', 'price > 100'), "
                "grouping by categories, comparisons, distinct values, "
                "top N queries, sorting, or any question requiring exact data matching. "
                "Also use for: 'how many', 'what are the distinct', 'list all X where Y', "
                "'top N by', 'group by', or queries asking for specific numerical results. "
                "NOT for: descriptions, explanations, similarity searches, or conceptual questions."
            ),
        ),
    ]


def create_query_workflow(
    workflow_type: WorkflowType,
    index: VectorStoreIndex,
    sql_query_engine: NLSQLTableQueryEngine,
    llm,
    similarity_top_k: int,
    reasoning_handler=None,
    reflection_threshold: float = 0.7,
    request_timeout: float = 180.0,
    context_window: int = 32768,
    num_output: int = 1024,
    config: Optional[Config] = None,
) -> Tuple["QueryWorkflow", List[QueryEngineTool], Optional[TreeSummarize]]:
    """
    Create a query workflow based on type.

    Unified factory function that eliminates branching logic by using
    separate workflow factories for each type.

    Args:
        workflow_type: "react_agent" or "custom_router"
        index: Vector store index for semantic queries
        sql_query_engine: SQL query engine for analytical queries
        llm: LLM instance for routing decisions
        similarity_top_k: Number of similar chunks to retrieve
        reasoning_handler: Optional handler for capturing reasoning tokens
        reflection_threshold: Confidence threshold for reflection (0-1, default: 0.7)
        request_timeout: Request timeout in seconds
        context_window: Context window size
        num_output: Maximum output tokens
        config: Optional configuration

    Returns:
        Tuple of (workflow, tools, summarizer)
        - workflow: QueryWorkflow instance (implements common protocol)
        - tools: Query engine tools
        - summarizer: Response synthesizer (None if using ReActAgent)

    Note:
        custom_router always includes sub-question decomposition and reflection.
        react_agent provides simpler routing without these features.
    """
    if not index:
        raise QueryError("Cannot initialize router: index is required")

    if not sql_query_engine:
        raise QueryError("Cannot initialize router: SQL engine is required")

    # Create tools (shared between both workflows)
    tools = _create_query_engine_tools(index, sql_query_engine, llm, similarity_top_k)

    # Create workflow based on type
    if workflow_type == "react_agent":
        llm_config = {
            "model_name": config.llm_model,
            "base_url": config.ollama_url,
            "request_timeout": request_timeout,
            "context_window": context_window,
            "num_output": num_output,
        }
        agentic_config = AgenticConfig()
        workflow = create_react_agent_workflow(tools, llm_config, agentic_config)
        return workflow, tools, None

    elif workflow_type == "custom_router":
        summarizer = TreeSummarize(llm=llm, summary_template=CUSTOM_SYNTHESIS_PROMPT)
        workflow = create_custom_router_workflow(
            tools=tools,
            llm=llm,
            reasoning_handler=reasoning_handler,
            summarizer=summarizer,
            reflection_threshold=reflection_threshold,
            request_timeout=request_timeout,
        )
        return workflow, tools, summarizer

    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

