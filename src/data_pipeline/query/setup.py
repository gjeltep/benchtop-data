"""
Query engine setup helpers.

Handles initialization of SQL and Router components for hybrid querying.
Uses LlamaIndex Workflows for native routing with reasoning token capture.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.prompts import PromptTemplate
from ..storage import StorageRepository
from ..exceptions import QueryError
from ..logging_config import get_logger
from .routing import RouterQueryEngineWorkflow

logger = get_logger(__name__)

# Custom SQL prompt that clarifies database access and reduces meta-reasoning
CUSTOM_TEXT_TO_SQL_PROMPT = PromptTemplate(
    """You are a SQL query generation assistant with direct access to the database.

Your task:
1. Analyze the user's question
2. Generate a syntactically correct DuckDB SQL query
3. The system will automatically execute your query and provide results

Important:
- You HAVE database access - the system executes queries automatically and provides results
- Focus your reasoning on SQL correctness, query structure, and column selection
- Be concise in your reasoning - avoid repetitive "But note:" or "However:" statements
- Don't question whether you have access - you do
- Don't speculate about query results - you'll receive them automatically
- Use only columns that exist in the schema description
- Order results by relevant columns when appropriate
- Never query for all columns - only select relevant ones for the question

Reasoning guidelines:
- Think step-by-step but keep it brief
- Focus on: which columns to select, what conditions to filter, how to order
- Avoid meta-commentary about the process itself
- Trust that the system will execute your query correctly
- Don't repeat yourself or add unnecessary caveats
- If you've decided on an approach, proceed - don't second-guess with "But" statements

Schema:
{schema}

Question: {query_str}

Generate the SQL query:"""
)


def initialize_sql_engine(
    storage_repo: StorageRepository, table_name: str, llm, embed_model, streaming: bool = False
) -> NLSQLTableQueryEngine:
    try:
        engine = storage_repo.get_sqlalchemy_engine()
        sql_database = SQLDatabase(engine, include_tables=[table_name])

        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[table_name],
            llm=llm,
            embed_model=embed_model,
            text_to_sql_prompt=CUSTOM_TEXT_TO_SQL_PROMPT,
        )
        logger.info(f"SQL query engine initialized for table '{table_name}' (streaming={streaming})")
        return sql_query_engine

    except Exception as e:
        logger.error(f"Failed to initialize SQL query engine: {e}", exc_info=True)
        raise QueryError(f"SQL engine initialization failed: {e}") from e


def initialize_router_workflow(
    index: VectorStoreIndex,
    llm,
    similarity_top_k: int,
    sql_query_engine: NLSQLTableQueryEngine,
    reasoning_handler=None,
    use_workflow: bool = True,
) -> RouterQueryEngineWorkflow:
    """
    Initialize Router Query Engine using LlamaIndex Workflows.

    This provides native LlamaIndex routing with:
    - Streaming event support for reasoning token capture
    - Explicit step-by-step execution
    - Better observability

    Args:
        index: Vector store index for semantic queries
        llm: LLM instance for routing decisions
        similarity_top_k: Number of similar chunks to retrieve
        sql_query_engine: SQL query engine for analytical queries
        reasoning_handler: Optional handler for capturing reasoning tokens
        use_workflow: If True, uses workflow-based router (native LlamaIndex)

    Returns:
        RouterQueryEngineWorkflow instance
    """
    if not index:
        raise QueryError("Cannot initialize router: index is required")

    if not sql_query_engine:
        raise QueryError("Cannot initialize router: SQL engine is required")

    # Create vector query engine
    # Note: Workflows support streaming natively, so we can enable it
    vector_engine = index.as_query_engine(
        llm=llm, similarity_top_k=similarity_top_k, streaming=True
    )

    # Create tools for routing with enhanced descriptions
    # These descriptions guide the router to make efficient decisions
    tools = [
        QueryEngineTool.from_defaults(
            query_engine=vector_engine,
            description=(
                "Useful for semantic queries, similarity searches, and descriptive questions. "
                "Use for: similar items, descriptions, examples, finding related concepts, "
                "questions like 'what is X' or 'describe X'. "
                "Be concise in your selection reasoning - focus on query type."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description=(
                "Useful for analytical queries involving numbers, filtering, or precise data retrieval. "
                "Use for: counting, aggregating (totals, averages, min/max), filtering by values "
                "(e.g., 'out of stock', 'price > 100'), grouping, comparisons, or any question "
                "requiring exact data matching. "
                "Be concise in your selection reasoning - focus on query type."
            ),
        ),
    ]

    # Create summarizer for combining multiple responses
    summarizer = TreeSummarize(llm=llm)

    # Create workflow-based router
    # Pass LLM, summarizer, and tools to constructor to avoid storing in context (which gets deepcopied)
    router_workflow = RouterQueryEngineWorkflow(
        reasoning_handler=reasoning_handler,
        llm=llm,  # Store as instance var, not in context (avoids deepcopy issues)
        summarizer=summarizer,  # Store as instance var (contains LLM)
        query_engine_tools=tools,  # Store as instance var (contain LLMs)
        timeout=180,
        verbose=True,
    )

    logger.info(
        f"RouterQueryEngineWorkflow initialized with SQL + vector engines "
        f"(workflow-based, native LlamaIndex)"
    )
    return router_workflow, tools, summarizer
