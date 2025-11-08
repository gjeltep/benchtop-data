"""
Query engine initialization helpers.

Handles initialization of SQL and RouterQueryEngine components for hybrid querying.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from ..storage import StorageRepository
from ..exceptions import QueryError
from ..logging_config import get_logger

logger = get_logger(__name__)


def initialize_sql_engine(
    storage_repo: StorageRepository, table_name: str, llm, embed_model
) -> NLSQLTableQueryEngine:
    try:
        engine = storage_repo.get_sqlalchemy_engine()
        sql_database = SQLDatabase(engine, include_tables=[table_name])

        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, tables=[table_name], llm=llm, embed_model=embed_model
        )
        logger.info(f"SQL query engine initialized for table '{table_name}'")
        return sql_query_engine

    except Exception as e:
        logger.error(f"Failed to initialize SQL query engine: {e}", exc_info=True)
        raise QueryError(f"SQL engine initialization failed: {e}") from e


def initialize_router(
    index: VectorStoreIndex, llm, similarity_top_k: int, sql_query_engine: NLSQLTableQueryEngine
) -> RouterQueryEngine:
    if not index:
        raise QueryError("Cannot initialize router: index is required")

    if not sql_query_engine:
        raise QueryError("Cannot initialize router: SQL engine is required")

    # Create vector query engine
    vector_engine = index.as_query_engine(llm=llm, similarity_top_k=similarity_top_k)

    # Create tools for routing
    tools = [
        QueryEngineTool.from_defaults(
            query_engine=vector_engine,
            description=(
                "Useful for semantic queries, similarity searches, and descriptive questions. "
                "Use for: similar items, descriptions, examples, finding related concepts."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description=(
                "Useful for analytical queries: counting, aggregating, filtering, comparing numbers. "
                "Use for: totals, averages, counts, max/min, grouping, filtering."
            ),
        ),
    ]

    # Create router with both engines
    router = RouterQueryEngine(
        query_engine_tools=tools,
        selector=LLMSingleSelector.from_defaults(llm=llm),
        llm=llm,
        verbose=True,
    )
    logger.info("RouterQueryEngine initialized with SQL + vector engines")
    return router
