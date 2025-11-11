"""SQL query engine setup."""

from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from ..storage import StorageRepository
from ..exceptions import QueryError
from ..logging import get_logger
from .prompts import CUSTOM_TEXT_TO_SQL_PROMPT, SQL_RESPONSE_SYNTHESIS_PROMPT

logger = get_logger(__name__)


def initialize_sql_engine(
    storage_repo: StorageRepository,
    table_name: str,
    llm,
    embed_model,
    streaming: bool = False,
) -> NLSQLTableQueryEngine:
    """
    Initialize SQL query engine for analytical queries.

    Args:
        storage_repo: Storage repository with database access
        table_name: Name of the table to query
        llm: LLM instance for SQL generation
        embed_model: Embedding model for semantic understanding
        streaming: Whether to enable streaming (default: False)

    Returns:
        Configured NLSQLTableQueryEngine instance

    Raises:
        QueryError: If initialization fails
    """
    try:
        engine = storage_repo.get_sqlalchemy_engine()
        sql_database = SQLDatabase(engine, include_tables=[table_name])

        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[table_name],
            llm=llm,
            embed_model=embed_model,
            text_to_sql_prompt=CUSTOM_TEXT_TO_SQL_PROMPT,
            response_synthesis_prompt=SQL_RESPONSE_SYNTHESIS_PROMPT,
            synthesize_response=True,  # Ensure synthesis happens natively
        )
        logger.info(
            f"SQL query engine initialized for table '{table_name}' (streaming={streaming})"
        )
        return sql_query_engine

    except Exception as e:
        logger.error(f"Failed to initialize SQL query engine: {e}", exc_info=True)
        raise QueryError(f"SQL engine initialization failed: {e}") from e
