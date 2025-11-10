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
from llama_index.core.prompts.prompt_type import PromptType
from ..storage import StorageRepository
from ..exceptions import QueryError
from ..logging_config import get_logger
from .routing import RouterQueryEngineWorkflow

logger = get_logger(__name__)

# Custom SQL prompt that clarifies database access and reduces meta-reasoning
CUSTOM_TEXT_TO_SQL_PROMPT = PromptTemplate(
    """You are a SQL query generation assistant. Generate ONLY a valid DuckDB SQL query - no explanations, no reasoning text, no comments.

CRITICAL OUTPUT RULES:
- Output ONLY the SQL query itself
- Do NOT include any text before the SQL query (no "To find...", "We need to...", etc.)
- Do NOT include any text after the SQL query
- Start directly with SELECT, WITH, or another SQL keyword
- End with the SQL statement (no trailing explanations)

SQL Requirements:
- Use only columns that exist in the schema description
- Order results by relevant columns when appropriate
- Never query for all columns - only select relevant ones for the question
- For window functions (RANK, ROW_NUMBER, etc.), use CTEs or subqueries if you need to filter by the window function result
- HAVING clauses cannot directly reference window functions - use WHERE on a subquery/CTE instead

Schema:
{schema}

Question: {query_str}

SQL Query:"""
)

# Custom SQL response synthesis prompt - uses native LlamaIndex format
# This controls how NLSQLTableQueryEngine formats SQL results into natural language
# Required template variables: query_str, sql_query, context_str (must match default exactly)
# Based on: https://github.com/run-llama/llama_index/blob/main/docs/examples/workflow/advanced_text_to_sql.ipynb
SQL_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    """Given an input question, synthesize a response from the query results.

CRITICAL RULES:
- Use ONLY the exact data from the SQL Response provided
- Include ALL information from the SQL Response - do not omit any details
- Present the data clearly with product names, categories, prices, and stock status
- Do NOT make up product names, prices, IDs, or any other information
- If information is missing, state that it's not available - do NOT invent it
- Be direct and factual - no speculation or assumptions

Query: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Response: """,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS,
)

# Custom synthesis prompt for TreeSummarize (sub-question synthesis)
# TreeSummarize formats context internally - we must strictly enforce using only provided data
# The {query_str} placeholder ensures the original query requirements are considered during synthesis
CUSTOM_SYNTHESIS_PROMPT = PromptTemplate(
    """Answer the question using ONLY the information provided in the context below.

IMPORTANT: Pay close attention to the question requirements - ensure your answer matches what was asked.

CRITICAL RULES:
- Use ONLY the exact data from the context provided
- Match the question requirements exactly
- Include ALL information mentioned in the context that answers the question
- Do NOT make up product names, prices, IDs, or any other information
- If information is missing from the context, state that it's not available - do NOT invent it
- Present the data exactly as it appears in the context
- Be direct and factual - no speculation or assumptions

Question: {query_str}

Context: {context_str}

Answer the question using ONLY the data from the context above. Ensure your answer matches the question requirements exactly:"""
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
            response_synthesis_prompt=SQL_RESPONSE_SYNTHESIS_PROMPT,
            synthesize_response=True,  # Ensure synthesis happens natively
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
    enable_sub_questions: bool = True,
    enable_reflection: bool = True,
    reflection_threshold: float = 0.7,
) -> RouterQueryEngineWorkflow:
    """
    Initialize Router Query Engine using LlamaIndex Workflows.

    This provides native LlamaIndex routing with:
    - Streaming event support for reasoning token capture
    - Explicit step-by-step execution
    - Sub-question decomposition for complex queries
    - Reflection for quality improvement
    - Better observability

    Args:
        index: Vector store index for semantic queries
        llm: LLM instance for routing decisions
        similarity_top_k: Number of similar chunks to retrieve
        sql_query_engine: SQL query engine for analytical queries
        reasoning_handler: Optional handler for capturing reasoning tokens
        use_workflow: If True, uses workflow-based router (native LlamaIndex)
        enable_sub_questions: Enable sub-question decomposition (default: True)
        enable_reflection: Enable reflection for quality improvement (default: True)
        reflection_threshold: Confidence threshold for reflection (0-1, default: 0.7)

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
                "questions like 'what is X', 'describe X', 'explain X', or 'tell me about X'. "
                "Also use for: finding items by description, understanding concepts, "
                "exploring relationships between entities. "
                "NOT for: counting, aggregations, distinct values, numerical calculations, "
                "filtering by exact values, or precise data retrieval. "
                "Be concise in your selection reasoning - focus on query type."
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
                "NOT for: descriptions, explanations, similarity searches, or conceptual questions. "
                "Be concise in your selection reasoning - focus on query type."
            ),
        ),
    ]

    # Create summarizer for combining multiple responses with optimized prompt
    summarizer = TreeSummarize(
        llm=llm,
        summary_template=CUSTOM_SYNTHESIS_PROMPT,
    )

    # Create workflow-based router
    # Pass LLM, summarizer, and tools to constructor to avoid storing in context (which gets deepcopied)
    router_workflow = RouterQueryEngineWorkflow(
        reasoning_handler=reasoning_handler,
        llm=llm,  # Store as instance var, not in context (avoids deepcopy issues)
        summarizer=summarizer,  # Store as instance var (contains LLM)
        query_engine_tools=tools,  # Store as instance var (contain LLMs)
        enable_sub_questions=enable_sub_questions,
        enable_reflection=enable_reflection,
        reflection_threshold=reflection_threshold,
        timeout=180,
        verbose=True,
    )

    features = []
    if enable_sub_questions:
        features.append("sub-question decomposition")
    if enable_reflection:
        features.append("reflection")

    feature_str = f" with {', '.join(features)}" if features else ""
    logger.info(
        f"RouterQueryEngineWorkflow initialized with SQL + vector engines "
        f"(workflow-based, native LlamaIndex{feature_str})"
    )
    return router_workflow, tools, summarizer
