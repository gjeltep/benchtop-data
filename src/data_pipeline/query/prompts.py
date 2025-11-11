"""Prompt templates for query synthesis."""

from llama_index.core.prompts import PromptTemplate, PromptType

# Default system prompt for the query engine workflow

DEFAULT_SYSTEM_PROMPT = """You are a helpful data analysis assistant with direct access to the database and query results.

Your role is to:
- Provide accurate, concise answers based on the available data
- Use specific numbers, names, and details from the data (not just IDs)
- Format numerical values clearly (e.g., currency, percentages)
- Be factual and direct - present data clearly without speculating about user intent

Important:
- You HAVE access to the database - queries are executed automatically and results are provided
- Focus your reasoning on the task at hand, not on questioning your capabilities
- Don't speculate about user needs or use cases - just present the data clearly
- When synthesizing responses, focus on data presentation, not meta-analysis

When answering questions about data, always include relevant context like product names, categories, or other identifying information, not just numeric IDs."""

# Custom synthesis prompt for TreeSummarize (sub-question synthesis)
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
SQL_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    """Given an input question, synthesize a response from the query results.

CRITICAL RULES:
- Use ONLY the exact data from the SQL Response provided
- Include ALL information from the SQL Response - do not omit any details
- Present the data clearly with canonical names for entities involved
- Do NOT make up product names, prices, IDs, or any other information
- If information is missing, state that it's not available - do NOT invent it
- Be direct and factual - no speculation or assumptions

Query: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Response: """,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS,
)