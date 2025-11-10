"""Utilities for extracting and processing Response objects."""

from llama_index.core.base.response.schema import Response


def extract_response_text(response: Response) -> str:
    """
    Extract text from a Response object following LlamaIndex best practices.

    With NLSQLTableQueryEngine.response_synthesis_prompt configured, response.response
    should contain the properly synthesized answer. However, if the SQL query failed,
    response.response may contain an error explanation instead of data. In that case,
    we check source_nodes for raw data tuples.

    Args:
        response: Response object from query engine

    Returns:
        Extracted text content
    """
    response_text = None

    # Check response.response first - this is the synthesized answer from LlamaIndex
    if hasattr(response, 'response') and response.response:
        response_text = str(response.response)

        # If response contains SQL error indicators, it's likely an error explanation, not data
        # Check for common error patterns in SQL responses
        error_indicators = [
            "contains an error",
            "SQL query contains an error",
            "invalid",
            "cannot be used",
            "Binder Error",
            "Error:",
            "syntax error",
            "cannot include",
        ]

        # If response looks like an error explanation, check source_nodes for actual data
        if any(indicator.lower() in response_text.lower() for indicator in error_indicators):
            # Check if source_nodes has actual data tuples
            if response.source_nodes:
                node_texts = [
                    node.text if hasattr(node, 'text') else
                    node.get_content() if hasattr(node, 'get_content') else
                    str(node)
                    for node in response.source_nodes
                ]
                if node_texts:
                    combined = "\n".join(node_texts)
                    # Only use source_nodes if it looks like actual data tuples (not errors)
                    if (not any(err in combined for err in ["Error:", "Binder Error"]) and
                        any(c.isdigit() for c in combined[:200]) and
                        ('(' in combined or ',' in combined)):
                        return combined

        # If no error indicators, trust the synthesized response
        return response_text

    # Fallback to response_text if available
    if hasattr(response, 'response_text'):
        return str(response.response_text)

    # For sub-question synthesis, check source_nodes for raw data if response.response is empty
    # This handles cases where we need raw tuples for synthesis
    if response.source_nodes:
        node_texts = [
            node.text if hasattr(node, 'text') else
            node.get_content() if hasattr(node, 'get_content') else
            str(node)
            for node in response.source_nodes
        ]
        if node_texts:
            combined = "\n".join(node_texts)
            # Only use source_nodes if it looks like actual data tuples (not errors)
            if (not any(err in combined for err in ["Error:", "Binder Error"]) and
                any(c.isdigit() for c in combined[:200]) and
                ('(' in combined or ',' in combined)):
                return combined

    # Final fallback
    return str(response)

