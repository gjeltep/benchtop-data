"""
Query engine module using LlamaIndex and Ollama.

Uses LlamaIndex's built-in capabilities for vector search and simplified SQL querying.
"""

from .engine import QueryEngine

__all__ = ["QueryEngine"]
