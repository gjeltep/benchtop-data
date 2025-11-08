"""
Embeddings module for Chroma vector storage.

Note: This module provides basic ChromaDB client initialization.
Actual embedding generation and indexing is handled by LlamaIndex in the query module.
"""

from typing import Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class ChromaRepository:
    """ChromaDB repository for vector storage client initialization."""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize Chroma repository.

        Args:
            persist_directory: Directory to persist Chroma data. If None, uses in-memory.
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))

    def create_collection(
        self, collection_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection
        """
        self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata or {"description": "Vector embeddings collection"},
        )

    def get_client(self):
        """
        Get the ChromaDB client.

        Returns:
            ChromaDB client object
        """
        return self.client
