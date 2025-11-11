"""Vector index setup."""

from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from ..exceptions import QueryError
from ..logging import get_logger

logger = get_logger(__name__)


def initialize_vector_index(
    collection_name: str,
    chroma_client,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    embed_model,
) -> VectorStoreIndex:
    """
    Initialize vector index for semantic queries.

    Creates embeddings, stores in Chroma, and builds a VectorStoreIndex.

    Args:
        collection_name: Name of the Chroma collection
        chroma_client: Chroma client instance
        texts: List of text strings to index
        metadatas: Metadata for each text
        ids: IDs for each text
        embed_model: Embedding model for generating embeddings

    Returns:
        Configured VectorStoreIndex instance

    Raises:
        QueryError: If initialization fails
    """
    try:
        # Get or create the collection
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create documents
        documents = [
            Document(text=text, metadata=meta, id_=doc_id)
            for text, meta, doc_id in zip(texts, metadatas, ids)
        ]

        # Create index from documents using prescribed embeddings
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        logger.info(
            f"Vector index initialized for collection '{collection_name}' ({len(documents)} documents)"
        )
        return index

    except Exception as e:
        logger.error(f"Failed to initialize vector index: {e}", exc_info=True)
        raise QueryError(f"Vector index initialization failed: {e}") from e


def load_existing_vector_index(
    collection_name: str,
    chroma_client,
    embed_model,
) -> VectorStoreIndex:
    """
    Load an existing vector index from Chroma.

    This skips embedding generation and connects to existing data.
    Much faster than initialize_vector_index() when data already exists.

    Args:
        collection_name: Name of the existing Chroma collection
        chroma_client: Chroma client instance
        embed_model: Embedding model (for query encoding)

    Returns:
        VectorStoreIndex connected to existing data

    Raises:
        QueryError: If collection doesn't exist or loading fails
    """
    try:
        # Get existing collection (will raise if doesn't exist)
        chroma_collection = chroma_client.get_collection(name=collection_name)

        # Create vector store from existing collection
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )

        doc_count = chroma_collection.count()
        logger.info(
            f"Vector index loaded from existing collection '{collection_name}' ({doc_count} documents)"
        )
        return index

    except Exception as e:
        logger.error(f"Failed to load existing vector index: {e}", exc_info=True)
        raise QueryError(
            f"Vector index loading failed: {e}. "
            f"Collection '{collection_name}' may not exist. Use process() to create it first."
        ) from e
