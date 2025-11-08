from typing import Optional, List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from ..storage import StorageRepository
from ..exceptions import QueryError
from ..logging_config import get_logger
from .initialization import initialize_sql_engine, initialize_router

logger = get_logger(__name__)


class QueryEngine:
    """
    Hybrid query engine using LlamaIndex and Ollama.

    Uses LlamaIndex RouterQueryEngine for automatic routing between:
    - SQL engine for analytical queries (counts, aggregations, calculations)
    - Vector engine for semantic queries (similarity, descriptions, context)
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful data analysis assistant. Your role is to:
- Provide accurate, concise answers based on the available data
- Explain your reasoning when performing calculations
- Use specific numbers, names, and details from the data (not just IDs)
- Admit when you don't have enough information to answer
- Format numerical values clearly (e.g., currency, percentages)
- Be factual and avoid speculation

When answering questions about data, always include relevant context like product names, categories, or other identifying information, not just numeric IDs."""

    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
        embed_model_name: Optional[str] = None,
        context_window: int = 32768,  # Mistral default
        num_output: int = 1024,
        temperature: float = 0.2,  # Lower for more deterministic answers
        similarity_top_k: int = 5,  # Number of chunks to retrieve for context
        system_prompt: Optional[str] = None,
        enable_chat_history: bool = True,
        chat_history_token_limit: int = 3000,
        embed_batch_size: int = 32,  # Batch size for embedding generation
    ):
        """
        Initialize query engine.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            embed_model_name: Name of the embedding model (defaults to model_name)
            context_window: Maximum context window size for the LLM
            num_output: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            similarity_top_k: Number of similar chunks to retrieve for vector search
            system_prompt: System prompt to guide LLM behavior (uses default if None)
            enable_chat_history: Whether to maintain conversation history
            chat_history_token_limit: Max tokens to keep in chat history
            embed_batch_size: Number of texts to batch per embedding API call (default: 32)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embed_model_name = embed_model_name or model_name
        self.similarity_top_k = similarity_top_k
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.enable_chat_history = enable_chat_history

        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            request_timeout=60.0,
            context_window=context_window,
            num_output=num_output,
            temperature=temperature,
            additional_kwargs={
                "num_ctx": context_window,  # Ollama-specific parameter
            },
            system_prompt=self.system_prompt,
        )

        self.embed_model = OllamaEmbedding(
            model_name=self.embed_model_name,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
        )

        # Chat memory
        self.chat_memory = (
            ChatMemoryBuffer.from_defaults(token_limit=chat_history_token_limit)
            if enable_chat_history
            else None
        )

        # Query engines
        self.index: Optional[VectorStoreIndex] = None
        self.router_query_engine = None

    def index_texts(
        self,
        collection_name: str,
        chroma_client,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        table_name: str,
        storage_repo: StorageRepository,
    ) -> None:
        """
        Index texts into Chroma and initialize hybrid query engines.

        Creates embeddings, stores in Chroma, and initializes both SQL and vector
        query engines with intelligent routing between them.

        Args:
            collection_name: Name of the Chroma collection
            chroma_client: Chroma client instance
            texts: List of text strings to index
            metadatas: Metadata for each text
            ids: IDs for each text
            table_name: Name of the table for SQL queries
            storage_repo: Storage repository (required for SQL queries)

        Raises:
            QueryError: If initialization fails
        """

        # Get the collection
        chroma_collection = chroma_client.get_collection(name=collection_name)

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
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        # SQL query engine
        sql_query_engine = initialize_sql_engine(
            storage_repo=storage_repo,
            table_name=table_name,
            llm=self.llm,
            embed_model=self.embed_model,
        )

        # Initialize router with both SQL and vector engines
        self.router_query_engine = initialize_router(
            index=self.index,
            llm=self.llm,
            similarity_top_k=self.similarity_top_k,
            sql_query_engine=sql_query_engine,
        )

    def _build_query_with_context(self, question: str) -> str:
        """Build query with chat history context to avoid recursive buildup."""
        if not self.chat_memory or not self.enable_chat_history:
            return question

        messages = self.chat_memory.get_all()
        if not messages:
            return question

        # Include last 3 exchanges (6 messages) - arbitrary but keeps context manageable
        context_parts = []
        for msg in messages[-6:]:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            context_parts.append(f"{role}: {msg.content}")

        if context_parts:
            context = "\n".join(context_parts)
            return f"""Previous conversation:
{context}

Current question: {question}"""

        return question

    def ask(self, question: str, return_metadata: bool = False) -> str | Dict[str, Any]:
        """
        Ask a natural language question over the data.

        Uses LlamaIndex RouterQueryEngine for automatic routing between SQL and vector engines.
        Routing decisions are logged to console when verbose mode is enabled.

        Chat history is included as context to enhance responses.

        Args:
            question: Natural language question
            return_metadata: If True, returns dict with answer and execution metadata

        Returns:
            Answer string, or dict with 'answer' and 'metadata' if return_metadata=True
        """
        if not self.router_query_engine:
            raise QueryError("Query engine not initialized. Call index_texts() first.")

        metadata = {"question": question, "sql_query": None, "error": None}

        # Build query with chat history context
        enhanced_query = self._build_query_with_context(question)

        # Route
        try:
            response = self.router_query_engine.query(enhanced_query)
            answer = str(response)

            # Hacky but... capture SQL query if available
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                metadata["sql_query"] = response.metadata.get("sql_query")

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            metadata["error"] = str(e)
            raise QueryError(f"Query execution failed: {e}") from e

        # Add to chat history
        if self.chat_memory:
            self.chat_memory.put(ChatMessage(role=MessageRole.USER, content=question))
            self.chat_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=answer))

        return {"answer": answer, "metadata": metadata} if return_metadata else answer

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.

        Returns:
            List of messages with 'role' and 'content' keys
        """
        if not self.chat_memory:
            return []

        messages = self.chat_memory.get_all()
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def clear_chat_history(self) -> None:
        """Clear the conversation history."""
        if self.chat_memory:
            self.chat_memory.reset()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the system prompt.

        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt
        # Note: Updating system_prompt on existing Ollama instance may not work
        if hasattr(self.llm, "system_prompt"):
            self.llm.system_prompt = prompt
