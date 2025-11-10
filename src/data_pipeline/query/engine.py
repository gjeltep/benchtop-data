from typing import Optional, List, Dict, Any, Union
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import Document
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from ..storage import StorageRepository
from ..exceptions import QueryError
from ..logging_config import get_logger
from .setup import initialize_sql_engine, initialize_router_workflow
from .llm import ReasoningTokenHandler, LLMFactory, EngineType

logger = get_logger(__name__)


class QueryEngine:
    """
    Hybrid query engine using LlamaIndex Workflows and Ollama.

    Architecture:
    - Uses LlamaIndex RouterQueryEngineWorkflow (native Workflow pattern) for routing:
      * SQL engine: Analytical queries (counts, aggregations, calculations)
      * Vector engine: Semantic queries (similarity, descriptions, context)

    - When enable_reasoning_logs=True:
      * Uses ReasoningTokenHandler to capture thinking_delta tokens
      * Tokens are captured directly from streaming responses in workflow steps
      * No wrapper needed - native LlamaIndex workflow pattern

    This design uses native LlamaIndex Workflows for better observability
    and reasoning token capture without custom abstractions.
    """

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
        request_timeout: float = 180.0,  # Request timeout in seconds
        enable_reasoning_logs: bool = True,  # Enable logging of reasoning tokens
    ):
        """
        Initialize query engine.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            embed_model_name: Name of the embedding model (defaults to model_name if not provided)
            context_window: Maximum context window size for the LLM
            num_output: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            similarity_top_k: Number of similar chunks to retrieve for vector search
            system_prompt: System prompt to guide LLM behavior (uses default if None)
            enable_chat_history: Whether to maintain conversation history
            chat_history_token_limit: Max tokens to keep in chat history
            embed_batch_size: Number of texts to batch per embedding API call (default: 32)
            request_timeout: Request timeout in seconds for Ollama API calls
            enable_reasoning_logs: If True, logs reasoning tokens (thinking_delta) to console.
                Works with any reasoning-based LLM that outputs thinking tokens via Ollama.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embed_model_name = embed_model_name or model_name
        self.similarity_top_k = similarity_top_k
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.enable_chat_history = enable_chat_history
        self.enable_reasoning_logs = enable_reasoning_logs
        self.request_timeout = request_timeout
        self.context_window = context_window
        self.num_output = num_output
        self.temperature = temperature

        # Set up reasoning handler if enabled
        self.reasoning_handler = self._setup_reasoning_handler() if enable_reasoning_logs else None

        # Create LLM using factory pattern
        self.llm = LLMFactory.create(
            model_name=model_name,
            base_url=base_url,
            reasoning_handler=self.reasoning_handler,
            request_timeout=request_timeout,
            context_window=context_window,
            num_output=num_output,
            temperature=temperature,
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
        self.router_workflow = None
        self.query_engine_tools = None
        self.summarizer = None

    def _setup_reasoning_handler(self) -> ReasoningTokenHandler:
        """Set up reasoning token handler and register with LlamaIndex callbacks."""
        handler = ReasoningTokenHandler(verbose=True)
        existing_handlers = [
            h for h in (Settings.callback_manager.handlers if Settings.callback_manager else [])
            if not isinstance(h, ReasoningTokenHandler)
        ]
        Settings.callback_manager = CallbackManager(existing_handlers + [handler])
        return handler

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

        # SQL query engine (streaming enabled for workflow-based router)
        # Use non-reasoning LLM with temperature=0.0 for deterministic SQL generation
        sql_llm = LLMFactory.create(
            model_name=self.model_name,
            base_url=self.base_url,
            reasoning_handler=None,
            request_timeout=self.request_timeout,
            context_window=self.context_window,
            num_output=self.num_output,
            temperature=self.temperature,
            system_prompt=None,  # SQL engine has its own prompt
        )

        sql_query_engine = initialize_sql_engine(
            storage_repo=storage_repo,
            table_name=table_name,
            llm=sql_llm,  # Use non-reasoning LLM
            embed_model=self.embed_model,
            streaming=True,  # Workflows support streaming natively
        )

        # Initialize workflow-based router (native LlamaIndex pattern)
        self.router_workflow, self.query_engine_tools, self.summarizer = initialize_router_workflow(
            index=self.index,
            llm=self.llm,
            similarity_top_k=self.similarity_top_k,
            sql_query_engine=sql_query_engine,
            reasoning_handler=self.reasoning_handler,
        )

    def _build_query_with_context(self, question: str) -> str:
        """Build query with chat history context."""
        if not (self.chat_memory and self.enable_chat_history):
            return question

        messages = self.chat_memory.get_all()
        if not messages:
            return question

        # Include last 3 exchanges (6 messages)
        context_parts = [
            f"{'User' if msg.role == MessageRole.USER else 'Assistant'}: {msg.content}"
            for msg in messages[-6:]
        ]

        return f"Previous conversation:\n{chr(10).join(context_parts)}\n\nCurrent question: {question}" if context_parts else question

    def ask(self, question: str, return_metadata: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Ask a natural language question over the data.

        Uses LlamaIndex RouterQueryEngineWorkflow for automatic routing between SQL and vector engines.
        Routing decisions are logged to console when verbose mode is enabled.

        Chat history is included as context to enhance responses.

        If enable_reasoning_logs is True, captures and logs reasoning tokens (thinking_delta)
        from reasoning-based LLMs via workflow streaming.

        Args:
            question: Natural language question
            return_metadata: If True, returns dict with answer and execution metadata

        Returns:
            Answer string, or dict with 'answer' and 'metadata' if return_metadata=True
        """
        if not self.router_workflow:
            raise QueryError("Query engine not initialized. Call index_texts() first.")

        metadata = {"question": question, "sql_query": None, "error": None, "reasoning": None}

        # Reset reasoning handler for new query
        if self.reasoning_handler:
            self.reasoning_handler.reset()

        # Build query with chat history context
        enhanced_query = self._build_query_with_context(question)

        # Use workflow-based router (native LlamaIndex pattern)
        try:
            # Run workflow asynchronously
            # Streamlit uses Tornado (not asyncio), so we can use asyncio.run() directly
            # If an event loop exists, nest_asyncio will handle it (applied at import if needed)
            async def run_workflow():
                return await self.router_workflow.run(
                    query=enhanced_query,
                    # Don't pass LLM, summarizer, or tools - they're stored as instance vars to avoid deepcopy issues
                    select_multi=False,  # Select single engine
                )

            # nest_asyncio is applied in ui.py before any imports
            # This allows asyncio.run() to work even if an event loop exists
            result = asyncio.run(run_workflow())

            # Extract answer from StopEvent result (which is a Response object)
            # Following LlamaIndex best practices: StopEvent.result is the Response
            if hasattr(result, 'result'):
                answer = str(result.result) if result.result else str(result)
            else:
                answer = str(result)

            # Extract metadata from workflow
            selected_engine_index = self.router_workflow.selected_engine_index
            if selected_engine_index is not None:
                try:
                    metadata["engine_type"] = EngineType(selected_engine_index).name_lower
                except Exception:
                    metadata["engine_type"] = "unknown"
                metadata["selected_engine_index"] = selected_engine_index

            # Get captured reasoning tokens if enabled
            if self.enable_reasoning_logs and self.reasoning_handler:
                reasoning = self.reasoning_handler.get_reasoning()
                if reasoning:
                    metadata["reasoning"] = reasoning
                    if self.reasoning_handler.verbose:
                        print("\n" + "=" * 60)
                        print("ANSWER:")
                        print("=" * 60)

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
