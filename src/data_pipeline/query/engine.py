from typing import Optional, List, Dict, Any, Union
import asyncio
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from ..storage import StorageRepository
from ..exceptions import QueryError
from ..logging import get_logger
from .prompts import DEFAULT_SYSTEM_PROMPT
from .workflow_setup import create_query_workflow
from .sql_setup import initialize_sql_engine
from .vector_setup import initialize_vector_index, load_existing_vector_index
from .llm import ReasoningTokenHandler, LLMFactory, EngineType
from .agents.config import AgenticConfig
from ..config import Config

logger = get_logger(__name__)


class QueryEngine:
    def __init__(
        self,
        config: Config,
        system_prompt: Optional[str] = None,
        enable_chat_history: bool = True,
    ):
        """
        Initialize query engine.

        Args:
            config: Configuration instance (required)
            system_prompt: System prompt (uses default if None)
            enable_chat_history: Enable chat history
        """
        self.config = config
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.enable_chat_history = enable_chat_history

        # Set up reasoning handler if enabled
        self.reasoning_handler = (
            self._setup_reasoning_handler() if self.config.enable_reasoning_logs else None
        )

        # Create LLM using factory pattern
        self.llm = LLMFactory.create(
            model_name=self.config.llm_model,
            base_url=self.config.ollama_url,
            reasoning_handler=self.reasoning_handler,
            request_timeout=self.config.request_timeout,
            context_window=self.config.context_window,
            num_output=self.config.num_output,
            temperature=self.config.temperature,
            system_prompt=self.system_prompt,
        )

        self.embed_model = OllamaEmbedding(
            model_name=self.config.embed_model,
            base_url=self.config.ollama_url,
            embed_batch_size=self.config.embed_batch_size,
        )

        # Chat memory
        self.chat_memory = (
            ChatMemoryBuffer.from_defaults(token_limit=self.config.chat_history_token_limit)
            if enable_chat_history
            else None
        )

        # Query engines
        self.index: Optional[VectorStoreIndex] = None
        self.router_workflow = None  # Strategy instance (QueryWorkflow protocol)
        self.query_engine_tools = None
        self.summarizer = None

    def _setup_reasoning_handler(self) -> ReasoningTokenHandler:
        """Set up reasoning token handler and register with LlamaIndex callbacks."""
        handler = ReasoningTokenHandler(verbose=True)
        existing_handlers = [
            h
            for h in (Settings.callback_manager.handlers if Settings.callback_manager else [])
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

        # Initialize vector index
        self.index = initialize_vector_index(
            collection_name=collection_name,
            chroma_client=chroma_client,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embed_model=self.embed_model,
        )

        # SQL query engine (streaming enabled for workflow-based router)
        sql_llm = LLMFactory.create(
            model_name=self.config.llm_model,
            base_url=self.config.ollama_url,
            reasoning_handler=None,
            request_timeout=self.config.request_timeout,
            context_window=self.config.context_window,
            num_output=self.config.num_output,
            temperature=self.config.temperature,
            system_prompt=None,  # SQL engine has its own prompt
        )

        sql_query_engine = initialize_sql_engine(
            storage_repo=storage_repo,
            table_name=table_name,
            llm=sql_llm,  # Use non-reasoning LLM
            embed_model=self.embed_model,
            streaming=True,  # Workflows support streaming natively
        )

        # Initialize router workflow using config
        workflow_type = "react_agent" if self.config.use_react_agent else "custom_router"
        self.router_workflow, self.query_engine_tools, self.summarizer = create_query_workflow(
            workflow_type=workflow_type,
            index=self.index,
            llm=self.llm,
            similarity_top_k=self.config.similarity_top_k,
            sql_query_engine=sql_query_engine,
            reasoning_handler=self.reasoning_handler,
            request_timeout=self.config.request_timeout,
            context_window=self.config.context_window,
            num_output=self.config.num_output,
            config=self.config,
        )

    def initialize_from_existing(
        self,
        collection_name: str,
        chroma_client,
        table_name: str,
        storage_repo: StorageRepository,
    ) -> None:
        """
        Initialize query engines from existing persisted data.

        This skips data ingestion and embedding generation, connecting to
        existing DuckDB and Chroma data instead. Much faster than index_texts()
        when data already exists.

        Args:
            collection_name: Name of the existing Chroma collection
            chroma_client: Chroma client instance
            table_name: Name of the table for SQL queries
            storage_repo: Storage repository (required for SQL queries)

        Raises:
            QueryError: If initialization fails
        """
        # Load existing vector index (no embedding generation)
        self.index = load_existing_vector_index(
            collection_name=collection_name,
            chroma_client=chroma_client,
            embed_model=self.embed_model,
        )

        # SQL query engine (same as index_texts)
        sql_llm = LLMFactory.create(
            model_name=self.config.llm_model,
            base_url=self.config.ollama_url,
            reasoning_handler=None,
            request_timeout=self.config.request_timeout,
            context_window=self.config.context_window,
            num_output=self.config.num_output,
            temperature=self.config.temperature,
            system_prompt=None,
        )

        sql_query_engine = initialize_sql_engine(
            storage_repo=storage_repo,
            table_name=table_name,
            llm=sql_llm,
            embed_model=self.embed_model,
            streaming=True,
        )

        # Initialize router workflow
        workflow_type = "react_agent" if self.config.use_react_agent else "custom_router"
        self.router_workflow, self.query_engine_tools, self.summarizer = create_query_workflow(
            workflow_type=workflow_type,
            index=self.index,
            llm=self.llm,
            similarity_top_k=self.config.similarity_top_k,
            sql_query_engine=sql_query_engine,
            reasoning_handler=self.reasoning_handler,
            request_timeout=self.config.request_timeout,
            context_window=self.config.context_window,
            num_output=self.config.num_output,
            config=self.config,
        )

    def _build_query_with_context(self, question: str) -> str:
        """Build query with chat history context."""
        if not (self.chat_memory and self.enable_chat_history):
            return question

        messages = self.chat_memory.get_all()
        if not messages:
            return question

        # Include last N exchanges based on config
        agentic_config = AgenticConfig()
        context_parts = [
            f"{'User' if msg.role == MessageRole.USER else 'Assistant'}: {msg.content}"
            for msg in messages[-agentic_config.chat_history_context_size :]
        ]

        return (
            f"Previous conversation:\n{'\n'.join(context_parts)}\n\nCurrent question: {question}"
            if context_parts
            else question
        )

    def ask(self, question: str, return_metadata: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Ask a natural language question over the data.
        Uses configured workflow for query execution. Chat history is included as context to enhance responses.

        Args:
            question: Natural language question
            return_metadata: If True, returns dict with answer and execution metadata

        Returns:
            Answer string, or dict with 'answer' and 'metadata' if return_metadata=True
        """
        if not self.router_workflow:
            raise QueryError(
                "Query engine not initialized. Call index_texts() first.",
            )

        metadata = {"question": question, "sql_query": None, "error": None, "reasoning": None}

        # Reset reasoning handler for new query
        if self.reasoning_handler:
            self.reasoning_handler.reset()

        # Build query with chat history context
        enhanced_query = self._build_query_with_context(question)

        try:
            answer = None

            async def run_query():
                nonlocal answer
                result = await self.router_workflow.run(enhanced_query)
                answer = str(result.response)

                # Extract metadata - strategy-agnostic (works with any QueryWorkflow)
                selected_engine_index = self.router_workflow.selected_engine_index
                if selected_engine_index is not None:
                    try:
                        metadata["engine_type"] = EngineType(selected_engine_index).name_lower
                        metadata["selected_engine_index"] = selected_engine_index
                    except Exception:
                        metadata["engine_type"] = "unknown"
                else:
                    # Strategy doesn't track engine index
                    metadata["engine_type"] = "unknown"

            # nest_asyncio is applied in ui.py before any imports
            # This allows asyncio.run() to work even if an event loop exists
            asyncio.run(run_query())

            # Get captured reasoning tokens if enabled
            if self.config.enable_reasoning_logs and self.reasoning_handler:
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
            raise QueryError(f"Query execution failed: {e}", query=question) from e

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
