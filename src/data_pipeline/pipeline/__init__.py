"""
Main pipeline orchestration module.
"""

from typing import Optional, Union
from pathlib import Path
import pandas as pd
from ..schema import Schema, SchemaParser, SchemaValidator
from ..storage import StorageRepository, DuckDBRepository
from ..semantic import SemanticTextGenerator
from ..embeddings import ChromaRepository
from ..query import QueryEngine
from ..exceptions import ValidationError
from ..logging import get_logger
from ..config import Config, load_config
from .file_loader import FileLoader

logger = get_logger(__name__)


class DataPipeline:
    """Main pipeline orchestrator."""

    def __init__(
        self,
        config: Optional[Config] = None,
        storage_repo: Optional[StorageRepository] = None,
    ):
        """
        Initialize the data pipeline.

        Args:
            config: Pipeline configuration (uses defaults from load_config() if None)
            storage_repo: Optional storage repository (constructor DI, overrides config.db_path)
        """
        # Use provided config or load defaults
        self.config = config or load_config()

        # Use provided storage_repo or create from config
        self.storage_repo = storage_repo or DuckDBRepository(self.config.db_path)
        self.embeddings_repo = ChromaRepository(self.config.chroma_path)

        # Query engine with config values
        self.query_engine = QueryEngine(config=self.config)

        # State
        self.schema: Optional[Schema] = None
        self.table_name: Optional[str] = None

    def load_schema(self, schema_path: Union[str, Path]) -> None:
        """
        Load schema definition from file.

        Args:
            schema_path: Path to schema YAML/JSON file
        """
        self.schema = SchemaParser.parse_file(str(schema_path))
        self.table_name = self.schema.table_name

    def load_dataset(self, dataset_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load dataset from file (CSV, Parquet, etc.).

        Args:
            dataset_path: Path to dataset file

        Returns:
            Loaded DataFrame
        """
        path = Path(dataset_path)
        return FileLoader.load(path, schema=self.schema)

    def process(self, dataset_path: Union[str, Path], schema_path: Union[str, Path]) -> None:
        """
        Process a dataset with its schema.

        This method:
        1. Loads the schema
        2. Loads the dataset
        3. Validates the dataset
        4. Stores in DuckDB
        5. Generates semantic texts
        6. Creates embeddings in Chroma
        7. Initializes query engine

        Args:
            dataset_path: Path to dataset file
            schema_path: Path to schema definition file
        """
        # Load schema
        self.load_schema(schema_path)

        # Load dataset
        df = self.load_dataset(dataset_path)

        # Validate dataset
        validator = SchemaValidator(self.schema)
        is_valid, errors = validator.validate(df)
        if not is_valid:
            raise ValidationError(
                "Dataset validation failed:\n" + "\n".join(errors),
                table_name=self.table_name,
            )

        # Store in DuckDB
        self.storage_repo.insert_dataframe(self.table_name, df)

        # Generate semantic texts
        generator = SemanticTextGenerator(self.schema)
        texts = generator.generate_all_texts(df)

        # Generate IDs (using primary key if available)
        pk_fields = self.schema.get_primary_key_fields()
        if pk_fields:
            # Use vectorized operations when possible
            pk_cols = [f.name for f in pk_fields]
            ids = df[pk_cols].astype(str).apply("_".join, axis=1).tolist()
        else:
            ids = [str(i) for i in range(len(df))]

        # Create metadata for each row
        metadatas = [
            {col: str(val) for col, val in row.items()} for row in df.to_dict(orient="records")
        ]

        # Create Chroma collection and initialize with LlamaIndex
        collection_name = f"{self.table_name}_embeddings"

        # Index texts and initialize query engines for hybrid SQL+vector queries
        self.query_engine.index_texts(
            collection_name=collection_name,
            chroma_client=self.embeddings_repo.get_client(),
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            table_name=self.table_name,
            storage_repo=self.storage_repo,
        )

    def load_existing(self, schema_path: Union[str, Path]) -> None:
        """
        Load from existing persisted database and embeddings.

        This method skips data ingestion and embedding generation, instead
        connecting to existing DuckDB and Chroma data. Much faster than
        process() when data already exists.

        Requirements:
        - DuckDB database must already exist at config.db_path
        - Chroma collection must already exist at config.chroma_path
        - Schema file must match the existing data

        Args:
            schema_path: Path to schema definition file

        Raises:
            FileNotFoundError: If database or embeddings don't exist
            QueryError: If query engine initialization fails
        """
        # Load schema to get table name
        self.load_schema(schema_path)

        # Verify table exists in database
        if not self.storage_repo.table_exists(self.table_name):
            raise FileNotFoundError(
                f"Table '{self.table_name}' not found in database. "
                f"Use process() to create it first."
            )

        # Initialize query engines from existing data
        collection_name = f"{self.table_name}_embeddings"

        self.query_engine.initialize_from_existing(
            collection_name=collection_name,
            chroma_client=self.embeddings_repo.get_client(),
            table_name=self.table_name,
            storage_repo=self.storage_repo,
        )

    def ask(self, question: str, return_metadata: bool = False):
        """
        Ask a natural language question over the dataset.

        Args:
            question: Natural language question
            return_metadata: If True, returns dict with answer and execution metadata

        Returns:
            Answer string, or dict with 'answer' and 'metadata' if return_metadata=True
        """
        return self.query_engine.ask(question, return_metadata=return_metadata)

    def get_chat_history(self):
        """Get the conversation history."""
        return self.query_engine.get_chat_history()

    def clear_chat_history(self):
        """Clear the conversation history."""
        self.query_engine.clear_chat_history()

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.query_engine.set_system_prompt(prompt)

    def close(self):
        """Close all connections."""
        self.storage_repo.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_pipeline(
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    context_window: Optional[int] = None,
    temperature: Optional[float] = None,
    similarity_top_k: Optional[int] = None,
    embed_batch_size: Optional[int] = None,
    enable_reasoning_logs: Optional[bool] = None,
    use_react_agent: Optional[bool] = None,
) -> DataPipeline:
    """
    Factory function to create a pipeline instance.

    All parameters default to None, which means they will use values from load_config().
    This allows config to be the single source of truth for defaults.

    Args:
        db_path: Path to DuckDB file (None = uses config default, typically in-memory)
        chroma_path: Path to Chroma persistence directory (None = uses config default, typically in-memory)
        ollama_model: Ollama model name to use (None = uses config default)
        ollama_base_url: Ollama API base URL (None = uses config default)
        context_window: Maximum context window size for the LLM (None = uses config default)
        temperature: Sampling temperature (None = uses config default, 0.0 = deterministic, 1.0 = creative)
        similarity_top_k: Number of similar chunks to retrieve for vector search (None = uses config default)
        embed_batch_size: Number of texts to batch per embedding API call (None = uses config default)
        enable_reasoning_logs: Enable logging of reasoning tokens from reasoning-based LLMs (None = uses config default)
        use_react_agent: Use ReActAgent as entire workflow instead of custom workflow (None = uses config default)

    Returns:
        Configured DataPipeline instance
    """
    base_config = load_config()
    config = base_config.with_overrides(
        db_path=db_path,
        chroma_path=chroma_path,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        context_window=context_window,
        temperature=temperature,
        similarity_top_k=similarity_top_k,
        embed_batch_size=embed_batch_size,
        enable_reasoning_logs=enable_reasoning_logs,
        use_react_agent=use_react_agent,
    )
    return DataPipeline(config=config)
