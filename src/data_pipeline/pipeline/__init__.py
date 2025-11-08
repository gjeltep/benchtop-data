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
from ..exceptions import ValidationError, ConfigurationError
from ..logging_config import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """Main pipeline orchestrator."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        chroma_path: Optional[str] = None,
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434",
        context_window: int = 32768,
        temperature: float = 0.1,
        similarity_top_k: int = 5,
        embed_batch_size: int = 32,
        storage_repo: Optional[StorageRepository] = None,
    ):
        """
        Initialize the data pipeline.

        Args:
            db_path: Path to DuckDB file (None = in-memory)
            chroma_path: Path to Chroma persistence directory (None = in-memory)
            ollama_model: Ollama model name to use
            ollama_base_url: Ollama API base URL
            context_window: Maximum context window size for the LLM
            temperature: Sampling temperature
            similarity_top_k: Number of similar chunks to retrieve for vector search
            embed_batch_size: Number of texts to batch per embedding API call (default: 32)
            storage_repo: Optional storage repository (constructor DI)
        """
        self.storage_repo = storage_repo or DuckDBRepository(db_path)
        self.embeddings_repo = ChromaRepository(chroma_path)

        # Query engine with config
        self.query_engine = QueryEngine(
            model_name=ollama_model,
            base_url=ollama_base_url,
            context_window=context_window,
            temperature=temperature,
            similarity_top_k=similarity_top_k,
            embed_batch_size=embed_batch_size,
        )

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

        if path.suffix == ".csv":
            df = pd.read_csv(path)

            # Parse date/datetime columns based on schema format specifications
            if self.schema:
                for field in self.schema.fields:
                    if field.type.value in ["date", "datetime"] and field.name in df.columns:
                        if field.format:
                            # Parse with specified format (e.g., "%m/%d/%Y")
                            df[field.name] = pd.to_datetime(df[field.name], format=field.format)
                        else:
                            # Let pandas infer format
                            df[field.name] = pd.to_datetime(df[field.name])
        elif path.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        else:
            raise ConfigurationError(f"Unsupported file format: {path.suffix}")

        return df

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
            raise ValidationError("Dataset validation failed:\n" + "\n".join(errors))

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

        # Create or get collection (without adding data yet)
        self.embeddings_repo.create_collection(collection_name)

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
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434",
    context_window: int = 32768,
    temperature: float = 0.2,
    similarity_top_k: int = 5,
    embed_batch_size: int = 32,
) -> DataPipeline:
    """
    Factory function to create a pipeline instance.

    Args:
        db_path: Path to DuckDB file (None = in-memory)
        chroma_path: Path to Chroma persistence directory (None = in-memory)
        ollama_model: Ollama model name to use
        ollama_base_url: Ollama API base URL
        context_window: Maximum context window size for the LLM
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        similarity_top_k: Number of similar chunks to retrieve for vector search
        embed_batch_size: Number of texts to batch per embedding API call (default: 32)

    Returns:
        Configured DataPipeline instance
    """
    return DataPipeline(
        db_path=db_path,
        chroma_path=chroma_path,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        context_window=context_window,
        temperature=temperature,
        similarity_top_k=similarity_top_k,
        embed_batch_size=embed_batch_size,
    )
