from abc import ABC, abstractmethod
from typing import Optional, Dict
import pandas as pd
from sqlalchemy import create_engine, Engine, text
from ..logging import get_logger

logger = get_logger(__name__)


class StorageRepository(ABC):
    """Abstract repository interface for storage operations."""

    @abstractmethod
    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """Create a table with the given schema."""
        pass

    @abstractmethod
    def insert_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert a DataFrame into a table."""
        pass

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        pass


class DuckDBRepository(StorageRepository):
    """
    DuckDB storage repository using SQLAlchemy.

    All operations use SQLAlchemy's engine for consistency and compatibility
    with LlamaIndex's SQL query engines. This avoids connection conflicts
    and provides a clean, unified interface.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize repository with SQLAlchemy engine.

        Args:
            db_path: Path to database file. If None, uses in-memory database.
        """
        self.db_path = db_path
        connection_string = f"duckdb:///{db_path}" if db_path else "duckdb:///:memory:"
        self.engine = create_engine(connection_string)
        logger.debug(f"Storage repository initialized: {connection_string}")

    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """Create a table with the given schema."""
        columns = ", ".join([f"{name} {dtype}" for name, dtype in schema.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        with self.engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()

    def insert_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert a DataFrame into a table, replacing if exists."""
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        return pd.read_sql(sql, self.engine)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        sql = f"SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = '{table_name}'"
        with self.engine.connect() as conn:
            result = conn.execute(text(sql)).fetchone()
            return result[0] > 0

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """
        Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            DataFrame with column_name and column_type columns
        """
        return pd.read_sql(f"DESCRIBE {table_name}", self.engine)

    def get_sqlalchemy_engine(self) -> Engine:
        """
        Get SQLAlchemy engine for LlamaIndex integration.

        Returns:
            SQLAlchemy Engine instance
        """
        return self.engine

    def close(self):
        """Dispose of the SQLAlchemy engine and close connections."""
        if self.engine:
            self.engine.dispose()
            self.engine = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
