"""
Storage module for DuckDB persistence.

Uses repository pattern for clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import duckdb
import pandas as pd
from sqlalchemy import create_engine, Engine
import tempfile
import os


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
    """DuckDB implementation of storage repository."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DuckDB repository.

        Args:
            db_path: Path to DuckDB file. If None, uses in-memory database.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path or ":memory:")
        self._temp_db_file: Optional[str] = None  # For SQLAlchemy engine in in-memory mode

    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """Create a table with the given schema."""
        columns = ", ".join([f"{name} {dtype}" for name, dtype in schema.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.conn.execute(sql)

    def insert_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert a DataFrame into a table."""
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        # Use Arrow format for better type preservation (especially for timestamps)
        try:
            arrow_table = self.conn.execute(sql).arrow()
            return arrow_table.to_pandas()
        except (ImportError, AttributeError):
            # Fallback to standard df() if Arrow is unavailable
            return self.conn.execute(sql).df()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        result = self.conn.execute(
            f"SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = '{table_name}'"
        ).fetchone()
        return result[0] > 0

    def get_connection(self):
        """
        Get the DuckDB connection.

        Returns:
            DuckDB connection object
        """
        return self.conn

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """
        Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            DataFrame with column_name and column_type columns
        """
        return self.conn.execute(f"DESCRIBE {table_name}").df()

    def get_sqlalchemy_engine(self) -> Engine:
        """
        Get SQLAlchemy engine for DuckDB.

        This enables integration with LlamaIndex's SQLTableQueryEngine.
        For in-memory databases, uses a temporary file to avoid duckdb-engine compatibility issues.

        Returns:
            SQLAlchemy Engine instance
        """
        # For file-based databases, use the file path
        if self.db_path:
            connection_string = f"duckdb:///{self.db_path}"
            return create_engine(connection_string)

        # For in-memory databases, use a temporary file
        # This avoids duckdb-engine compatibility issues with :memory: connections
        if self._temp_db_file is None:
            # Create temporary file path (don't create the file yet - let DuckDB create it)
            fd, self._temp_db_file = tempfile.mkstemp(suffix=".duckdb")
            os.close(fd)  # Close file descriptor
            os.unlink(self._temp_db_file)  # Delete empty file - DuckDB will create it

            # Initialize the DuckDB file by creating a connection first
            # DuckDB will create the file if it doesn't exist
            temp_conn = duckdb.connect(self._temp_db_file)

            # Copy data from in-memory connection to temp file
            try:
                tables = self.conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()

                if tables:
                    for (table_name,) in tables:
                        # Copy table structure and data
                        # DuckDB can reference Python variables directly in SQL
                        df = self.conn.execute(f"SELECT * FROM {table_name}").df()  # noqa: F841
                        temp_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                        temp_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

                # Commit and close to ensure data is written
                temp_conn.close()
            except Exception:
                # Clean up temp file on error
                if temp_conn:
                    temp_conn.close()
                if os.path.exists(self._temp_db_file):
                    os.unlink(self._temp_db_file)
                    self._temp_db_file = None
                raise

        # Use the temporary file for SQLAlchemy engine
        connection_string = f"duckdb:///{self._temp_db_file}"
        return create_engine(connection_string)

    def close(self):
        """Close the database connection."""
        self.conn.close()
        # Clean up temporary file if used
        if self._temp_db_file and os.path.exists(self._temp_db_file):
            os.unlink(self._temp_db_file)
            self._temp_db_file = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
