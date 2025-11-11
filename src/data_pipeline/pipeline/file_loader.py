from pathlib import Path
from typing import Optional
import pandas as pd
from ..exceptions import ConfigurationError
from ..schema import Schema


class FileLoader:
    """
    Utility for loading dataset files with format detection and date parsing.

    Supports CSV and Parquet formats with automatic date/datetime parsing
    based on schema field definitions.
    """

    @staticmethod
    def load(
        dataset_path: Path,
        schema: Optional[Schema] = None,
    ) -> pd.DataFrame:
        """
        Load a dataset file (CSV or Parquet).

        Args:
            dataset_path: Path to the dataset file
            schema: Optional schema for date/datetime parsing

        Returns:
            Loaded DataFrame

        Raises:
            ConfigurationError: If file format is unsupported
        """
        suffix = dataset_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(dataset_path)
        else:
            raise ConfigurationError(
                f"Unsupported file format: {suffix}",
                config_key="dataset_format",
            )

        # Parse date/datetime columns based on schema format specifications
        if schema:
            FileLoader._parse_date_columns(df, schema)

        return df

    @staticmethod
    def _parse_date_columns(df: pd.DataFrame, schema: Schema) -> None:
        """
        Parse date/datetime columns based on schema field definitions.

        Modifies the DataFrame in place.

        Args:
            df: DataFrame to modify
            schema: Schema with field definitions
        """
        for field in schema.fields:
            if field.type.value in ["date", "datetime"] and field.name in df.columns:
                if field.format:
                    # Parse with specified format (e.g., "%m/%d/%Y")
                    df[field.name] = pd.to_datetime(df[field.name], format=field.format)
                else:
                    # Let pandas infer format
                    df[field.name] = pd.to_datetime(df[field.name])

