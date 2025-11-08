"""
Semantic text generation module.

Generates semantic text representations of rows for embedding.
"""

from typing import List
import pandas as pd
from ..schema import Schema


class SemanticTextGenerator:
    """Generates semantic text representations from data rows."""

    def __init__(self, schema: Schema):
        """Initialize generator with a schema."""
        self.schema = schema

    def generate_row_text(self, row: pd.Series) -> str:
        """
        Generate semantic text representation for a single row.

        Args:
            row: A pandas Series representing a single row

        Returns:
            A semantic text string
        """
        parts = []

        # Add table description if available
        if self.schema.description:
            parts.append(f"Table: {self.schema.table_name} - {self.schema.description}")

        # Add each field with its description
        for field in self.schema.fields:
            value = row[field.name]

            if pd.isna(value):
                continue

            # Build field description
            field_desc = f"{field.name}"
            if field.description:
                field_desc += f" ({field.description})"

            # Format value based on type
            if field.type.value == "boolean":
                value_str = "yes" if value else "no"
            else:
                value_str = str(value)

            parts.append(f"{field_desc}: {value_str}")

        return " | ".join(parts)

    def generate_all_texts(self, df: pd.DataFrame) -> List[str]:
        """
        Generate semantic text representations for all rows in a DataFrame.

        Optimized to use apply() instead of iterrows() for better performance.

        Args:
            df: DataFrame to process

        Returns:
            List of semantic text strings, one per row
        """
        return df.apply(self.generate_row_text, axis=1).tolist()
