"""Unit tests for semantic text generation module."""

import pandas as pd
import pytest

from data_pipeline.schema import Field, FieldType, Schema
from data_pipeline.semantic import SemanticTextGenerator


class TestSemanticTextGenerator:
    """Tests for SemanticTextGenerator class."""

    def test_generate_row_text_table_description(self):
        """Test that table description is included when available and skipped when not."""
        # With table description
        schema_with_desc = Schema(
            table_name="products",
            description="Product catalog",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True),
                Field(name="name", type=FieldType.STRING),
            ],
        )
        generator = SemanticTextGenerator(schema_with_desc)
        row = pd.Series({"id": 1, "name": "Widget"})
        text = generator.generate_row_text(row)
        assert text.startswith("Table: products - Product catalog")

        # Without table description
        schema_without_desc = Schema(
            table_name="products",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True),
                Field(name="name", type=FieldType.STRING),
            ],
        )
        generator = SemanticTextGenerator(schema_without_desc)
        text = generator.generate_row_text(row)
        assert not text.startswith("Table:")

    def test_generate_row_text_boolean_conversion(self):
        """Test that booleans are converted to yes/no."""
        schema = Schema(
            table_name="products",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True),
                Field(name="active", type=FieldType.BOOLEAN, description="Is active"),
            ],
        )
        generator = SemanticTextGenerator(schema)

        # Test True -> yes
        row_true = pd.Series({"id": 1, "active": True})
        text = generator.generate_row_text(row_true)
        assert "active (Is active): yes" in text

        # Test False -> no
        row_false = pd.Series({"id": 1, "active": False})
        text = generator.generate_row_text(row_false)
        assert "active (Is active): no" in text

    def test_generate_row_text_skips_null_values(self):
        """Test that null values are skipped."""
        schema = Schema(
            table_name="products",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True),
                Field(name="name", type=FieldType.STRING, description="Product name"),
                Field(name="price", type=FieldType.FLOAT, description="Product price"),
            ],
        )
        generator = SemanticTextGenerator(schema)
        row = pd.Series({"id": 1, "name": "Widget", "price": None})

        text = generator.generate_row_text(row)

        assert "id: 1" in text
        assert "name (Product name): Widget" in text
        assert "price" not in text  # Should be skipped

    def test_generate_row_text_field_without_description(self):
        """Test field formatting when description is not provided."""
        schema = Schema(
            table_name="products",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True),
                Field(name="code", type=FieldType.STRING),  # No description
            ],
        )
        generator = SemanticTextGenerator(schema)
        row = pd.Series({"id": 1, "code": "ABC123"})

        text = generator.generate_row_text(row)

        assert "code: ABC123" in text
        assert "code (" not in text  # No parentheses when no description

    def test_generate_row_text_all_field_types(self):
        """Test text generation with all field types."""
        schema = Schema(
            table_name="test",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True),
                Field(name="name", type=FieldType.STRING, description="Name"),
                Field(name="price", type=FieldType.FLOAT, description="Price"),
                Field(name="quantity", type=FieldType.INTEGER, description="Quantity"),
                Field(name="active", type=FieldType.BOOLEAN, description="Active"),
                Field(name="created", type=FieldType.DATE, description="Created date"),
            ],
        )
        generator = SemanticTextGenerator(schema)
        row = pd.Series(
            {
                "id": 1,
                "name": "Product A",
                "price": 29.99,
                "quantity": 100,
                "active": True,
                "created": pd.Timestamp("2024-01-15"),
            }
        )

        text = generator.generate_row_text(row)

        assert "id: 1" in text
        assert "name (Name): Product A" in text
        assert "price (Price): 29.99" in text
        assert "quantity (Quantity): 100" in text
        assert "active (Active): yes" in text
        assert "created (Created date):" in text
        assert "2024-01-15" in text

    def test_generate_all_texts_multiple_rows(self):
        """Test generating text for multiple rows and return type."""
        schema = Schema(
            table_name="products",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True),
                Field(name="name", type=FieldType.STRING, description="Product name"),
            ],
        )
        generator = SemanticTextGenerator(schema)
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Widget", "Gadget", "Doohickey"]})

        texts = generator.generate_all_texts(df)

        assert isinstance(texts, list)
        assert len(texts) == 3
        assert "id: 1" in texts[0]
        assert "name (Product name): Widget" in texts[0]
        assert "id: 2" in texts[1]
        assert "name (Product name): Gadget" in texts[1]
        assert "id: 3" in texts[2]
        assert "name (Product name): Doohickey" in texts[2]
