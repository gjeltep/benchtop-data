"""Shared pytest fixtures for data pipeline tests."""

import pandas as pd
import pytest

from data_pipeline.schema import Field, FieldType, Schema


@pytest.fixture
def sample_string_field():
    """A simple string field."""
    return Field(name="product_name", type=FieldType.STRING, description="Name of the product")


@pytest.fixture
def sample_integer_field():
    """A simple integer field."""
    return Field(name="quantity", type=FieldType.INTEGER, description="Quantity in stock")


@pytest.fixture
def sample_float_field():
    """A simple float field."""
    return Field(name="price", type=FieldType.FLOAT, description="Product price")


@pytest.fixture
def sample_boolean_field():
    """A simple boolean field."""
    return Field(name="in_stock", type=FieldType.BOOLEAN, description="Is the product in stock")


@pytest.fixture
def sample_date_field():
    """A simple date field."""
    return Field(name="release_date", type=FieldType.DATE, description="Product release date")


@pytest.fixture
def sample_datetime_field():
    """A simple datetime field."""
    return Field(name="last_updated", type=FieldType.DATETIME, description="Last update timestamp")


@pytest.fixture
def sample_primary_key_field():
    """A primary key field."""
    return Field(name="id", type=FieldType.INTEGER, description="Product ID", primary_key=True)


@pytest.fixture
def sample_schema(sample_primary_key_field, sample_string_field, sample_float_field):
    """A complete valid schema with multiple fields."""
    return Schema(
        table_name="products",
        fields=[
            sample_primary_key_field,
            sample_string_field,
            sample_float_field,
        ],
    )


@pytest.fixture
def comprehensive_schema():
    """A schema with all field types for comprehensive testing."""
    return Schema(
        table_name="test_table",
        fields=[
            Field(name="id", type=FieldType.INTEGER, description="ID", primary_key=True),
            Field(name="name", type=FieldType.STRING, description="Name"),
            Field(name="price", type=FieldType.FLOAT, description="Price"),
            Field(name="quantity", type=FieldType.INTEGER, description="Quantity"),
            Field(name="active", type=FieldType.BOOLEAN, description="Is active"),
            Field(name="created_date", type=FieldType.DATE, description="Creation date"),
            Field(name="updated_at", type=FieldType.DATETIME, description="Last update"),
        ],
    )


@pytest.fixture
def sample_dataframe():
    """A simple pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "product_name": ["Widget", "Gadget", "Doohickey"],
            "price": [10.99, 25.50, 5.00],
        }
    )


@pytest.fixture
def comprehensive_dataframe():
    """A DataFrame with all field types for comprehensive testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Product A", "Product B", "Product C"],
            "price": [10.99, 25.50, 5.00],
            "quantity": [100, 50, 200],
            "active": [True, False, True],
            "created_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "updated_at": pd.to_datetime(["2024-01-15 10:30:00", "2024-01-16 14:45:00", "2024-01-17 09:15:00"]),
        }
    )


@pytest.fixture
def dataframe_with_nulls():
    """A DataFrame with null values for testing null handling."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Product A", None, "Product C"],
            "price": [10.99, 25.50, None],
            "active": [True, None, False],
        }
    )


@pytest.fixture
def schema_dict_valid():
    """A valid schema dictionary for SchemaParser testing."""
    return {
        "table_name": "products",
        "fields": [
            {"name": "id", "type": "integer", "description": "Product ID", "primary_key": True},
            {"name": "name", "type": "string", "description": "Product name"},
            {"name": "price", "type": "float", "description": "Product price"},
        ],
    }


@pytest.fixture
def schema_dict_with_string_types():
    """A schema dictionary with string type representations."""
    return {
        "table_name": "test_table",
        "fields": [
            {"name": "id", "type": "integer", "primary_key": True},
            {"name": "count", "type": "integer"},
            {"name": "value", "type": "float"},
            {"name": "flag", "type": "boolean"},
            {"name": "label", "type": "string"},
            {"name": "created", "type": "date"},
            {"name": "timestamp", "type": "datetime"},
        ],
    }
