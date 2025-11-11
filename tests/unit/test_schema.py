"""Unit tests for schema module."""

import pandas as pd
import pytest

from data_pipeline.exceptions import SchemaError
from data_pipeline.schema import Field, FieldType, Schema, SchemaParser, SchemaValidator


class TestField:
    """Tests for Field dataclass."""

    def test_field_creation_with_valid_data(self):
        """Test creating a field with valid data and defaults."""
        # Test basic field with defaults
        field = Field(name="test_field", type=FieldType.STRING, description="Test field")
        assert field.name == "test_field"
        assert field.type == FieldType.STRING
        assert field.description == "Test field"
        assert field.nullable is True
        assert field.primary_key is False

        # Test primary key field
        pk_field = Field(name="id", type=FieldType.INTEGER, primary_key=True)
        assert pk_field.primary_key is True

        # Test nullable=False
        non_null_field = Field(name="required", type=FieldType.STRING, nullable=False)
        assert non_null_field.nullable is False

    def test_field_with_string_type_conversion(self):
        """Test that string types are converted to FieldType enum."""
        field = Field(name="test", type="string")
        assert field.type == FieldType.STRING
        assert isinstance(field.type, FieldType)

    def test_field_validation_errors(self):
        """Test Field validation errors."""
        # Empty name raises SchemaError
        with pytest.raises(SchemaError, match="Field name cannot be empty"):
            Field(name="", type=FieldType.STRING)

        # Invalid type string raises ValueError
        with pytest.raises(ValueError):
            Field(name="test", type="invalid_type")


class TestSchema:
    """Tests for Schema dataclass."""

    def test_schema_required_fields_errors(self, sample_primary_key_field):
        """Test Schema required field validation errors."""
        # Empty table name
        with pytest.raises(SchemaError, match="Table name cannot be empty"):
            Schema(table_name="", fields=[sample_primary_key_field])

        # Empty fields list
        with pytest.raises(SchemaError, match="Schema must have at least one field"):
            Schema(table_name="test", fields=[])

    def test_schema_validation_rules(self):
        """Test Schema validation rules."""
        # No primary key - should warn but not error
        fields = [
            Field(name="name", type=FieldType.STRING),
            Field(name="value", type=FieldType.FLOAT),
        ]
        schema = Schema(table_name="test", fields=fields)
        assert len(schema.get_primary_key_fields()) == 0

        # Duplicate field names
        fields = [
            Field(name="id", type=FieldType.INTEGER, primary_key=True),
            Field(name="name", type=FieldType.STRING),
            Field(name="name", type=FieldType.STRING),  # Duplicate
        ]
        with pytest.raises(SchemaError, match="Field names must be unique"):
            Schema(table_name="test", fields=fields)

    def test_get_field(self, sample_schema):
        """Test getting fields by name."""
        # Existing field
        field = sample_schema.get_field("product_name")
        assert field is not None
        assert field.name == "product_name"
        assert field.type == FieldType.STRING

        # Non-existing field
        field = sample_schema.get_field("non_existent")
        assert field is None

    def test_get_primary_key_fields_multiple(self):
        """Test getting primary key fields when there are multiple."""
        fields = [
            Field(name="id1", type=FieldType.INTEGER, primary_key=True),
            Field(name="id2", type=FieldType.INTEGER, primary_key=True),
            Field(name="name", type=FieldType.STRING),
        ]
        schema = Schema(table_name="test", fields=fields)
        pk_fields = schema.get_primary_key_fields()
        assert len(pk_fields) == 2
        assert pk_fields[0].name == "id1"
        assert pk_fields[1].name == "id2"


class TestSchemaParser:
    """Tests for SchemaParser class."""

    def test_parse_dict_all_field_types(self, schema_dict_with_string_types):
        """Test parsing a dictionary with all field types."""
        schema = SchemaParser.parse_dict(schema_dict_with_string_types)
        assert len(schema.fields) == 7
        assert schema.fields[0].type == FieldType.INTEGER
        assert schema.fields[2].type == FieldType.FLOAT
        assert schema.fields[3].type == FieldType.BOOLEAN
        assert schema.fields[4].type == FieldType.STRING
        assert schema.fields[5].type == FieldType.DATE
        assert schema.fields[6].type == FieldType.DATETIME

    def test_parse_dict_structure_errors(self):
        """Test parsing errors for missing structure elements."""
        # Missing table_name
        data = {"fields": [{"name": "id", "type": "integer"}]}
        with pytest.raises(SchemaError, match="Schema must have a 'table_name' field"):
            SchemaParser.parse_dict(data)

        # Missing fields
        data = {"table_name": "test"}
        with pytest.raises(SchemaError, match="Schema must have a 'fields' array"):
            SchemaParser.parse_dict(data)

        # Empty fields
        data = {"table_name": "test", "fields": []}
        with pytest.raises(SchemaError, match="Schema must have a 'fields' array"):
            SchemaParser.parse_dict(data)

    def test_parse_dict_field_errors(self):
        """Test parsing errors for invalid fields."""
        # Field missing name
        data = {"table_name": "test", "fields": [{"type": "integer"}]}
        with pytest.raises(KeyError):
            SchemaParser.parse_dict(data)

        # Field missing type
        data = {"table_name": "test", "fields": [{"name": "id"}]}
        with pytest.raises(KeyError):
            SchemaParser.parse_dict(data)

    def test_parse_dict_with_optional_fields(self):
        """Test parsing fields with optional attributes (nullable, description)."""
        data = {
            "table_name": "test",
            "fields": [
                {
                    "name": "id",
                    "type": "integer",
                    "description": "Primary key",
                    "primary_key": True,
                },
                {"name": "name", "type": "string", "nullable": False, "description": "Name field"},
            ],
        }
        schema = SchemaParser.parse_dict(data)
        # Check descriptions
        assert schema.fields[0].description == "Primary key"
        assert schema.fields[1].description == "Name field"
        # Check nullable
        assert schema.fields[0].nullable is True  # Default
        assert schema.fields[1].nullable is False


class TestSchemaValidator:
    """Tests for SchemaValidator class."""

    def test_validate_comprehensive_dataframe(self, comprehensive_schema, comprehensive_dataframe):
        """Test validating a comprehensive DataFrame with all field types."""
        validator = SchemaValidator(comprehensive_schema)
        is_valid, errors = validator.validate(comprehensive_dataframe)
        assert is_valid is True
        assert errors == []

    def test_validate_missing_column(self, sample_schema):
        """Test validating a DataFrame with missing columns."""
        df = pd.DataFrame({"id": [1, 2], "product_name": ["A", "B"]})  # Missing 'price'
        validator = SchemaValidator(sample_schema)
        is_valid, errors = validator.validate(df)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_wrong_type(self, sample_schema):
        """Test validating a DataFrame with wrong data types."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "product_name": ["A", "B", "C"],
                "price": ["not", "a", "number"],  # Should be float
            }
        )
        validator = SchemaValidator(sample_schema)
        is_valid, errors = validator.validate(df)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_duplicate_primary_key(self, sample_schema):
        """Test validating a DataFrame with duplicate primary keys."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 2],  # Duplicate ID
                "product_name": ["A", "B", "C"],
                "price": [10.0, 20.0, 30.0],
            }
        )
        validator = SchemaValidator(sample_schema)
        is_valid, errors = validator.validate(df)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_nullable_constraints(self):
        """Test validating nullable and non-nullable field constraints."""
        # Nullable field with nulls - should be valid
        schema_nullable = Schema(
            table_name="test",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True, nullable=False),
                Field(name="name", type=FieldType.STRING, nullable=True),
            ],
        )
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", None, "C"]})
        validator = SchemaValidator(schema_nullable)
        is_valid, errors = validator.validate(df)
        assert is_valid is True

        # Non-nullable field with nulls - should be invalid
        schema_non_nullable = Schema(
            table_name="test",
            fields=[
                Field(name="id", type=FieldType.INTEGER, primary_key=True, nullable=False),
                Field(name="name", type=FieldType.STRING, nullable=False),
            ],
        )
        validator = SchemaValidator(schema_non_nullable)
        is_valid, errors = validator.validate(df)
        assert is_valid is False
        assert len(errors) > 0
