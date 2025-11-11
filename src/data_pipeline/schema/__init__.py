"""
Schema definition and validation module.

This module handles parsing and validation of schema definitions.
Uses Pandera for robust DataFrame validation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import yaml
import json
import pandas as pd
from pandera.pandas import Column, DataFrameSchema
from pandera.errors import SchemaError as PanderaSchemaError, SchemaErrors as PanderaSchemaErrors
from ..exceptions import SchemaError
from ..logging import get_logger

logger = get_logger(__name__)


class FieldType(Enum):
    """Supported field types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"


@dataclass
class Field:
    """Represents a field definition in the schema."""

    name: str
    type: FieldType
    description: Optional[str] = None
    nullable: bool = True
    primary_key: bool = False
    format: Optional[str] = None  # For date/datetime parsing (e.g., "%m/%d/%Y")

    def __post_init__(self):
        """Validate field definition."""
        if not self.name:
            raise SchemaError("Field name cannot be empty")
        if isinstance(self.type, str):
            self.type = FieldType(self.type)


@dataclass
class Relationship:
    """Represents a relationship between tables."""

    from_field: str
    to_table: str
    to_field: str
    description: Optional[str] = None


@dataclass
class Schema:
    """Represents a complete schema definition."""

    table_name: str
    fields: List[Field]
    relationships: Optional[List[Relationship]] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate schema definition."""
        if not self.table_name:
            raise SchemaError("Table name cannot be empty")
        if not self.fields:
            raise SchemaError("Schema must have at least one field")

        primary_keys = [f for f in self.fields if f.primary_key]
        if not primary_keys:
            logger.warning(
                f"Schema '{self.table_name}' has no primary key defined. "
                "Auto-generated IDs will be used for embeddings. "
                "Consider adding a primary_key field for better data identity."
            )

        # Ensure field names are unique
        field_names = [f.name for f in self.fields]
        if len(field_names) != len(set(field_names)):
            raise SchemaError("Field names must be unique")

    def get_field(self, name: str) -> Optional[Field]:
        """Get a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_primary_key_fields(self) -> List[Field]:
        """Get all primary key fields."""
        return [f for f in self.fields if f.primary_key]


class SchemaParser:
    """Parser for schema definition files (YAML/JSON)."""

    @staticmethod
    def parse_file(file_path: str) -> Schema:
        """Parse a schema file (YAML or JSON)."""
        try:
            with open(file_path, "r") as f:
                if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                    data = yaml.safe_load(f)
                elif file_path.endswith(".json"):
                    data = json.load(f)
                else:
                    raise SchemaError(f"Unsupported file format: {file_path}", schema_path=file_path)

            return SchemaParser.parse_dict(data)
        except FileNotFoundError:
            raise SchemaError(f"Schema file not found: {file_path}", schema_path=file_path)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise SchemaError(f"Failed to parse schema file: {e}", schema_path=file_path)

    @staticmethod
    def parse_dict(data: Dict[str, Any]) -> Schema:
        """Parse a schema from a dictionary."""
        table_name = data.get("table_name")
        if not table_name:
            raise SchemaError("Schema must have a 'table_name' field")

        fields_data = data.get("fields", [])
        if not fields_data:
            raise SchemaError("Schema must have a 'fields' array")

        fields = []
        for field_data in fields_data:
            field = Field(
                name=field_data["name"],
                type=FieldType(field_data["type"]),
                description=field_data.get("description"),
                nullable=field_data.get("nullable", True),
                primary_key=field_data.get("primary_key", False),
                format=field_data.get("format"),
            )
            fields.append(field)

        relationships = None
        if "relationships" in data:
            relationships = [
                Relationship(
                    from_field=rel["from_field"],
                    to_table=rel["to_table"],
                    to_field=rel["to_field"],
                    description=rel.get("description"),
                )
                for rel in data["relationships"]
            ]

        return Schema(
            table_name=table_name,
            fields=fields,
            relationships=relationships,
            description=data.get("description"),
        )


class SchemaValidator:
    """Validates datasets against schema definitions using Pandera."""

    def __init__(self, schema: Schema):
        """Initialize validator with a schema."""
        self.schema = schema
        self.pandera_schema = self._create_pandera_schema()

    def _map_field_type_to_dtype(self, field_type: FieldType) -> Any:
        """Map our FieldType enum to pandas/pandera dtype."""
        type_mapping = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: float,
            FieldType.BOOLEAN: bool,
            FieldType.DATE: "datetime64[ns]",
            FieldType.DATETIME: "datetime64[ns]",
        }
        return type_mapping.get(field_type, str)

    def _create_pandera_schema(self) -> DataFrameSchema:
        """Create a Pandera DataFrameSchema from our Schema definition."""
        columns = {}
        primary_key_fields = []

        for field in self.schema.fields:
            dtype = self._map_field_type_to_dtype(field.type)
            nullable = field.nullable

            # Create column with type and nullability
            columns[field.name] = Column(
                dtype,
                nullable=nullable,
                description=field.description,
            )

            if field.primary_key:
                primary_key_fields.append(field.name)

        # Create schema with uniqueness constraint for primary keys
        schema_kwargs = {"columns": columns}

        if primary_key_fields:
            if len(primary_key_fields) == 1:
                # Single column primary key
                schema_kwargs["unique"] = primary_key_fields[0]
            else:
                # Composite primary key - use unique_subset
                schema_kwargs["unique_subset"] = primary_key_fields

        return DataFrameSchema(**schema_kwargs)

    def validate(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Validate a pandas DataFrame against the schema using Pandera.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Pandera validates; pandera gives detailed error messages
            self.pandera_schema.validate(df, lazy=True)
            return True, []
        except (PanderaSchemaError, PanderaSchemaErrors) as e:
            # Pandera provides detailed error messages
            logger.debug(f"Validation failed: {e}")
            # SchemaErrors has schema_errors attribute, SchemaError doesn't
            error_list = getattr(e, 'schema_errors', [e])
            for error in error_list:
                errors.append(str(error))
            return False, errors
