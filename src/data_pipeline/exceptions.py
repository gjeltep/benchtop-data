"""
Custom exceptions for the data pipeline.
"""


class DataPipelineError(Exception):
    """Base exception for all data pipeline errors."""

    pass


class SchemaError(DataPipelineError):
    """Raised when schema parsing or validation fails."""

    pass


class ValidationError(DataPipelineError):
    """Raised when data validation fails."""

    pass


class StorageError(DataPipelineError):
    """Raised when storage operations fail."""

    pass


class QueryError(DataPipelineError):
    """Raised when query operations fail."""

    pass


class ConfigurationError(DataPipelineError):
    """Raised when configuration is invalid."""

    pass
