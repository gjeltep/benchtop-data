from typing import Optional


class DataPipelineError(Exception):
    """Base exception for all data pipeline errors."""

    def __init__(self, message: str, context: Optional[dict] = None):
        """
        Initialize exception with message and optional context.

        Args:
            message: Error message
            context: Optional dictionary with context (e.g., table_name, query, etc.)
        """
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            message = f"[{context_str}] {message}"
        super().__init__(message)
        self.context = context or {}


class SchemaError(DataPipelineError):
    """Raised when schema parsing or validation fails."""

    def __init__(self, message: str, schema_path: Optional[str] = None, **kwargs):
        """
        Initialize schema error.

        Args:
            message: Error message
            schema_path: Optional path to schema file
            **kwargs: Additional context
        """
        context = {**kwargs}
        if schema_path:
            context["schema_path"] = schema_path
        super().__init__(message, context=context or None)


class ValidationError(DataPipelineError):
    """Raised when data validation fails."""

    def __init__(self, message: str, table_name: Optional[str] = None, **kwargs):
        """
        Initialize validation error.

        Args:
            message: Error message
            table_name: Optional table name that failed validation
            **kwargs: Additional context
        """
        context = {**kwargs}
        if table_name:
            context["table_name"] = table_name
        super().__init__(message, context=context or None)


class StorageError(DataPipelineError):
    """Raised when storage operations fail."""

    def __init__(
        self,
        message: str,
        table_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize storage error.

        Args:
            message: Error message
            table_name: Optional table name involved in the operation
            operation: Optional operation name (e.g., "insert", "query", "create_table")
            **kwargs: Additional context
        """
        context = {**kwargs}
        if table_name:
            context["table_name"] = table_name
        if operation:
            context["operation"] = operation
        super().__init__(message, context=context or None)


class QueryError(DataPipelineError):
    """Raised when query operations fail."""

    def __init__(
        self, message: str, query: Optional[str] = None, engine_type: Optional[str] = None, **kwargs
    ):
        """
        Initialize query error.

        Args:
            message: Error message
            query: Optional query that failed
            engine_type: Optional engine type (e.g., "sql", "vector")
            **kwargs: Additional context
        """
        context = {**kwargs}
        if query:
            # Truncate long queries for readability
            query_preview = query[:100] + "..." if len(query) > 100 else query
            context["query"] = query_preview
        if engine_type:
            context["engine_type"] = engine_type
        super().__init__(message, context=context or None)


class ConfigurationError(DataPipelineError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_key: Optional configuration key that is invalid
            **kwargs: Additional context
        """
        context = {**kwargs}
        if config_key:
            context["config_key"] = config_key
        super().__init__(message, context=context or None)
