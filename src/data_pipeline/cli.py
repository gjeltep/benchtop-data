#!/usr/bin/env python3
"""
Simple CLI for the data pipeline.
"""

import argparse
import sys
import logging
from pathlib import Path
from data_pipeline import create_pipeline
from data_pipeline.logging import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset and enable natural language queries"
    )
    parser.add_argument("dataset", type=str, help="Path to dataset file (CSV or Parquet)")
    parser.add_argument("schema", type=str, help="Path to schema definition file (YAML or JSON)")
    parser.add_argument(
        "--db-path", type=str, default=None, help="Path to DuckDB file (default: in-memory)"
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=None,
        help="Path to Chroma persistence directory (default: in-memory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: from config)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama API base URL (default: from config)",
    )
    parser.add_argument("--query", type=str, help="Ask a question and exit")
    parser.add_argument(
        "--enable-reasoning-logs",
        action="store_true",
        help="Enable logging of reasoning tokens from reasoning-based LLMs (e.g., DeepSeek R1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show query execution details (which engine used, SQL generated, etc.)",
    )

    args = parser.parse_args()

    # Set up logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    # Validate paths
    dataset_path = Path(args.dataset)
    schema_path = Path(args.schema)

    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        sys.exit(1)

    # Create and run pipeline
    try:
        pipeline = create_pipeline(
            db_path=args.db_path,
            chroma_path=args.chroma_path,
            ollama_model=args.model,
            ollama_base_url=args.ollama_url,
            enable_reasoning_logs=args.enable_reasoning_logs,
        )

        logger.info("Processing dataset...")
        pipeline.process(dataset_path=str(dataset_path), schema_path=str(schema_path))
        logger.info("Dataset processed successfully!")

        if args.query:
            if args.verbose:
                # Get detailed execution metadata
                result = pipeline.ask(args.query, return_metadata=True)

                logger.info("=" * 60)
                logger.info("QUERY EXECUTION DETAILS")
                logger.info("=" * 60)

                metadata = result.get("metadata", {})

                # Show which engine was used
                engine_type = metadata.get("engine_type")
                if engine_type == "sql":
                    logger.info("SQL Engine Used")
                    if metadata.get("sql_query"):
                        logger.info(f"Query: {metadata['sql_query']}")
                elif engine_type == "vector":
                    logger.info("Vector Search Engine Used")
                else:
                    logger.info("Unknown Engine Used")

                # Show error if occurred
                if metadata.get("error"):
                    logger.warning(f"Error: {metadata['error']}")

                logger.info("")
                logger.info("=" * 60)
                logger.info("ANSWER:")
                logger.info("=" * 60)
                # Answer goes to stdout for piping
                print(result.get("answer", ""))
            else:
                # Simple output - answer only to stdout
                answer = pipeline.ask(args.query)
                print(answer)
        else:
            logger.info("\nReady for queries. Use --query to ask a question.")
            logger.info("Example: --query 'What products are out of stock?'")
            logger.info("Use --verbose or -v to see query execution details.")

        pipeline.close()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
