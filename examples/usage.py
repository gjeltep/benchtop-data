"""
Example usage of the data pipeline.
"""

from data_pipeline import create_pipeline


# Example 1: Basic usage
def example_basic():
    """Basic example of processing a dataset."""
    # Create pipeline
    pipeline = create_pipeline(db_path="data.db", chroma_path="chroma_db", ollama_model="mistral")

    # Process dataset
    pipeline.process(
        dataset_path="examples/products.csv", schema_path="examples/products_schema.yaml"
    )

    # Ask questions
    answer = pipeline.ask("What products are out of stock?")
    print(answer)  # SQL

    answer = pipeline.ask("What is the average price by category?")
    print(answer)

    answer = pipeline.ask("List all products in the Electronics category")
    print(answer)  # Semantic

    # Clean up
    pipeline.close()


if __name__ == "__main__":
    example_basic()
