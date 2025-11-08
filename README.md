# Benchtop Data

A few ideas have been of interest recently, and wanted to explore a bit:
- what's possible to set up locally for analyzing credit transactions?
- small (<10B) parameter models can certainly provide interesting behavior - what sort of mileage can they provide?
- generalizing specific need to a simple E2E data analysis pipeline

Those explorations ultimately yielded a fairly general-purpose, local AI data analysis pipeline capable of processing datasets, validating them against schemas, storing them in DuckDB, capturing semantic embeddings, and offering a natural language querying interface. And - should you choose - without token burn!

## Features

- **Schema-based validation**: Define datasets using YAML/JSON schemas
- **DuckDB storage**: Fast, local SQL database for structured data
- **Semantic embeddings**: Automatic text generation and Chroma vector storage
- **Natural language queries**: Ask questions using LlamaIndex + Ollama (I've been defaulting to 8B Mistral)

## Installation

### Setup

1. Install dependencies:
```bash
uv pip install -e .
```

2. Verify Ollama is running:
```bash
ollama list
```
Make sure you've pulled the model of your choice.

## Quick Start

Take a look in `examples/usage.py` to get going.

## Web UI (Streamlit)

The easiest way to use the pipeline is through the Streamlit web interface:

```bash
uv run streamlit run src/data_pipeline/ui.py
```

Or after installation:
```bash
streamlit-ui
```

## CLI Usage

For command-line usage:

```bash
# Process a dataset
data-pipeline examples/products.csv examples/products_schema.yaml \
    --db-path data.db \
    --chroma-path chroma_db

# Process and ask a question
data-pipeline examples/products.csv examples/products_schema.yaml \
    --query "What products are out of stock?"
```

## Schema Format

Schemas support:

- **Field types**: `string`, `integer`, `float`, `boolean`, `date`, `datetime`
- **Constraints**: `nullable`, `primary_key`
- **Metadata**: `description` for semantic meaning
- **Relationships**: Foreign key relationships between tables

## License

MIT

