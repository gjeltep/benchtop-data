# CRITICAL: Apply nest_asyncio FIRST, before any other imports
# This must happen before Streamlit or any async code is imported
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from pathlib import Path
import pandas as pd
import tempfile
import yaml
from data_pipeline import create_pipeline
from data_pipeline.config import config
from data_pipeline.logging_config import setup_logging

# Initialize logging with HTTP log suppression
setup_logging(suppress_http_logs=True)

@st.cache_data
def load_preview(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Cache file preview to avoid re-reading on every rerun."""
    if filename.endswith(".csv"):
        return pd.read_csv(pd.io.common.BytesIO(file_bytes))
    return pd.read_parquet(pd.io.common.BytesIO(file_bytes))

def main():
    st.set_page_config(page_title="Data Pipeline", page_icon="📊", layout="wide")

    st.title("📊 Benchtop Data")
    st.markdown("Upload a dataset and schema, then ask natural language questions about your data.")

    # Initialize session state
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "table_name" not in st.session_state:
        st.session_state.table_name = None
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = config.llm_model
    if "ollama_url" not in st.session_state:
        st.session_state.ollama_url = config.ollama_url
    # Invalidate existing pipeline if code version changed (ensures new workflow logic is applied)
    existing_pipeline = st.session_state.get("pipeline")
    if existing_pipeline is None:
        st.session_state.processed = False
        st.session_state.table_name = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        ollama_model = st.text_input(
            "Ollama Model",
            value=st.session_state.ollama_model,
            help="Name of the Ollama model to use",
            key="ollama_model_input",
        )
        st.session_state.ollama_model = ollama_model

        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.ollama_url,
            help="Base URL for Ollama API",
            key="ollama_url_input",
        )
        st.session_state.ollama_url = ollama_url

        use_persistent = st.checkbox(
            "Persist Data", value=False, help="Save database and embeddings to disk"
        )

        db_path = None
        chroma_path = None
        if use_persistent:
            db_path = st.text_input("DuckDB Path", value="data.db")
            chroma_path = st.text_input("Chroma Path", value="chroma_db")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "💬 Query", "📋 Dataset Info"])

    with tab1:
        st.header("Upload Dataset and Schema")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset File")
            dataset_file = st.file_uploader(
                "Upload CSV or Parquet file",
                type=["csv", "parquet"],
                help="Upload your dataset file",
            )

            if dataset_file is not None:
                st.success(f"✅ Uploaded: {dataset_file.name}")

                # Preview dataset (cached to avoid re-reading)
                df_preview = load_preview(dataset_file.getvalue(), dataset_file.name)
                st.dataframe(df_preview.head(10), width="stretch")
                st.caption(f"Preview: {len(df_preview)} rows, {len(df_preview.columns)} columns")

        with col2:
            st.subheader("Schema File")
            schema_file = st.file_uploader(
                "Upload YAML or JSON schema",
                type=["yaml", "yml", "json"],
                help="Upload your schema definition file",
            )

            if schema_file is not None:
                st.success(f"✅ Uploaded: {schema_file.name}")

                # Preview schema (cache content to avoid re-reading)
                schema_bytes = schema_file.getvalue()
                schema_content = schema_bytes.decode("utf-8")
                st.code(
                    schema_content,
                    language="yaml" if schema_file.name.endswith((".yaml", ".yml")) else "json",
                )

        # Process button
        if dataset_file is not None and schema_file is not None:
            st.divider()

            if st.button("🚀 Process Dataset", type="primary", use_container_width=True):
                with st.spinner("Processing dataset..."):
                    try:
                        # Save uploaded files temporarily (cross-platform temp dir)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(dataset_file.name).suffix) as f:
                            f.write(dataset_file.getvalue())
                            dataset_path = f.name

                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(schema_file.name).suffix) as f:
                            f.write(schema_file.getvalue())
                            schema_path = f.name

                        # Parse schema for table name (reuse already-loaded content)
                        schema_data = yaml.safe_load(schema_content)
                        table_name = schema_data.get("table_name", "unknown")

                        # Create pipeline
                        pipeline = create_pipeline(
                            db_path=db_path,
                            chroma_path=chroma_path,
                            ollama_model=st.session_state.ollama_model,
                            ollama_base_url=st.session_state.ollama_url
                        )

                        # Process dataset
                        pipeline.process(dataset_path=dataset_path, schema_path=schema_path)

                        # Store in session state
                        st.session_state.pipeline = pipeline
                        st.session_state.processed = True
                        st.session_state.table_name = table_name

                        # Check for missing primary key and warn user
                        if not pipeline.schema.get_primary_key_fields():
                            st.warning(
                                "⚠️ No primary key defined in schema. "
                                "Auto-generated IDs will be used. "
                                "Consider adding a primary key field for better data tracking."
                            )

                        st.success("✅ Dataset processed successfully!")
                        st.balloons()

                    except FileNotFoundError as e:
                        st.error(f"❌ File not found: {e.filename}")
                    except yaml.YAMLError as e:
                        st.error(f"❌ Invalid schema file: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Processing failed: {type(e).__name__}: {str(e)}")
                        st.exception(e)

    with tab2:
        st.header("Ask Questions")

        if not st.session_state.processed:
            st.info("👆 Please upload and process a dataset first in the 'Upload & Process' tab.")
        else:
            # Create two columns: main chat area and sidebar
            col_main, col_sidebar = st.columns([3, 1])

            with col_sidebar:
                st.subheader("💬 Chat History")

                # Clear history button
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.pipeline.clear_chat_history()
                    st.rerun()

                # Display chat history
                history = st.session_state.pipeline.get_chat_history()
                if history:
                    st.caption(f"{len(history) // 2} messages")
                    with st.container(height=400):
                        for msg in history:
                            role_emoji = "👤" if msg["role"] == "user" else "🤖"
                            role_label = "You" if msg["role"] == "user" else "Assistant"

                            with st.chat_message(msg["role"]):
                                st.caption(f"{role_emoji} **{role_label}**")
                                # Truncate long messages in history
                                content = msg["content"]
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                st.write(content)
                else:
                    st.info("No conversation history yet. Start asking questions!")

            with col_main:
                st.success(f"✅ Ready to query: {st.session_state.table_name}")

                # Example questions
                st.subheader("💡 Example Questions")
                example_questions = [
                    "What are the main insights from this dataset?",
                    "What is the average value?",
                    "What are the unique categories?",
                    "Show me summary statistics",
                ]

                selected_example = st.selectbox("Or try an example:", [""] + example_questions)

                # Query input form (enables Cmd+Enter / Ctrl+Enter to submit)
                with st.form(key="query_form", clear_on_submit=False):
                    query = st.text_area(
                        "Enter your question:",
                        value=selected_example if selected_example else "",
                        height=100,
                        placeholder="e.g., What products are out of stock?",
                        help="Press Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux) to submit",
                    )

                    submitted = st.form_submit_button(
                        "🔍 Ask", type="primary", use_container_width=True
                    )

                if submitted and query.strip():
                    with st.spinner("Thinking..."):
                        try:
                            # Get answer with metadata for observability
                            result = st.session_state.pipeline.ask(query, return_metadata=True)

                            if isinstance(result, dict):
                                answer = result["answer"]
                                metadata = result["metadata"]
                            else:
                                # Fallback for backwards compatibility
                                answer = result
                                metadata = None

                            st.subheader("Answer:")
                            st.write(answer)

                            # Show query execution details
                            with st.expander("🔍 Query Details", expanded=False):
                                if metadata:
                                    # Show engine type (now explicitly tracked)
                                    engine_type = metadata.get("engine_type", "unknown")
                                    if engine_type == "sql":
                                        st.info("🔧 SQL Engine Used")
                                    elif engine_type == "vector":
                                        st.info("🔍 Vector Search Engine Used")
                                    else:
                                        st.info(f"🔧 Engine: {engine_type}")

                                    # Show SQL query if captured
                                    if metadata.get("sql_query"):
                                        st.code(metadata["sql_query"], language="sql")

                                    # Show reasoning tokens if captured
                                    if metadata.get("reasoning"):
                                        with st.expander("🧠 Model Reasoning", expanded=False):
                                            st.text(metadata["reasoning"])

                                    # Show errors if any
                                    if metadata.get("error"):
                                        st.error(f"Error: {metadata['error']}")

                                    st.caption(f"Model: {st.session_state.ollama_model}")
                                else:
                                    st.code(f"Model: {st.session_state.ollama_model}\nQuery: {query}")

                        except ConnectionError:
                            st.error(f"❌ Cannot connect to Ollama at {st.session_state.ollama_url}. Is it running?")
                        except KeyError as e:
                            st.error(f"❌ Missing required field: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Query failed: {type(e).__name__}: {str(e)}")
                            st.exception(e)

    with tab3:
        st.header("Dataset Information")

        if not st.session_state.processed:
            st.info("👆 Please upload and process a dataset first in the 'Upload & Process' tab.")
        else:
            try:
                pipeline = st.session_state.pipeline

                # Get dataset info
                table_name = st.session_state.table_name
                df = pipeline.storage_repo.query(f"SELECT * FROM {table_name}")

                st.subheader(f"Dataset: {table_name}")
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))

                st.subheader("Data Preview")
                st.dataframe(df.head(20), width="stretch")

                st.subheader("Column Information")
                col_info = pd.DataFrame(
                    {
                        "Column": df.columns,
                        "Type": [str(df[col].dtype) for col in df.columns],
                        "Non-Null Count": [df[col].count() for col in df.columns],
                        "Null Count": [df[col].isna().sum() for col in df.columns],
                    }
                )
                st.dataframe(col_info, width="stretch")

                st.subheader("Summary Statistics")
                st.dataframe(df.describe(), width="stretch")

            except KeyError as e:
                st.error(f"❌ Table not found: {str(e)}")
            except Exception as e:
                st.error(f"❌ Failed to load dataset: {type(e).__name__}: {str(e)}")
                st.exception(e)

    # NOTE: Cleanup on app close - Streamlit doesnt seem to have it but may need to use session state more carefully
    if st.session_state.pipeline is not None:
        pass

if __name__ == "__main__":
    main()
