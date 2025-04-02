**Feature Spec: dsRAG Logging Framework**

**1. Overview / Goal**

This specification details the implementation of a flexible and standardized logging framework within the `dsrag` library. The goal is to provide users and developers with detailed insights into the library's operations (document ingestion, querying, chat interactions) including performance metrics (latency), configuration parameters, intermediate results, and errors, while allowing end-users full control over log handling and formatting.

**2. Motivation / Rationale**

*   **Observability:** Enable users and maintainers to understand the behavior, performance, and potential issues within `dsrag` workflows.
*   **Debugging:** Provide detailed step-by-step information to diagnose problems during ingestion, querying, or chat.
*   **Performance Analysis:** Capture latency information for key steps to identify bottlenecks.
*   **AI Behavior Insight:** Log parameters (models used, RSE settings) and intermediate results (query counts, segment scores) to understand the RAG process.
*   **Flexibility for Users:** As an open-source library, `dsrag` should not dictate a specific logging setup. Users must be able to integrate `dsrag` logs into their existing application logging infrastructure (e.g., console, file, ELK, Splunk, Datadog, custom handlers).
*   **Standard Compliance:** Adhere to Python best practices for library logging using the built-in `logging` module.
*   **Lightweight:** Avoid adding heavy dependencies solely for logging.

**Why this Approach?**

*   **Built-in `logging`:** It's the Python standard, requires no extra dependencies for `dsrag`, and provides maximum flexibility for the *user* to configure handlers, formatters, and filters.
*   **`extra` for Structured Data:** Allows passing machine-readable context (IDs, metrics, parameters) alongside human-readable messages without breaking standard formatters. Crucial for log analysis tools.
*   **Operation IDs for Correlation:** Enables users to easily filter and view all logs related to a specific document ingestion, query, or chat message, even in concurrent environments, using external tools.
*   **No Internal Consolidation:** Avoids buffering logs within `dsrag`, ensuring real-time visibility, preventing log loss on crashes, simplifying `dsrag`'s internal logic, and working correctly with standard log management systems that expect individual event streams.

**3. Core Design**

1.  **Use `logging` Module:** All logging within `dsrag` will utilize Python's standard `logging` module.
2.  **Hierarchical Loggers:** Loggers will be obtained using `logging.getLogger()`, typically following a hierarchical structure like `dsrag`, `dsrag.ingestion`, `dsrag.query`, `dsrag.chat`, `dsrag.ingestion.dsparse`.
3.  **`NullHandler` Default:** The top-level `dsrag` logger will have `logging.NullHandler` added by default (in `dsrag/__init__.py`) to prevent "No handler found" warnings if the user hasn't configured logging. The library itself will *not* configure `StreamHandler`, `FileHandler`, or any specific formatting.
4.  **Structured Data via `extra`:** Key contextual information (operation IDs, parameters, metrics, timings) will be passed via the `extra` dictionary in logging calls (`logger.info("...", extra=...)`).
5.  **Operation Identifiers:** Unique IDs will be generated or retrieved and consistently included in the `extra` data for all logs within a specific operation's scope:
    *   `kb_id`: Included in almost all logs.
    *   `doc_id`: Included in all logs related to a specific document's ingestion.
    *   `query_id`: A unique ID (e.g., UUID) generated for each `KnowledgeBase.query` call, included in all logs during that query process.
    *   `thread_id`: The chat thread ID, included in all logs related to `get_chat_thread_response`.
    *   `message_id`: The unique ID for a specific user/assistant turn within a chat thread, included in all logs during the processing of that turn.

**4. Implementation Details**

*   **Logger Naming:**
    *   Root: `logging.getLogger("dsrag")`
    *   Sub-modules: `logging.getLogger("dsrag.knowledge_base")`, `logging.getLogger("dsrag.query")`, `logging.getLogger("dsrag.ingestion")`, `logging.getLogger("dsrag.chat")`, `logging.getLogger("dsrag.dsparse")`, etc.
*   **Log Levels:**
    *   `DEBUG`: Detailed step timings, parameters, intermediate results/metrics (e.g., individual segment scores, chunk counts, model names used). Intended for deep dives and debugging `dsrag` itself.
    *   `INFO`: Major operation start/end points, key outcomes (e.g., ingestion complete for doc X, query returned Y segments, chat response sent). Intended for general application monitoring.
    *   `WARNING`: Potential issues that don't prevent operation (e.g., fallback behavior, unexpected config).
    *   `ERROR`: Operation failed but the application might recover (e.g., specific API call failed during ingestion).
    *   `CRITICAL`: Severe errors likely causing application failure.
*   **Latency Measurement:** Use `time.perf_counter()` for accurate duration measurement. Log durations consistently (in seconds as floats with 4 decimal points of precision).
*   **Error Logging:** Wrap potentially failing operations (API calls, DB interactions) in `try...except` blocks. Log errors using `logger.error("...", exc_info=True)` within the `except` block, ensuring relevant operation IDs are included in `extra`.
*   **Sensitive Data:** Avoid logging full document content, large text chunks, or raw user input/LLM output by default at INFO level. Use previews (`text[:50] + "..."`) or log lengths/counts instead. Logging full content at the DEBUG level is okay. API keys must *never* be logged.
*   **Key Logging Points & Data (`extra` fields):**

    *   **Document Ingestion (`KnowledgeBase.add_document`)**
        *   `kb_id`, `doc_id`, `file_path` (if applicable)
        *   **Start (INFO):** Log initiation.
        *   **Params (DEBUG):** Log full config dictionaries.
        *   **Steps (DEBUG):** For Parsing, Sectioning, Chunking, AutoContext, Embedding, DB Add:
            *   `step`: Name of the step (e.g., "parsing", "embedding").
            *   `duration_s`: Duration of the step.
            *   Relevant metrics (e.g., `num_elements`, `num_sections`, `num_chunks`, `num_embeddings`).
        *   **End (INFO):** Log completion, `total_duration_s`.
        *   **Error (ERROR):** Log failure, `step` where failure occurred, `exc_info=True`.

    *   **Querying (`KnowledgeBase.query`)**
        *   `query_id`, `kb_id`
        *   **Start (INFO):** Log initiation, `num_search_queries`.
        *   **Params (DEBUG):** Log `rse_params`, `metadata_filter` (if present), `reranker.model`, `search_queries`.
        *   **Steps (DEBUG):**
            *   `search_rerank`: `duration_s`, `num_initial_results`. Relevance scores of top 5 results.
            *   `rse`: `duration_s`, `num_final_segments`, `segment_scores` (list of floats).
        *   **End (INFO):** Log completion, `num_final_segments`, `total_duration_s`.
        *   **Error (ERROR):** Log failure, `step`, `exc_info=True`.

    *   **Chat (`get_chat_thread_response`)**
        *   `thread_id`, `message_id`, `kb_ids` (list)
        *   **Start (INFO):** Log initiation.
        *   **Params (DEBUG):** Log `chat_thread_params` and `user_input`. Log number of messages in chat thread so far (if available w/o additional database query).
        *   **Steps (DEBUG):**
            *   `auto_query`: `duration_s`, `num_queries_generated`, `queries` (list of dicts).
            *   `kb_search`: `duration_s`, `num_segments_retrieved` (total).
            *   `llm_response`: `duration_s`, `model_name`, `response_length` (if easily available), `full_response`.
        *   **End (INFO):** Log completion, `total_duration_s`.
        *   **Error (ERROR):** Log failure, `step`, `exc_info=True`.

**5. Code Examples (Illustrative)**

```python
# dsrag/knowledge_base.py
import logging
import time
import uuid

logger = logging.getLogger("dsrag.knowledge_base") # Or more specific

class KnowledgeBase:
    # ... (init) ...

    def add_document(self, doc_id: str, file_path: str | None = None, config: dict = None, **kwargs):
        ingestion_logger = logging.getLogger("dsrag.ingestion")
        base_extra = {"kb_id": self.kb_id, "doc_id": doc_id}
        if file_path:
            base_extra["file_path"] = file_path

        ingestion_logger.info("Starting document ingestion", extra=base_extra)
        ingestion_logger.debug("Ingestion parameters", extra={
            **base_extra,
            "config": config,
            "kwargs": kwargs
        })
        overall_start_time = time.perf_counter()

        try:
            # --- Parsing Step ---
            step_start_time = time.perf_counter()
            sections, chunks = parse_and_chunk(...) # Assume this function might log internally too
            step_duration = time.perf_counter() - step_start_time
            ingestion_logger.debug("Parsing and Chunking complete", extra={
                **base_extra, "step": "parse_chunk", "duration_s": round(step_duration, 4),
                "num_sections": len(sections), "num_chunks": len(chunks)
            })

            # --- AutoContext Step ---
            step_start_time = time.perf_counter()
            chunks, chunks_to_embed = auto_context(...)
            step_duration = time.perf_counter() - step_start_time
            ingestion_logger.debug("AutoContext complete", extra={
                **base_extra, "step": "auto_context", "duration_s": round(step_duration, 4),
                "model": self.auto_context_model.__class__.__name__ # Log class name
            })

            # --- Embedding Step ---
            step_start_time = time.perf_counter()
            chunk_embeddings = get_embeddings(...)
            step_duration = time.perf_counter() - step_start_time
            ingestion_logger.debug("Embedding complete", extra={
                **base_extra, "step": "embedding", "duration_s": round(step_duration, 4),
                "num_embeddings": len(chunk_embeddings), "model": self.embedding_model.model
            })

            # --- DB Add Step ---
            step_start_time = time.perf_counter()
            add_chunks_to_db(...)
            add_vectors_to_db(...)
            step_duration = time.perf_counter() - step_start_time
            ingestion_logger.debug("Database add complete", extra={
                **base_extra,
                "step": "db_add", "duration_s": round(step_duration, 4)
            })

            overall_duration = time.perf_counter() - overall_start_time
            ingestion_logger.info("Document ingestion successful", extra={
                **base_extra,
                "total_duration_s": round(overall_duration, 4)
            })

        except Exception as e:
            overall_duration = time.perf_counter() - overall_start_time
            ingestion_logger.error("Document ingestion failed", extra={
                **base_extra,
                "total_duration_s": round(overall_duration, 4)
            }, exc_info=True)
            # Re-raise or handle as appropriate for the library's design
            raise

    def query(self, search_queries: list[str], rse_params: dict, metadata_filter: dict | None = None, **kwargs):
        query_logger = logging.getLogger("dsrag.query")
        query_id = str(uuid.uuid4())
        base_extra = {"kb_id": self.kb_id, "query_id": query_id}

        query_logger.info("Starting query", extra={**base_extra, "num_search_queries": len(search_queries)})
        query_logger.debug("Query parameters", extra={
            **base_extra,
            "search_queries": search_queries,
            "rse_params": rse_params,
            "metadata_filter": metadata_filter,
            "reranker_model": self.reranker.model if hasattr(self.reranker, 'model') else self.reranker.__class__.__name__,
            "kwargs": kwargs
        })
        overall_start_time = time.perf_counter()

        try:
            # --- Search/Rerank Step ---
            step_start_time = time.perf_counter()
            all_ranked_results = self._get_all_ranked_results(...)
            step_duration = time.perf_counter() - step_start_time
            query_logger.debug("Search/Rerank complete", extra={
                **base_extra, "step": "search_rerank", "duration_s": round(step_duration, 4),
                "num_initial_results_per_query": [len(r) for r in all_ranked_results], # Example metric
                "reranker": self.reranker.__class__.__name__
            })

            # --- RSE Step ---
            step_start_time = time.perf_counter()
            # ... RSE logic ...
            best_segments, scores = get_best_segments(...)
            step_duration = time.perf_counter() - step_start_time
            query_logger.debug("RSE complete", extra={
                **base_extra,
                "step": "rse", "duration_s": round(step_duration, 4),
                "num_final_segments": len(best_segments), "segment_scores": [round(s, 4) for s in scores]
            })

            # --- Content Retrieval Step ---
            step_start_time = time.perf_counter()
            # ... _get_segment_content_from_database loop ...
            step_duration = time.perf_counter() - step_start_time
            query_logger.debug("Content retrieval complete", extra={
                 **base_extra, "step": "content_retrieval", "duration_s": round(step_duration, 4)
            })

            overall_duration = time.perf_counter() - overall_start_time
            query_logger.info("Query successful", extra={
                **base_extra, "total_duration_s": round(overall_duration, 4), "num_final_segments": len(best_segments)
            })
            return relevant_segment_info

        except Exception as e:
            overall_duration = time.perf_counter() - overall_start_time
            query_logger.error("Query failed", extra={
                **base_extra, "total_duration_s": round(overall_duration, 4)
            }, exc_info=True)
            # Re-raise or return empty list based on desired behavior
            raise

```

**6. User Configuration Examples**

*   **Basic Console Logging (User's application code):**
    ```python
    import logging

    # Configure basic logging to see INFO level messages from dsrag
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # To see detailed debug messages:
    # logging.getLogger("dsrag").setLevel(logging.DEBUG)
    # logging.getLogger("dsrag.query").setLevel(logging.DEBUG) # Or specific sub-loggers

    # --- Now use dsrag ---
    # kb = KnowledgeBase(...)
    # kb.add_document(...)
    # results = kb.query(...)
    ```

*   **JSON Logging to Console (User's application code):**
    ```python
    import logging
    import sys
    from pythonjsonlogger import jsonlogger # Requires pip install python-json-logger

    log = logging.getLogger() # Get root logger or specific 'dsrag' logger
    log.setLevel(logging.DEBUG) # Set level for logs you want to capture

    handler = logging.StreamHandler(sys.stdout)
    # Include standard fields + automatically includes fields from 'extra'
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d',
        datefmt='%Y-%m-%dT%H:%M:%S%z'
    )
    handler.setFormatter(formatter)

    # Clear existing handlers if configuring root logger
    if log.hasHandlers():
        log.handlers.clear()
    log.addHandler(handler)

    # Prevent dsrag's NullHandler from blocking if we configure the root/dsrag logger
    logging.getLogger("dsrag").propagate = True # Or remove the NullHandler if configuring 'dsrag' directly

    # --- Now use dsrag ---
    # kb = KnowledgeBase(...)
    # kb.add_document(...) # Logs will appear as JSON lines
    # results = kb.query(...)
    ```
    *User can then pipe this output to a file or log shipper.*

**7. Benefits**

*   Provides detailed visibility into `dsrag` internals.
*   Enables effective debugging and performance tuning.
*   Allows users to track operations using unique IDs.
*   Integrates seamlessly with standard Python logging and external log management systems.
*   Maintains library flexibility by not imposing specific logging configurations.

---