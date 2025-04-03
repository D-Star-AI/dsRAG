# dsRAG Logging Framework

This document provides an overview of the logging framework implemented in dsRAG.

## Overview

The dsRAG logging framework provides detailed visibility into the library's operations, including document ingestion, querying, and chat interactions. It uses Python's standard `logging` module, making it easy to integrate with existing application logging.

## Key Features

- **Standard Python Logging**: Uses the built-in `logging` module
- **Hierarchical Loggers**: Organized by component (`dsrag.ingestion`, `dsrag.query`, `dsrag.chat`)
- **Structured Data**: Uses the `extra` parameter to include structured data with log messages
- **Operation IDs**: Includes unique IDs to correlate logs from the same operation
- **Comprehensive Metrics**: Tracks durations, counts, and other metrics
- **Configurable Levels**: Different verbosity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Logger Hierarchy

- `dsrag`: Root logger
- `dsrag.ingestion`: Document ingestion operations
- `dsrag.query`: Knowledge base querying operations
- `dsrag.chat`: Chat interactions

## Log Levels

- **DEBUG**: Detailed step timings, parameters, intermediate results
- **INFO**: Major operation start/end points, key outcomes
- **WARNING**: Potential issues that don't prevent operation
- **ERROR**: Operation failures that might be recoverable
- **CRITICAL**: Severe errors likely causing application failure

## Configuring Logging

### Basic Configuration

```python
import logging

# Configure basic logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use dsRAG normally
from dsrag.knowledge_base import KnowledgeBase
kb = KnowledgeBase("example_kb")
```

### Detailed Configuration

```python
import logging
import sys

# Set up a detailed formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'
)

# Create a handler for console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Configure the dsrag logger
logger = logging.getLogger("dsrag")
logger.setLevel(logging.DEBUG)

# Clear existing handlers and add our handler
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(console_handler)
```

### JSON Logging

For structured logging suitable for log analysis systems:

```python
import logging
import sys
from pythonjsonlogger import jsonlogger

# Create a JSON formatter
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d'
)

# Create a handler for console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Configure the dsrag logger
logger = logging.getLogger("dsrag")
logger.setLevel(logging.DEBUG)
logger.propagate = True

# Clear existing handlers and add our handler
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(console_handler)
```

## Log Examples

### Document Ingestion (INFO)

```
2025-04-03 12:34:56 - dsrag.ingestion - INFO - Starting document ingestion - {"kb_id": "kb123", "doc_id": "doc456", "file_path": "example.pdf"}
2025-04-03 12:35:01 - dsrag.ingestion - INFO - Document ingestion successful - {"kb_id": "kb123", "doc_id": "doc456", "total_duration_s": 5.1234}
```

### Document Ingestion (DEBUG)

```
2025-04-03 12:34:56 - dsrag.ingestion - DEBUG - Parsing and Chunking complete - {"kb_id": "kb123", "doc_id": "doc456", "step": "parse_chunk", "duration_s": 2.1234, "num_sections": 5, "num_chunks": 25}
2025-04-03 12:34:58 - dsrag.ingestion - DEBUG - AutoContext complete - {"kb_id": "kb123", "doc_id": "doc456", "step": "auto_context", "duration_s": 1.5432, "model": "OpenAIChatAPI"}
2025-04-03 12:34:59 - dsrag.ingestion - DEBUG - Embedding complete - {"kb_id": "kb123", "doc_id": "doc456", "step": "embedding", "duration_s": 0.8765, "num_embeddings": 25, "model": "OpenAIEmbedding"}
```

### Query (INFO)

```
2025-04-03 12:40:12 - dsrag.query - INFO - Starting query - {"kb_id": "kb123", "query_id": "q789", "num_search_queries": 3}
2025-04-03 12:40:15 - dsrag.query - INFO - Query successful - {"kb_id": "kb123", "query_id": "q789", "total_duration_s": 3.2109, "num_final_segments": 4}
```

### Chat (INFO)

```
2025-04-03 12:45:23 - dsrag.chat - INFO - Starting chat response - {"thread_id": "t123", "message_id": "m456", "kb_ids": ["kb123", "kb789"], "stream": false, "user_input_length": 182}
2025-04-03 12:45:30 - dsrag.chat - INFO - Chat response completed - {"thread_id": "t123", "message_id": "m456", "kb_ids": ["kb123", "kb789"], "total_duration_s": 7.6543, "response_length": 512, "num_queries": 2, "num_segments": 5, "num_citations": 3}
```

## More Information

For more examples and details, see `examples/logging_example.py` in the repository.