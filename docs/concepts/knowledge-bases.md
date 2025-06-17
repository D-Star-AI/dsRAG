# Knowledge Bases

A knowledge base in dsRAG is a searchable collection of documents that can be queried to find relevant information. The `KnowledgeBase` class handles document processing, storage, and retrieval.

## Creating a Knowledge Base

To create a knowledge base:

```python
from dsrag.knowledge_base import KnowledgeBase

# Create a basic knowledge base
kb = KnowledgeBase(
    kb_id="my_kb",
    title="Product Documentation",
    description="Technical documentation for XYZ product"
)

# Or with custom configuration
kb = KnowledgeBase(
    kb_id="my_kb",
    storage_directory="path/to/storage",  # Where to store KB data
    embedding_model=custom_embedder,      # Custom embedding model
    reranker=custom_reranker,            # Custom reranking model
    vector_db=custom_vector_db,          # Custom vector database
    chunk_db=custom_chunk_db            # Custom chunk database
)
```

## Adding Documents

Documents can be added from text or files:

```python
# Add from text
kb.add_document(
    doc_id="intro-guide",
    text="This is the introduction guide...",
    document_title="Introduction Guide",
    metadata={"type": "guide", "version": "1.0"}
)

# Add from file
kb.add_document(
    doc_id="user-manual",
    file_path="path/to/manual.pdf",
    metadata={"type": "manual", "department": "engineering"}
)

# Add with advanced configuration
kb.add_document(
    doc_id="technical-spec",
    file_path="path/to/spec.pdf",
    file_parsing_config={
        "use_vlm": True,                # Use vision language model for PDFs
        "always_save_page_images": True  # Save page images for visual content
    },
    chunking_config={
        "chunk_size": 800,              # Characters per chunk
        "min_length_for_chunking": 2000 # Minimum length to chunk
    },
    auto_context_config={
        "use_generated_title": True,    # Generate title if not provided
        "get_document_summary": True,   # Generate document summary
        "llm_max_concurrent_requests": 5  # Maximum concurrent requests
    }
)
```

## Querying the Knowledge Base

Search the knowledge base for relevant information:

```python
# Simple query
results = kb.query(
    search_queries=["How to configure the system?"]
)

# Advanced query with filtering and parameters
results = kb.query(
    search_queries=[
        "System configuration steps",
        "Configuration prerequisites"
    ],
    metadata_filter={
        "field": "doc_id",
        "operator": "equals",
        "value": "user_manual"
    },
    rse_params="precise",  # Use preset RSE parameters
    return_mode="text"     # Return text content
)

# Process results
for segment in results:
    print(f"""
Document: {segment['doc_id']}
Pages: {segment['segment_page_start']} - {segment['segment_page_end']}
Content: {segment['content']}
Relevance: {segment['score']}
""")
```

## RSE Parameters

The Relevant Segment Extraction (RSE) system can be tuned using different parameter presets:
- `"balanced"`: Default preset balancing precision and comprehensiveness
- `"precise"`: Favors shorter, more focused segments
- `"comprehensive"`: Returns longer segments with more context

Or configure custom RSE parameters:
```python
results = kb.query(
    search_queries=["system requirements"],
    rse_params={
        "max_length": 5,                # Max segments length (in number of chunks)
        "overall_max_length": 20,       # Total length limit across all segments (in number of chunks)
        "minimum_value": 0.5,           # Minimum relevance score
        "irrelevant_chunk_penalty": 0.2 # Penalty for irrelevant chunks in a segment - higher penalty leads to shorter segments
    }
)
```

## Metadata Query Filters

Certain vector DBs support metadata filtering when running a query (currently only ChromaDB). This allows you to have more control over what document(s) get searched. A common use case would be asking questions about a single document in a knowledge base, in which case you would supply the `doc_id` as a metadata filter.

The metadata filter should be a dictionary with the following structure:

```python
metadata_filter = {
    "field": "doc_id",      # The metadata field to filter on
    "operator": "equals",   # The comparison operator
    "value": "doc123"      # The value to compare against
}
```

Supported operators:
- `equals`
- `not_equals` 
- `in`
- `not_in`
- `greater_than`
- `less_than`
- `greater_than_equals`
- `less_than_equals`

For operators that take multiple values (`in` and `not_in`), the value should be a list where all items are of the same type (string, integer, or float).

Example usage:
```python
# Query a specific document
results = kb.query(
    search_queries=["system requirements"],
    metadata_filter={
        "field": "doc_id",
        "operator": "equals",
        "value": "technical_spec_v1"
    }
)

# Query documents from multiple departments
results = kb.query(
    search_queries=["security protocols"],
    metadata_filter={
        "field": "department",
        "operator": "in", 
        "value": ["security", "compliance"]
    }
)
```