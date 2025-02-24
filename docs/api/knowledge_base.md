# KnowledgeBase

The KnowledgeBase class is the main interface for working with dsRAG. It handles document processing, storage, and retrieval.

### Public Methods

The following methods are part of the public API:

- `__init__`: Initialize a new KnowledgeBase instance
- `add_document`: Add a single document to the knowledge base
- `add_documents`: Add multiple documents in parallel
- `delete`: Delete the entire knowledge base and all associated data
- `delete_document`: Delete a specific document from the knowledge base
- `query`: Search the knowledge base with one or more queries

::: dsrag.knowledge_base.KnowledgeBase
    options:
      show_root_heading: false
      show_root_full_path: false
      members:
        - __init__
        - add_document
        - add_documents
        - delete
        - delete_document
        - query

## KB Components

### Vector Databases

::: dsrag.database.vector.VectorDB
    options:
      show_root_heading: true
      show_root_full_path: false

### Chunk Databases

::: dsrag.database.chunk.ChunkDB
    options:
      show_root_heading: true
      show_root_full_path: false

### Embedding Models

::: dsrag.embedding.Embedding
    options:
      show_root_heading: true
      show_root_full_path: false

### Rerankers

::: dsrag.reranker.Reranker
    options:
      show_root_heading: true
      show_root_full_path: false

### LLM Providers

::: dsrag.llm.LLM
    options:
      show_root_heading: true
      show_root_full_path: false

### File Systems

::: dsrag.dsparse.file_parsing.file_system.FileSystem
    options:
      show_root_heading: true
      show_root_full_path: false