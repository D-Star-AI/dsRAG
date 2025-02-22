# Architecture Overview

dsRAG is built around three key methods that improve performance over vanilla RAG systems:

1. Semantic sectioning
2. AutoContext
3. Relevant Segment Extraction (RSE)

## Key Methods

### Semantic Sectioning

Semantic sectioning uses an LLM to break a document into cohesive sections. The process works as follows:

1. The document is annotated with line numbers
2. An LLM identifies the starting and ending lines for each "semantically cohesive section"
3. Sections typically range from a few paragraphs to a few pages long
4. Sections are broken into smaller chunks if needed
5. The LLM generates descriptive titles for each section
6. Section titles are used in contextual chunk headers created by AutoContext

This process provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

### AutoContext (Contextual Chunk Headers)

AutoContext creates contextual chunk headers that contain:
- Document-level context
- Section-level context

These headers are prepended to chunks before embedding. Benefits include:
- More accurate and complete representation of text content and meaning
- Dramatic improvement in retrieval quality
- Reduced rate of irrelevant results
- Reduced LLM misinterpretation in downstream applications

### Relevant Segment Extraction (RSE)

RSE is a query-time post-processing step that:
1. Takes clusters of relevant chunks
2. Intelligently combines them into longer sections (segments)
3. Provides better context to the LLM than individual chunks

RSE is particularly effective for:
- Complex questions where answers span multiple chunks
- Identifying appropriate context length based on query type
- Maintaining coherent context while avoiding irrelevant information

## Document Processing Flow

1. Documents → VLM file parsing
2. Semantic sectioning
3. Chunking
4. AutoContext
5. Embedding
6. Chunk and vector database upsert

## Query Processing Flow

1. Queries
2. Vector database search
3. Reranking
4. RSE
5. Results

For more detailed information about specific components and configuration options, please refer to:
- [Components Documentation](components.md)
- [Configuration Options](config.md)
- [Knowledge Base Details](knowledge-base.md) 