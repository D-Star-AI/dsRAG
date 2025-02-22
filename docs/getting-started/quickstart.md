# Quick Start Guide

This guide will help you get started with dsRAG quickly. We'll cover the basics of creating a knowledge base and querying it.

## Setting Up

First, make sure you have the necessary API keys set as environment variables:
- `OPENAI_API_KEY` for embeddings and AutoContext
- `CO_API_KEY` for reranking with Cohere

## Creating a Knowledge Base

You can create a new knowledge base directly from a file using the `create_kb_from_file` helper function:

```python
from dsrag.create_kb import create_kb_from_file

file_path = "path/to/your/document.pdf"
kb_id = "my_knowledge_base"
kb = create_kb_from_file(kb_id, file_path)
```

The KnowledgeBase object persists to disk automatically, so you don't need to explicitly save it.

## Querying the Knowledge Base

Once you have created a knowledge base, you can load it by its `kb_id` and query it:

```python
from dsrag.knowledge_base import KnowledgeBase

# Load the knowledge base
kb = KnowledgeBase("my_knowledge_base")

# You can query with multiple search queries
search_queries = [
    "What are the main topics covered?",
    "What are the key findings?"
]

# Get results
results = kb.query(search_queries)
for segment in results:
    print(segment)
```

## Using OpenAI-Only Configuration

If you prefer to use only OpenAI services (without Cohere), you can customize the configuration:

```python
from dsrag.llm import OpenAIChatAPI
from dsrag.reranker import NoReranker

# Configure components
llm = OpenAIChatAPI(model='gpt-4o-mini')
reranker = NoReranker()

# Create knowledge base with custom configuration
kb = KnowledgeBase(
    kb_id="my_knowledge_base",
    reranker=reranker,
    auto_context_model=llm
)

# Add documents
file_path = "path/to/your/document.pdf"
kb.add_document(doc_id=file_path, file_path=file_path)
```

## Next Steps

- Learn more about basic usage patterns in the [Basic Usage Guide](basic-usage.md)
- Explore the [Concepts](../concepts/overview.md) section to understand dsRAG's architecture
- Check out the [Guides](../guides/customization.md) section for advanced usage 