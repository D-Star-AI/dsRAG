# Quick Start Guide

This guide will help you get started with dsRAG quickly. We'll cover the basics of creating a knowledge base and querying it.

## Setting Up

First, make sure you have the necessary API keys set as environment variables:
- `OPENAI_API_KEY` for embeddings and AutoContext
- `CO_API_KEY` for reranking with Cohere

If you don't have both of these available, you can use the OpenAI-only configuration or local-only configuration below.

## Creating a Knowledge Base

Create a new knowledge base and add documents to it:

```python
from dsrag.knowledge_base import KnowledgeBase

# Create a knowledge base
kb = KnowledgeBase(kb_id="my_knowledge_base")

# Add documents
kb.add_document(
    doc_id="user_manual",  # Use a meaningful ID if possible
    file_path="path/to/your/document.pdf",
    document_title="User Manual",  # Optional but recommended
    metadata={"type": "manual"}    # Optional metadata
)
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
kb.add_document(
    doc_id="user_manual",  # Use a meaningful ID
    file_path="path/to/your/document.pdf",
    document_title="User Manual",  # Optional but recommended
    metadata={"type": "manual"}    # Optional metadata
)
```

## Semantic Sectioning Configuration

dsRAG supports semantic sectioning with multiple LLM providers:
- OpenAI (e.g., `gpt-4.1-mini`)
- Anthropic (e.g., `claude-3-5-haiku-latest`)
- Gemini (e.g., `gemini-2.0-flash`)

You can configure semantic sectioning like this:

```python
semantic_sectioning_config = {
    "llm_provider": "anthropic",  # or "openai" or "gemini"
    "model": "claude-3-5-haiku-latest",
    "use_semantic_sectioning": True
}

kb.add_document(
    doc_id="user_manual",
    file_path="path/to/your/document.pdf",
    semantic_sectioning_config=semantic_sectioning_config
)
```

## Local-only configuration

If you don't want to use any third-party services, you can configure dsRAG to run fully locally using Ollama. There will be a couple limitations:
- You will not be able to use VLM file parsing
- You will not be able to use semantic sectioning
- You will not be able to use a reranker, unless you implement your own custom Reranker class

```python
from dsrag.llm import OllamaChatAPI
from dsrag.reranker import NoReranker
from dsrag.embedding import OllamaEmbedding

llm = OllamaChatAPI(model="llama3.1:8b")
reranker = NoReranker()
embedding = OllamaEmbedding(model="nomic-embed-text")

kb = KnowledgeBase(
    kb_id="my_knowledge_base",
    reranker=reranker,
    auto_context_model=llm,
    embedding_model=embedding
)

# Disable semantic sectioning
semantic_sectioning_config = {
    "use_semantic_sectioning": False,
}

# Add documents
kb.add_document(
    doc_id="user_manual",
    file_path="path/to/your/document.pdf",
    document_title="User Manual",
    semantic_sectioning_config=semantic_sectioning_config
)
```