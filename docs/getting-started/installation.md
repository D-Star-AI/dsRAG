# Installation

## Prerequisites

Before installing dsRAG, you'll need to set up API keys for the following services (using default configuration):

- `OPENAI_API_KEY`: Required for embeddings and AutoContext
- `CO_API_KEY`: Required for reranking

## Installing dsRAG

Install the package using pip:

```bash
pip install dsrag
```

## Alternative Configurations

If you prefer not to use Cohere for reranking, you can configure dsRAG to use only OpenAI services. See the [Basic Usage](basic-usage.md) guide for details on customizing your configuration.

Available Model Providers:

### Vector Databases
- BasicVectorDB
- WeaviateVectorDB
- ChromaDB
- QdrantVectorDB
- MilvusDB
- PineconeDB

### Chunk Databases
- BasicChunkDB
- SQLiteDB

### Embedding Models
- OpenAIEmbedding
- CohereEmbedding
- VoyageAIEmbedding
- OllamaEmbedding

### Rerankers
- CohereReranker
- VoyageReranker
- NoReranker

### LLM Providers
- OpenAIChatAPI
- AnthropicChatAPI
- OllamaChatAPI

### File Systems
- LocalFileSystem
- S3FileSystem 