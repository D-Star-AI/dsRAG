# Components

There are six key components that define the configuration of a KnowledgeBase. Each component is customizable, with several built-in options available.

## VectorDB

The VectorDB component stores embedding vectors and associated metadata.

Available options:

- `BasicVectorDB`
- `WeaviateVectorDB`
- `ChromaDB`
- `QdrantVectorDB`
- `MilvusDB`
- `PineconeDB`

## ChunkDB

The ChunkDB stores the content of text chunks in a nested dictionary format, keyed on `doc_id` and `chunk_index`. This is used by RSE to retrieve the full text associated with specific chunks.

Available options:

- `BasicChunkDB`
- `SQLiteDB`

## Embedding

The Embedding component defines the embedding model used for vectorizing text.

Available options:

- `OpenAIEmbedding`
- `CohereEmbedding`
- `VoyageAIEmbedding`
- `OllamaEmbedding`

## Reranker

The Reranker component provides more accurate ranking of chunks after vector database search and before RSE. This is optional but highly recommended.

Available options:

- `CohereReranker`
- `VoyageReranker`
- `NoReranker`

## LLM

The LLM component is used in AutoContext for:

- Document title generation
- Document summarization
- Section summarization

Available options:

- `OpenAIChatAPI`
- `AnthropicChatAPI`
- `OllamaChatAPI`

## FileSystem

The FileSystem component defines where to save PDF images and extracted elementsfor VLM file parsing.

Available options:

- `LocalFileSystem`
- `S3FileSystem`

#### LocalFileSystem Configuration
Only requires a `base_path` parameter to define where files will be stored on the system.

#### S3FileSystem Configuration
Requires the following parameters:

- `base_path`: Used when downloading files from S3
- `bucket_name`: S3 bucket name
- `region_name`: AWS region
- `access_key`: AWS access key
- `access_secret`: AWS secret key

Note: Files must be stored locally temporarily for use in the retrieval system, even when using S3. 