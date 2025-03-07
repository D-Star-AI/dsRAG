from .db import VectorDB
from .types import VectorSearchResult, Vector, ChunkMetadata

# Always import the basic DB as it has no dependencies
from .basic_db import BasicVectorDB

# Define what's in __all__ for "from dsrag.database.vector import *"
__all__ = [
    "VectorDB", 
    "BasicVectorDB", 
    "VectorSearchResult", 
    "Vector", 
    "ChunkMetadata"
]

# Lazy load database modules to avoid importing all dependencies at once
def __getattr__(name):
    """Lazily import database implementations only when accessed."""
    if name == "ChromaDB":
        from .chroma_db import ChromaDB
        return ChromaDB
    elif name == "WeaviateVectorDB":
        from .weaviate_db import WeaviateVectorDB
        return WeaviateVectorDB
    elif name == "QdrantVectorDB":
        from .qdrant_db import QdrantVectorDB
        return QdrantVectorDB
    elif name == "MilvusDB":
        from .milvus_db import MilvusDB
        return MilvusDB
    elif name == "PostgresVectorDB":
        from .postgres_db import PostgresVectorDB
        return PostgresVectorDB
    elif name == "PineconeDB":
        from .pinecone_db import PineconeDB
        return PineconeDB
    else:
        raise AttributeError(f"module 'dsrag.database.vector' has no attribute '{name}'")
