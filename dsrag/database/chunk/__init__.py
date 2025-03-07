from .db import ChunkDB
from .types import FormattedDocument

# Always import the basic DB as it has no dependencies
from .basic_db import BasicChunkDB

# Define what's in __all__ for "from dsrag.database.chunk import *"
__all__ = [
    "ChunkDB", 
    "BasicChunkDB", 
    "FormattedDocument"
]

# Lazy load database modules to avoid importing all dependencies at once
def __getattr__(name):
    """Lazily import database implementations only when accessed."""
    if name == "SQLiteDB":
        from .sqlite_db import SQLiteDB
        return SQLiteDB
    elif name == "PostgresChunkDB":
        from .postgres_db import PostgresChunkDB
        return PostgresChunkDB
    elif name == "DynamoDB":
        from .dynamo_db import DynamoDB
        return DynamoDB
    else:
        raise AttributeError(f"module 'dsrag.database.chunk' has no attribute '{name}'")
