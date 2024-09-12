from typing import Optional, Sequence, Union
from typing_extensions import TypedDict


class ChunkMetadata(TypedDict):
    doc_id: str
    chunk_text: str
    chunk_index: int
    chunk_header: str


Vector = Union[Sequence[float], Sequence[int]]


class VectorSearchResult(TypedDict):
    doc_id: Optional[str]
    vector: Optional[Vector]
    metadata: ChunkMetadata
    similarity: float
      
class MetadataFilter(TypedDict):
    field: str
    operator: str # Can be one of the following: 'equals', 'not_equals', 'in', 'not_in', 'greater_than', 'less_than', 'greater_than_equals', 'less_than_equals'
    value: Union[str, int, float]


# declarative_base, Column, String, Integer, func, deferred need to be imported from SQLAlchemy
Base = declarative_base()

class ChunkEmbedding(Base):
    __tablename__ = "chunk_embedding"

    id = Column(UUID, primary_key=True, server_default=func.uuid_generate_v1mc())
    kb_id = Column(String, nullable=False)
    transcript_id = Column(String, nullable=False)
    # The vector column we use is from pgvector. Not sure how well this transfers to non-Postgres databases.
    # https://github.com/pgvector/pgvector-python/blob/master/pgvector/sqlalchemy/vector.py
    vector = deferred(Column(Vector(768), nullable=False))
    chunk_metadata: ChunkMetadata = Column(MUTABLE_JSONB, nullable=False)  # type: ignore

class TranscriptChunk(Base):
    __tablename__ = "transcript_chunk"
    doc_id = Column(String, primary_key=True)
    document_title = Column(String)
    document_summary = Column(String)
    section_title = Column(String)
    section_summary = Column(String)
    chunk_text = Column(String)
    chunk_index = Column(Integer, primary_key=True)