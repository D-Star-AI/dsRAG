from typing import Optional, Sequence
from typing_extensions import TypedDict


class ChunkMetadata(TypedDict):
    doc_id: str
    chunk_text: str
    chunk_index: int
    chunk_header: str
    document_title: Optional[str]
    document_summary: Optional[str]
    section_title: Optional[str]
    section_summary: Optional[str]


Vector = Sequence[float] | Sequence[int]


class VectorSearchResult(TypedDict):
    doc_id: Optional[str]
    vector: Vector | None
    metadata: ChunkMetadata
    similarity: float


# declarative_base, Column, String, Integer, func, deferred, Vector need to be imported from SQLAlchemy
Base = declarative_base()


class ChunkMetadata(TypedDict):
    doc_id: str
    chunk_text: str
    chunk_index: int
    chunk_header: str
    document_title: str
    document_summary: str
    section_title: str
    section_summary: str


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embedding"

    id = Column(UUID, primary_key=True, server_default=func.uuid_generate_v1mc())
    kb_id = Column(String, nullable=False)
    transcript_id = Column(String, nullable=False)
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
