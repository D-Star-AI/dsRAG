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
