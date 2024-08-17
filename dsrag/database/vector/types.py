from typing import Sequence
from typing_extensions import TypedDict


class ChunkMetadata(TypedDict):
    doc_id: str
    chunk_text: str
    chunk_index: int
    chunk_header: str
    document_title: str | None
    document_summary: str | None
    section_title: str | None
    section_summary: str | None


Vector = Sequence[float] | Sequence[int]


class VectorSearchResult(TypedDict):
    doc_id: str | None
    vector: Vector | None
    metadata: ChunkMetadata
    similarity: float
