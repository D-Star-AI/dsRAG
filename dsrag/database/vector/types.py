from typing import Sequence
from typing_extensions import TypedDict


class ChunkMetadata(TypedDict):
    doc_id: str
    chunk_text: str
    chunk_index: int
    chunk_header: str
    document_title: str
    document_summary: str
    section_title: str
    section_summary: str


Vector = Sequence[float] | Sequence[int]


class VectorSearchResult(TypedDict):
    doc_id: str | None
    vector: Vector | None
    metadata: ChunkMetadata
    similarity: float
