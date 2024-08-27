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
