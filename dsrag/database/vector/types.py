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
    value: Union[str, int, float, list[str], list[int], list[float]]