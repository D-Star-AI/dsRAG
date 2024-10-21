from typing import Optional, TypedDict

class Element(TypedDict):
    bounding_box: list[int]
    type: str
    content: Optional[str]
    description: Optional[str]
    page_number: Optional[int]
    image_path: Optional[str]

class Line(TypedDict):
    content: str
    element_type: str
    page_number: Optional[int]
    image_path: Optional[str]

class Sections(TypedDict):
    title: str
    start: int
    end: int
    content: str

class Chunks(TypedDict):
    line_start: int
    line_end: int
    content: str
    image_path: str
    page_start: int
    page_end: int
    section_index: int

class VLMConfig(TypedDict):
    provider: str
    model: str
    project_id: Optional[str]
    location: Optional[str]
    save_path: str
    exclude_elements: list[str]

class SemanticSectioningConfig(TypedDict):
    use_semantic_sectioning: bool
    llm_provider: str
    model: str
    language: str

class ChunkingConfig(TypedDict):
    chunk_size: int
    min_length_for_chunking: int

class FileParsingConfig(TypedDict):
    vlm_config: VLMConfig
    semantic_sectioning_config: SemanticSectioningConfig
    chunking_config: ChunkingConfig

class DocumentLines(TypedDict):
    element_type: str
    content: str
    page_number: Optional[int]
    image_path: Optional[str]
