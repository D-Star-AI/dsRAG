from typing import Optional, TypedDict

class ElementType(TypedDict):
    name: str
    instructions: str
    is_visual: bool

class Element(TypedDict):
    type: str
    content: str
    page_number: Optional[int]

class Line(TypedDict):
    content: str
    element_type: str
    page_number: Optional[int]
    is_visual: Optional[bool]

class Section(TypedDict):
    title: str
    start: int
    end: int
    content: str

class Chunk(TypedDict):
    line_start: int
    line_end: int
    content: str
    page_start: int
    page_end: int
    section_index: int
    is_visual: bool

class VLMConfig(TypedDict):
    provider: str
    model: str
    project_id: Optional[str]
    location: Optional[str]
    save_path: str
    exclude_elements: Optional[list[str]]
    element_types: Optional[list[ElementType]]

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