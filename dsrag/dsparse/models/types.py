from typing import Optional, TypedDict
from ..file_parsing.file_system import FileSystem

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
    provider: Optional[str]
    model: Optional[str]
    project_id: Optional[str]
    location: Optional[str]
    exclude_elements: Optional[list[str]]
    element_types: Optional[list[ElementType]]

class SemanticSectioningConfig(TypedDict):
    use_semantic_sectioning: Optional[bool]
    llm_provider: Optional[str]
    model: Optional[str]
    language: Optional[str]

class ChunkingConfig(TypedDict):
    chunk_size: Optional[int]
    min_length_for_chunking: Optional[int]

class FileParsingConfig(TypedDict):
    use_vlm: Optional[bool]
    vlm_config: Optional[VLMConfig]
    always_save_page_images: Optional[bool]