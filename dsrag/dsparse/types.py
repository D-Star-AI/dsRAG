from typing import Optional, TypedDict

class Element(TypedDict):
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

class Section(TypedDict):
    title: str
    line_start: int
    line_end: int
    content: str

class Chunk(TypedDict):
    line_start: int
    line_end: int
    content: str
    description: Optional[str]
    image_path: Optional[str]
    page_start: int
    page_end: int