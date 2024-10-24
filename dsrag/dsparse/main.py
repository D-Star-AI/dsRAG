import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .file_parsing.vlm_file_parsing import parse_file
from .file_parsing.non_vlm_file_parsing import parse_file_no_vlm
from .file_parsing.element_types import default_element_types
from .sectioning_and_chunking.semantic_sectioning import get_sections_from_elements, get_sections_from_str, get_sections_from_pages
from .sectioning_and_chunking.chunking import chunk_document
from .types import FileParsingConfig, VLMConfig, SemanticSectioningConfig, ChunkingConfig, Section, Chunk
from .file_parsing.file_system import FileSystem

from typing import List, Tuple
import json



def parse_and_chunk(kb_id: str, doc_id: str, file_parsing_config: FileParsingConfig, semantic_sectioning_config: SemanticSectioningConfig, chunking_config: ChunkingConfig, file_path: str = None, text: str = None) -> Tuple[List[Section], List[Chunk]]:
    # file parsing, semantic sectioning, and chunking
    use_vlm = file_parsing_config.get("use_vlm", False)
    if use_vlm:
        # make sure a file_path is provided
        if not file_path:
            raise ValueError("VLM parsing requires a file_path, not text. Please provide a file_path instead.")
        vlm_config = file_parsing_config.get("vlm_config", {})
        sections, chunks = parse_and_chunk_vlm(
            file_path=file_path,
            kb_id=kb_id,
            doc_id=doc_id,
            file_system=file_parsing_config["file_system"],
            vlm_config=vlm_config,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config,
        )
    else:
        if file_path:
            sections, chunks = parse_and_chunk_no_vlm(
                semantic_sectioning_config=semantic_sectioning_config,
                chunking_config=chunking_config,
                file_path=file_path,
            )
        else:
            sections, chunks = parse_and_chunk_no_vlm(
                semantic_sectioning_config=semantic_sectioning_config,
                chunking_config=chunking_config,
                text=text,
            )

    return sections, chunks


def parse_and_chunk_vlm(file_path: str, kb_id: str, doc_id: str, file_system: FileSystem, vlm_config: VLMConfig, semantic_sectioning_config: SemanticSectioningConfig, chunking_config: ChunkingConfig, testing_mode: bool = False) -> Tuple[List[Section], List[Chunk]]:
    """
    Inputs
    - file_path: the path to the file to parse and chunk
        - supported file types: .pdf
    - vlm_config: a dictionary containing the configuration for the VLM parser
        - provider: the VLM provider to use - only "vertex_ai" and "gemini" are supported at the moment
        - model: the VLM model to use
        - project_id: the GCP project ID (required if provider is "vertex_ai")
        - location: the GCP location (required if provider is "vertex_ai")
        - save_path: the path to save intermediate files created during VLM processing
        - exclude_elements: a list of element types to exclude from the parsed text. Default is ["Header", "Footer"].
        - element_types: a list of dictionaries, each containing 'name', 'instructions', and 'is_visual' keys
            - default (defined in element_types.py) will be used if not provided
    - semantic_sectioning_config: a dictionary containing the configuration for the semantic sectioning algorithm
    - chunking_config: a dictionary containing the configuration for the chunking algorithm

    Outputs
    - sections: a list of dictionaries, each containing the following keys:
        - title: str - the main topic of this section of the document (very descriptive)
        - content: str - the text of the section
        - start: int - line number where the section begins (inclusive)
        - end: int - line number where the section ends (inclusive)
    - chunks: a list of dictionaries, each containing the following keys:
        - line_start: int - line number where the chunk begins (inclusive)
        - line_end: int - line number where the chunk ends (inclusive)
        - content: str - the text of the chunk (or a description of the image if applicable)
        - image_path: str - the path to the image file (if applicable)
        - page_start: int - the page number the chunk starts on
        - page_end: int - the page number the chunk ends on (inclusive)
        - section_index: int - the index of the section this chunk belongs to
    """

    # Step 1: Parse the file

    #save_path = vlm_config["save_path"]
    elements = parse_file(pdf_path=file_path, kb_id=kb_id, doc_id=doc_id, vlm_config=vlm_config, file_system=file_system)
    
    if testing_mode:
        # dump to json for testing
        with open('elements.json', 'w') as f:
            json.dump(elements, f, indent=4)

    
    # Step 2: Get the sections from the elements
    
    if testing_mode:
        # load from json for testing
        with open('elements.json', 'r') as f:
            elements = json.load(f)

    # get the exclude_elements from the vlm_config
    exclude_elements = vlm_config.get('exclude_elements', ["Header", "Footer"])
    element_types = vlm_config.get("element_types", default_element_types)
    
    sections, document_lines = get_sections_from_elements(
        elements=elements,
        element_types=element_types,
        exclude_elements=exclude_elements,
        max_characters=20000,
        semantic_sectioning_config=semantic_sectioning_config
    )
    
    if testing_mode:
        # dump to json for testing
        with open('document_lines.json', 'w') as f:
            json.dump(document_lines, f, indent=4)
        with open('sections.json', 'w') as f:
            json.dump(sections, f, indent=4)

    
    # Step 3: Chunk the document

    if testing_mode:
        # load from json for testing
        with open('document_lines.json', 'r') as f:
            document_lines = json.load(f)
        with open('sections.json', 'r') as f:
            sections = json.load(f)

    chunk_size = chunking_config.get('chunk_size', 800)
    min_length_for_chunking = chunking_config.get('min_length_for_chunking', 1600)

    chunks = chunk_document(
        sections=sections, 
        document_lines=document_lines, 
        chunk_size=chunk_size, 
        min_length_for_chunking=min_length_for_chunking
    )
    
    if testing_mode:
        # dump to json for testing
        with open('chunks.json', 'w') as f:
            json.dump(chunks, f, indent=4)

    return sections, chunks

def parse_and_chunk_no_vlm(semantic_sectioning_config: SemanticSectioningConfig, chunking_config: ChunkingConfig, file_path: str = "", text: str = "", testing_mode: bool = False) -> tuple[List[Section], List[Chunk]]:
    """
    Inputs
    - semantic_sectioning_config: a dictionary containing the configuration for the semantic sectioning algorithm
    - chunking_config: a dictionary containing the configuration for the chunking algorithm
    - file_path: the path to the file to parse and chunk
        - supported file types: .txt, .pdf, .docx, .md
    - text: the text of the document to parse and chunk (include either text or file_path, not both)

    Outputs
    - sections: a list of dictionaries, each containing the following
    - chunks: a list of dictionaries, each containing the following keys:
        - line_start: int - line number where the chunk begins (inclusive)
        - line_end: int - line number where the chunk ends (inclusive)
        - content: str - the text of the chunk
        - page_start: int - the page number the chunk starts on
        - page_end: int - the page number the chunk ends on (inclusive)
        - section_index: int - the index of the section this chunk belongs to
    """
    if text == "" and file_path == "":
        raise ValueError("Either text or file_path must be provided")

    # Step 1: Parse the file

    if file_path:
        text, pdf_pages = parse_file_no_vlm(file_path)
    else:
        pdf_pages = None
        # text is already provided
    
    if testing_mode:
        # dump to txt file for testing
        with open('text.txt', 'w') as f:
            f.write(text)

    
    # Step 2: Get the sections from the elements
    
    if testing_mode:
        # load from json for testing
        with open('text.txt', 'r') as f:
            text = f.read()
    
    if pdf_pages:
        # If we have pdf pages then we want to use them so we can keep track of the page numbers
        sections, document_lines = get_sections_from_pages(
            pages=pdf_pages,
            max_characters=20000,
            semantic_sectioning_config=semantic_sectioning_config
        )
    else:
        sections, document_lines = get_sections_from_str(
            document=text,
            max_characters=20000,
            semantic_sectioning_config=semantic_sectioning_config
        )
    
    if testing_mode:
        # dump to json for testing
        with open('document_lines.json', 'w') as f:
            json.dump(document_lines, f, indent=4)
        with open('sections.json', 'w') as f:
            json.dump(sections, f, indent=4)

    
    # Step 3: Chunk the document

    if testing_mode:
        # load from json for testing
        with open('document_lines.json', 'r') as f:
            document_lines = json.load(f)
        with open('sections.json', 'r') as f:
            sections = json.load(f)

    chunk_size = chunking_config.get('chunk_size', 800)
    min_length_for_chunking = chunking_config.get('min_length_for_chunking', 1600)

    chunks = chunk_document(
        sections=sections, 
        document_lines=document_lines, 
        chunk_size=chunk_size, 
        min_length_for_chunking=min_length_for_chunking
    )
    
    if testing_mode:
        # dump to json for testing
        with open('chunks.json', 'w') as f:
            json.dump(chunks, f, indent=4)

    return sections, chunks