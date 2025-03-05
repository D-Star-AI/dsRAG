import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .file_parsing.vlm_file_parsing import parse_file, pdf_to_images
from .file_parsing.non_vlm_file_parsing import parse_file_no_vlm
from .file_parsing.element_types import default_element_types
from .sectioning_and_chunking.semantic_sectioning import get_sections_from_elements, get_sections_from_str, get_sections_from_pages
from .sectioning_and_chunking.chunking import chunk_document
from .models.types import FileParsingConfig, VLMConfig, SemanticSectioningConfig, ChunkingConfig, Section, Chunk
from .file_parsing.file_system import FileSystem, LocalFileSystem

from typing import List, Tuple
import json

def parse_and_chunk(
    kb_id: str, 
    doc_id: str, 
    file_parsing_config: FileParsingConfig = {}, 
    semantic_sectioning_config: SemanticSectioningConfig = {}, 
    chunking_config: ChunkingConfig = {}, 
    file_system: FileSystem = {}, 
    file_path: str = None, 
    text: str = None,
) -> Tuple[List[Section], List[Chunk]]:
    """
    Inputs
    - kb_id: a knowledge base / collection identifier for the document - used to determine location to store page images
    - doc_id: a document identifier for the document - used to determine location to store page images
    - file_parsing_config: a dictionary with configuration parameters for parsing the file
        - use_vlm: bool - whether to use VLM (vision language model) for parsing the file (default is False)
        - vlm_config: a dictionary containing the configuration for the VLM parser
            - provider: the VLM provider to use - only "vertex_ai" and "gemini" are supported at the moment
            - model: the VLM model to use
            - project_id: the GCP project ID (required if provider is "vertex_ai")
            - location: the GCP location (required if provider is "vertex_ai")
            - exclude_elements: a list of element types to exclude from the parsed text. Default is ["Header", "Footer"].
            - element_types: a list of dictionaries, each containing 'name', 'instructions', and 'is_visual' keys
                - default (defined in element_types.py) will be used if not provided
            - images_already_exist: bool, whether the images have already been extracted and saved (default is False)
        - always_save_page_images: bool - whether to save page images even if VLM is not used (default is False)
    - semantic_sectioning_config: a dictionary with configuration for the semantic sectioning model (defaults will be used if not provided)
        - use_semantic_sectioning: if False, semantic sectioning will be skipped (default is True)
        - llm_provider: the LLM provider to use for semantic sectioning - only "openai" and "anthropic" are supported at the moment
        - model: the LLM model to use for semantic sectioning
        - language: the language of the document - used for prompting the LLM model to generate section titles in the correct language
    - chunking_config: a dictionary with configuration for chunking the document/sections into smaller pieces (defaults will be used if not provided)
        - chunk_size: the maximum number of characters to include in each chunk
        - min_length_for_chunking: the minimum length of text to allow chunking (measured in number of characters); if the text is shorter than this, it will be added as a single chunk. If semantic sectioning is used, this parameter will be applied to each section. Setting this to a higher value than the chunk_size can help avoid unnecessary chunking of short documents or sections.
    - file_system: a FileSystem object for defining where to store page images (used if either use_vlm or always_save_page_images is True; defaults to local storage)
    - file_path: the path to the file to parse and chunk
        - supported file types: .pdf, .docx, .txt, .md
    - text: the text to parse and chunk
        - only one of file_path or text should be provided

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
        - page_start: int - the page number the chunk starts on
        - page_end: int - the page number the chunk ends on (inclusive)
        - section_index: int - the index of the section this chunk belongs to
        - is_visual: bool - whether the chunk is visual (e.g., an image)
    """
    
    # file parsing, semantic sectioning, and chunking
    use_vlm = file_parsing_config.get("use_vlm", False)

    # We can only run VLM file parsing on .pdf files
    if use_vlm and file_path and not file_path.lower().endswith(".pdf"):
        raise ValueError("VLM parsing requires a .pdf file. Please provide a .pdf file_path.")
    
    # Create a FileSystem object if not provided
    if not file_system:
        file_system = LocalFileSystem(base_path=os.path.expanduser("~/dsParse"))
    
    if use_vlm:
        # make sure a file_path is provided
        if not file_path:
            raise ValueError("VLM parsing requires a file_path, not text. Please provide a file_path instead.")
        vlm_config = file_parsing_config.get("vlm_config", {})
        sections, chunks = parse_and_chunk_vlm(
            file_path=file_path,
            kb_id=kb_id,
            doc_id=doc_id,
            file_system=file_system,
            vlm_config=vlm_config,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config,
        )
    else:
        if file_path:
            sections, chunks = parse_and_chunk_no_vlm(
                semantic_sectioning_config=semantic_sectioning_config,
                chunking_config=chunking_config,
                kb_id=kb_id,
                doc_id=doc_id,
                file_path=file_path,
                file_system=file_system,
                always_save_page_images=file_parsing_config.get("always_save_page_images", False),
            )
        else:
            sections, chunks = parse_and_chunk_no_vlm(
                semantic_sectioning_config=semantic_sectioning_config,
                chunking_config=chunking_config,
                kb_id=kb_id,
                doc_id=doc_id,
                text=text,
            )

    return sections, chunks


def parse_and_chunk_vlm(
    file_path: str, kb_id: str, doc_id: str, file_system: FileSystem, vlm_config: VLMConfig,
    semantic_sectioning_config: SemanticSectioningConfig, chunking_config: ChunkingConfig,
    testing_mode: bool = False) -> Tuple[List[Section], List[Chunk]]:
    # Step 1: Parse the file

    #save_path = vlm_config["save_path"]
    elements = parse_file(
        pdf_path=file_path, 
        kb_id=kb_id, 
        doc_id=doc_id, 
        vlm_config=vlm_config, 
        file_system=file_system,
    )
    
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

def parse_and_chunk_no_vlm(semantic_sectioning_config: SemanticSectioningConfig, chunking_config: ChunkingConfig, kb_id: str, doc_id: str, file_path: str = "", text: str = "", file_system: FileSystem = None, always_save_page_images: bool = False, testing_mode: bool = False) -> tuple[List[Section], List[Chunk]]:
    if text == "" and file_path == "":
        raise ValueError("Either text or file_path must be provided")
    
    if always_save_page_images and file_system is None:
        raise ValueError("If always_save_page_images is True, a file_system must be provided")

    # Step 1: Parse the file

    if file_path:
        text, pdf_pages = parse_file_no_vlm(file_path)
    else:
        pdf_pages = None
        # text is already provided

    if pdf_pages and always_save_page_images:
        # If the PDF pages exist and the config says to save them, convert each page to an image and save it
        pdf_to_images(file_path, kb_id, doc_id, file_system, dpi=150)
    
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