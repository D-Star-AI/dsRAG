import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dsrag.dsparse.vlm_file_parsing import parse_file
from dsrag.dsparse.non_vlm_file_parsing import parse_file_no_vlm
from dsrag.dsparse.semantic_sectioning import get_sections_from_elements, get_sections_from_str, get_sections_from_pages
from dsrag.dsparse.chunking import chunk_document

from typing import List, Dict, Tuple
import json

def parse_and_chunk_vlm(file_path: str, vlm_config: dict, semantic_sectioning_config: dict, chunking_config: dict, testing_mode: bool = False) -> Tuple[List[Dict], List[Dict]]:
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

    save_path = vlm_config["save_path"]
    elements = parse_file(pdf_path=file_path, save_path=save_path, vlm_config=vlm_config)
    
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
    
    sections, document_lines = get_sections_from_elements(
        elements=elements,
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

def parse_and_chunk_no_vlm(semantic_sectioning_config: dict, chunking_config: dict, file_path: str = "", text: str = "", testing_mode: bool = False) -> List[Dict]:
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


# Test the function
if __name__ == "__main__":
    user_id = "zmcc"

    pdf_path = "/Users/nickmccormick/Documents/D-Star-AI/dsRAG/tests/data/mck_energy_first_5_pages.pdf" # '/Users/zach/Code/dsRAG/tests/data/levels_of_agi.pdf'
    file_id = "levels_of_agi"
    
    #pdf_path = "/Users/zach/Code/mck_energy.pdf"
    #file_id = "mck_energy"

    save_path = f"{user_id}/{file_id}" # base directory to save the page images, pages with bounding boxes, and extracted images

    vlm_config = {
        "provider": "gemini",
        "model": "gemini-1.5-pro-002",
        #"project_id": os.environ["VERTEX_PROJECT_ID"],
        "location": "us-central1",
        "save_path": save_path,
        "exclude_elements": ["Header", "Footer"],
    }

    semantic_sectioning_config = {
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "language": "en",
    }
    
    
    """sections, chunks = parse_and_chunk_vlm(
        file_path=pdf_path,
        vlm_config=vlm_config,
        semantic_sectioning_config=semantic_sectioning_config,
        chunking_config={}
    )"""
        
    """chunks = parse_and_chunk_no_vlm(
        file_path=pdf_path,
        semantic_sectioning_config=semantic_sectioning_config
        )"""