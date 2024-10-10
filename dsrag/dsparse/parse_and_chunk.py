from vlm_file_parsing import parse_file
from non_vlm_file_parsing import parse_file_no_vlm
from semantic_sectioning import get_sections_from_elements, get_sections_from_str
from chunking import chunk_document
from typing import List, Dict, Tuple
import json
import os

def parse_and_chunk_vlm(file_path: str, save_path: str, vlm_config: dict, semantic_sectioning_config: dict, exclude_elements: List[str] = ["Header", "Footer"]) -> Tuple[List[Dict], List[Dict]]:
    """
    Inputs
    - file_path: the path to the file to parse and chunk
        - supported file types: .pdf
    - save_path: the path to save the parsed elements to
    - vlm_config: a dictionary containing the configuration for the VLM parser
    - semantic_sectioning_config: a dictionary containing the configuration for the semantic sectioning algorithm

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

    testing_mode = True

    # Step 1: Parse the file

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

    chunks = chunk_document(
        sections=sections, 
        document_lines=document_lines, 
        chunk_size=1000, 
        min_length_for_chunking=2000
        )
    
    if testing_mode:
        # dump to json for testing
        with open('chunks.json', 'w') as f:
            json.dump(chunks, f, indent=4)

    return sections, chunks

def parse_and_chunk_no_vlm(file_path: str, semantic_sectioning_config: dict) -> List[Dict]:
    """
    Inputs
    - file_path: the path to the file to parse and chunk
        - supported file types: .txt, .pdf, .docx, .md
    - semantic_sectioning_config: a dictionary containing the configuration for the semantic sectioning algorithm

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

    testing_mode = True

    # Step 1: Parse the file

    text, pdf_pages = parse_file_no_vlm(file_path)
    
    if testing_mode:
        # dump to json for testing
        with open('elements.json', 'w') as f:
            json.dump(elements, f, indent=4)

    
    # Step 2: Get the sections from the elements
    
    if testing_mode:
        # load from json for testing
        with open('elements.json', 'r') as f:
            elements = json.load(f)
    
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

    chunks = chunk_document(
        sections=sections, 
        document_lines=document_lines, 
        chunk_size=1000, 
        min_length_for_chunking=2000
        )
    
    if testing_mode:
        # dump to json for testing
        with open('chunks.json', 'w') as f:
            json.dump(chunks, f, indent=4)

    return sections, chunks


# Test the function
if __name__ == "__main__":
    user_id = "zmcc"

    #pdf_path = '/Users/zach/Code/dsRAG/tests/data/levels_of_agi.pdf'
    #file_id = "levels_of_agi"
    
    pdf_path = "/Users/zach/Code/mck_energy.pdf"
    file_id = "mck_energy"

    save_path = f"{user_id}/{file_id}" # base directory to save the page images, pages with bounding boxes, and extracted images

    vlm_config = {
        "provider": "vertex_ai",
        "model": "gemini-1.5-pro-002",
        "project_id": os.environ["VERTEX_PROJECT_ID"],
        "location": "us-central1"
    }

    semantic_sectioning_config = {
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "language": "en",
    }
    
    chunks = parse_and_chunk_vlm(
        file_path=pdf_path,
        save_path=save_path,
        vlm_config=vlm_config,
        semantic_sectioning_config=semantic_sectioning_config
        )