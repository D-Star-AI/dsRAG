from typing import List, Dict, Any

def chunk_document(sections: List[Dict], document_lines: List[Dict], chunk_size: int, min_length_for_chunking: int) -> List[Dict]:
    """
    Inputs
    - sections: a list of dictionaries, each containing the following keys:
        - title: str - the main topic of this section of the document (very descriptive)
        - line_start: int - line number where the section begins (inclusive)
        - line_end: int - line number where the section ends (inclusive)
        - content: str - the text of the section
    - document_lines: a list of dictionaries, each containing the following keys:
        - content: str - the content of the line
        - element_type: str - the type of the element (e.g. "NarrativeText", "Image", etc.)
        - page_number: int - the page number of the line
        - image_path: str - the path to the image file (if applicable)
    - chunk_size: the maximum number of characters to include in each chunk
    - min_length_for_chunking: the minimum length of text to allow chunking (measured in number of characters); if the text is shorter than this, it will be added as a single chunk. If semantic sectioning is used, this parameter will be applied to each section. Setting this to a higher value than the chunk_size can help avoid unnecessary chunking of short documents or sections.

    Outputs
    - chunks: a list of dictionaries, each containing the following keys:
        - line_start: int - line number where the chunk begins (inclusive)
        - line_end: int - line number where the chunk ends (inclusive)
        - content: str - the text of the chunk
        - description: str - the description of the image (if applicable)
        - image_path: str - the path to the image file (if applicable)
        - page_start: int - the page number the chunk starts on
        - page_end: int - the page number the chunk ends on (inclusive)
    """

    chunk_line_indices = [] # list of tuples (start, end) for each chunk

    for section in sections:
        section_chunk_line_indices = [] # list of tuples (start, end) for each chunk within the section
        line_start = section['start']
        line_end = section['end']
        print(f"Chunking section from line {line_start} to {line_end}")
        for i in range(line_start, line_end+1):
            if document_lines[i]['element_type'] in ['Image', 'Figure']:
                # Split before and after the image/figure
                if line_start < i:
                    section_chunk_line_indices.append((line_start, i-1))
                section_chunk_line_indices.append((i, i))
                line_start = i + 1
            # TODO: handle the else case

        for line_start, line_end in section_chunk_line_indices:
            # don't try to chunk single-line sections
            if line_start == line_end:
                chunk_line_indices.append((line_start, line_end))
            
            text = "\n".join([document_lines[i]['content'] for i in range(line_start, line_end+1)])
            if len(text) >= min_length_for_chunking:
                chunks = chunk_sub_section(line_start, line_end, document_lines, chunk_size)
                for chunk_line_start, chunk_line_end in chunks:
                    chunk_line_indices.append((chunk_line_start, chunk_line_end))
            else:
                chunk_line_indices.append((line_start, line_end))

    print(chunk_line_indices)

    chunks = []
    for chunk_line_start, chunk_line_end in chunk_line_indices:
        chunk = {
            'line_start': chunk_line_start,
            'line_end': chunk_line_end,
            'content': "\n".join([document_lines[i]['content'] for i in range(chunk_line_start, chunk_line_end+1)]),
            'description': document_lines[chunk_line_start]['description'],
            'image_path': document_lines[chunk_line_start]['image_path'],
            'page_start': document_lines[chunk_line_start]['page_number'],
            'page_end': document_lines[chunk_line_end]['page_number']
        }
        chunks.append(chunk)

    return chunks

def chunk_sub_section(line_start: int, line_end: int, document_lines: List[Dict], max_length: int) -> List[Dict]:
    """
    Try to get as close to the max_length as possible without going over.

    Returns a list of (line_start, line_end) tuples for each chunk in the sub-section. The indices are inclusive.
    """
    chunks = []
    current_chunk = []
    current_length = 0
    for i in range(line_start, line_end+1):
        line = document_lines[i]['content']
        assert document_lines[i]['element_type'] not in ['Image', 'Figure'], "Images should be split out before calling this function"
        if current_length + len(line) > max_length:
            chunks.append((current_chunk[0], current_chunk[-1]))
            current_chunk = []
            current_length = 0
        current_chunk.append(i)
        current_length += len(line)
    if current_chunk:
        chunks.append((current_chunk[0], current_chunk[-1]))
    return chunks