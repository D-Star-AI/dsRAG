from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..models.types import Line, Section, Chunk

def chunk_document(sections: List[Section], document_lines: List[Line], chunk_size: int, min_length_for_chunking: int) -> List[Chunk]:
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
        - content: str - the text of the chunk (or a description of the image if applicable)
        - page_start: int - the page number the chunk starts on (inclusive)
        - page_end: int - the page number the chunk ends on (inclusive)
        - section_index: int - the index of the section this chunk belongs to
        - is_visual: bool - whether the chunk is a visual element (image, figure, etc.)
    """

    chunks = []
    for section_index, section in enumerate(sections):
        section_chunk_line_indices = [] # list of tuples (start, end) for each chunk within the section
        line_start = section['start']
        line_end = section['end']

        visual_line_indices = [i for i in range(line_start, line_end+1) if document_lines[i]['is_visual']]

        # get the sub-section indices
        prev_line = line_start
        for idx in sorted(visual_line_indices):
            if prev_line < idx:
                # Add text sub-section before the image
                section_chunk_line_indices.append((prev_line, idx - 1))
            # Add the image as a separate sub-section
            section_chunk_line_indices.append((idx, idx))
            prev_line = idx + 1

        # Add any remaining text after the last image
        if prev_line <= line_end:
            section_chunk_line_indices.append((prev_line, line_end))

        # chunk the sub-sections
        for line_start, line_end in section_chunk_line_indices:
            text = "\n".join([document_lines[i]['content'] for i in range(line_start, line_end+1)])
            
            # don't chunk visual elements
            if document_lines[line_start]['is_visual']:
                # add the sub-section as a single chunk
                chunk = Chunk(
                    line_start=line_start,
                    line_end=line_end,
                    content=text,
                    page_start=document_lines[line_start].get('page_number', None),
                    page_end=document_lines[line_end].get('page_number', None),
                    section_index=section_index,
                    is_visual=True,
                )
                chunks.append(chunk)
            elif len(text) < min_length_for_chunking:
                # add the sub-section as a single chunk
                chunk = Chunk(
                    line_start=line_start,
                    line_end=line_end,
                    content=text,
                    page_start=document_lines[line_start].get('page_number', None),
                    page_end=document_lines[line_end].get('page_number', None),
                    section_index=section_index,
                    is_visual=False,
                )
                chunks.append(chunk)
            else:
                chunks_text, chunk_line_indices = chunk_sub_section(line_start, line_end, document_lines, chunk_size)
                for chunk_text, (chunk_line_start, chunk_line_end) in zip(chunks_text, chunk_line_indices):
                    chunk = Chunk(
                        line_start=chunk_line_start,
                        line_end=chunk_line_end,
                        content=chunk_text,
                        page_start=document_lines[chunk_line_start].get('page_number', None),
                        page_end=document_lines[chunk_line_end].get('page_number', None),
                        section_index=section_index,
                        is_visual=False,
                    )
                    chunks.append(chunk)

    return chunks

def chunk_sub_section(line_start: int, line_end: int, document_lines: List[Line], max_length: int) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    A sub-section is a portion of a section that is separated by images or figures. 
    - If there are no images or figures then the entire section is considered a sub-section.
    - This function chunks the sub-section into smaller pieces of text.
    """
    # Concatenate the lines into a single string with newline delimiters
    concatenated_text = ""
    line_offsets = []  # List of tuples (start_char_index, end_char_index) for each line
    current_offset = 0

    for i in range(line_start, line_end + 1):
        line = document_lines[i]['content']
        concatenated_text += line + "\n"  # Adding newline as delimiter
        start = current_offset
        end = current_offset + len(line)
        line_offsets.append((start, end))
        current_offset = end + 1  # +1 for the newline character

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=0,
        length_function=len
    )

    # Split the text into chunks
    documents = text_splitter.create_documents([concatenated_text])
    chunks_text = [doc.page_content for doc in documents]

    # To map chunks back to line numbers, track the character indices
    # For each chunk, determine which lines fall into it
    # We'll iterate through the concatenated text and map characters to lines

    # First, prepare a list of line boundaries
    # Each entry corresponds to a line and contains its start and end character indices
    # Example: [(0, 50), (51, 100), ...]
    # Note: The end index is exclusive

    # Adjust the last line's end to exclude the final newline
    if line_offsets:
        last_line_start, last_line_end = line_offsets[-1]
        line_offsets[-1] = (last_line_start, last_line_end)  # Exclude the newline

    # Create list of tuples with line indices and their character ranges
    line_char_ranges = [
        (i, line_offsets[i - line_start][0], line_offsets[i - line_start][1])
        for i in range(line_start, line_end + 1)
    ]

    # Iterate through each chunk and determine the corresponding line indices
    chunk_line_indices = []
    current_char = 0
    for chunk_text in chunks_text:
        chunk_length = len(chunk_text)
        chunk_start_char = current_char
        chunk_end_char = current_char + chunk_length
        current_char = chunk_end_char + 1  # +1 for the newline delimiter

        # Map chunk to line indices
        chunk_line_start, chunk_line_end = find_lines_in_range(chunk_start_char, chunk_end_char, line_char_ranges, line_start, line_end)
        chunk_line_indices.append((chunk_line_start, chunk_line_end))

    # merge the last two chunks if the last chunk is too small
    if len(chunks_text) > 1:
        last_chunk_text = chunks_text[-1]
        penultimate_chunk_text = chunks_text[-2]
        if len(last_chunk_text) < max_length // 2:
            # merge the last two chunks
            merged_text = penultimate_chunk_text + "\n" + last_chunk_text
            chunks_text[-2] = merged_text
            chunk_line_indices[-2] = (chunk_line_indices[-2][0], chunk_line_indices[-1][1])
            chunks_text.pop()
            chunk_line_indices.pop()

    assert len(chunks_text) == len(chunk_line_indices), "Mismatch between chunk text and line indices"
    return chunks_text, chunk_line_indices

# Function to find lines within a given character range
def find_lines_in_range(chunk_start: int, chunk_end: int, line_char_ranges: List[Tuple], line_start: int, line_end: int) -> Tuple[int, int]:
    """
    Inputs
    - chunk_start: Start character index of the chunk
    - chunk_end: End character index of the chunk
    - line_char_ranges: List of tuples (line_idx, start_char, end_char) for each line
    - line_start: start line index for this section
    - line_end: end line index for this section

    Outputs
    - Tuple of line indices corresponding to the chunk
    """
    chunk_line_start = None
    chunk_line_end = None

    # First pass: Look for direct overlaps
    for line_idx, start, end in line_char_ranges:
        # Check if chunk starts at or within this line
        if start <= chunk_start <= end + 1:  # +1 to include newline position
            chunk_line_start = line_idx
            
        # Check if chunk ends at or within this line
        if start <= chunk_end <= end + 1:    # +1 to include newline position
            chunk_line_end = line_idx
            
        # Check for lines fully contained within chunk
        if chunk_start < start and chunk_end > end:
            if chunk_line_start is None:
                chunk_line_start = line_idx
            chunk_line_end = line_idx

    # Fallback in case no lines are found
    if chunk_line_start is None:
        chunk_line_start = line_start
    if chunk_line_end is None:
        chunk_line_end = line_end

    return (chunk_line_start, chunk_line_end)