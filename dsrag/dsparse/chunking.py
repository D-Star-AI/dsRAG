from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    #chunk_line_indices = [] # list of tuples (start, end) for each chunk
    chunks = []

    for section in sections:
        section_chunk_line_indices = [] # list of tuples (start, end) for each chunk within the section
        line_start = section['start']
        line_end = section['end']
        print(f"Chunking section from line {line_start} to {line_end}")

        image_indices = [i for i in range(line_start, line_end+1) if document_lines[i]['element_type'] in ['Image', 'Figure']]

        # get the sub-section indices
        prev_line = line_start
        for idx in sorted(image_indices):
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
            
            # don't chunk images/figures or short sections
            if (line_start == line_end and document_lines[line_start]['element_type'] in ['Image', 'Figure']) or len(text) < min_length_for_chunking:
                # add the sub-section as a single chunk
                chunk = {
                    'line_start': line_start,
                    'line_end': line_end,
                    'content': text,
                    'description': document_lines[line_start].get('description', ''),
                    'image_path': document_lines[line_start].get('image_path', ''),
                    'page_start': document_lines[line_start].get('page_number', None),
                    'page_end': document_lines[line_end].get('page_number', None)
                }
                chunks.append(chunk)
            else:
                chunks_text, chunk_line_indices = chunk_sub_section(line_start, line_end, document_lines, chunk_size)
                for chunk_text, (chunk_line_start, chunk_line_end) in zip(chunks_text, chunk_line_indices):
                    chunk = {
                        'line_start': chunk_line_start,
                        'line_end': chunk_line_end,
                        'content': chunk_text,
                        'description': document_lines[chunk_line_start].get('description', ''),
                        'image_path': document_lines[chunk_line_start].get('image_path', ''),
                        'page_start': document_lines[chunk_line_start].get('page_number', None),
                        'page_end': document_lines[chunk_line_end].get('page_number', None)
                    }
                    chunks.append(chunk)

    return chunks

def chunk_sub_section(line_start: int, line_end: int, document_lines: List[Dict], max_length: int) -> List[Dict]:
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

    # Function to find lines within a given character range
    def find_lines_in_range(chunk_start: int, chunk_end: int) -> Tuple[int, int]:
        chunk_line_start = None
        chunk_line_end = None

        for line_idx, start, end in line_char_ranges:
            if start <= chunk_start < end:
                chunk_line_start = line_idx
            if start < chunk_end <= end:
                chunk_line_end = line_idx
            if chunk_start < start and chunk_end > end:
                if chunk_line_start is None:
                    chunk_line_start = line_idx
                chunk_line_end = line_idx

        # Handle cases where the chunk starts or ends exactly on a boundary
        if chunk_line_start is None:
            for line_idx, start, end in line_char_ranges:
                if start >= chunk_start and start < chunk_end:
                    chunk_line_start = line_idx
                    break

        if chunk_line_end is None:
            for line_idx, start, end in reversed(line_char_ranges):
                if end <= chunk_end and end > chunk_start:
                    chunk_line_end = line_idx
                    break

        # Fallback in case no lines are found
        if chunk_line_start is None:
            chunk_line_start = line_start
        if chunk_line_end is None:
            chunk_line_end = line_end

        return (chunk_line_start, chunk_line_end)

    # Iterate through each chunk and determine the corresponding line indices
    chunk_line_indices = []
    current_char = 0
    for chunk_text in chunks_text:
        chunk_length = len(chunk_text)
        chunk_start_char = current_char
        chunk_end_char = current_char + chunk_length
        current_char = chunk_end_char + 1  # +1 for the newline delimiter

        # Map chunk to line indices
        chunk_line_start, chunk_line_end = find_lines_in_range(chunk_start_char, chunk_end_char)
        chunk_line_indices.append((chunk_line_start, chunk_line_end))

    assert len(chunks_text) == len(chunk_line_indices), "Mismatch between chunk text and line indices"
    return chunks_text, chunk_line_indices


if __name__ == "__main__":
    sections = [
        {
            'title': 'Section 1',
            'start': 0,
            'end': 3,
            'content': 'This is the first section of the document.'
        },
        {
            'title': 'Section 2',
            'start': 4,
            'end': 7,
            'content': 'This is the second section of the document.'
        }
    ]

    document_lines = [
        {
            'content': 'This is the first line of the document. And here is another sentence.',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        },
        {
            'content': 'This is the second line of the document.',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        },
        {
            'content': 'This is the third line of the document........',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        },
        {
            'content': 'This is the fourth line of the document.',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        },
        {
            'content': 'This is the fifth line of the document. With another sentence.',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        },
        {
            'content': 'This is the sixth line of the document.',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        },
        {
            'content': 'This is the seventh line of the document.',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        },
        {
            'content': 'This is the eighth line of the document. And here is another sentence that is a bit longer',
            'element_type': 'NarrativeText',
            'page_number': 1,
            'image_path': None
        }
    ]

    chunk_size = 90
    min_length_for_chunking = 180

    # print line lengths
    for i in range(0, len(document_lines)):
        print(f"Line {i}: {len(document_lines[i]['content'])}")

    """
    # test chunk_sub_section
    chunks = chunk_sub_section(4, 7, document_lines, chunk_size)
    print ("\n")
    for chunk in chunks:
        print(chunk)
        chunk_length = len("\n".join([document_lines[i]['content'] for i in range(chunk[0], chunk[1]+1)]))
        print(f"Chunk length: {chunk_length}\n")

    """
    
    chunks = chunk_document(sections, document_lines, chunk_size, min_length_for_chunking)
    for chunk in chunks:
        chunk_start = chunk['line_start']
        chunk_end = chunk['line_end']
        print(f"Chunk from line {chunk_start} to {chunk_end}")
        chunk_length = len("\n".join([document_lines[i]['content'] for i in range(chunk_start, chunk_end+1)]))
        print(f"Chunk length: {chunk_length}\n")