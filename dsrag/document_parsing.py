import PyPDF2
import docx2txt


def extract_text_from_pdf(file_path: str) -> tuple[str, list]:
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store the extracted text
        extracted_text = ""

        # Loop through all the pages in the PDF file
        pages = []
        for page_num in range(len(pdf_reader.pages)):
            # Extract the text from the current page
            page_text = pdf_reader.pages[page_num].extract_text()
            pages.append(page_text)

            # Add the extracted text to the final text
            extracted_text += page_text

    return extracted_text, pages


def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path)


def parse_file(file_path: str) -> str:
    pdf_pages = None
    if file_path.endswith(".pdf"):
        text, pdf_pages = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r") as file:
            text = file.read()
    else:
        raise ValueError(
            "Unsupported file format. Only .txt, .md, .pdf and .docx files are supported."
        )
    
    return text, pdf_pages


def get_pages_from_chunks(full_text: str, pages: list[dict], chunks: list) -> list:

    current_page_index = 0
    current_page_start = 0

    formatted_chunks = []

    for chunk in chunks:
        chunk_text = chunk["chunk_text"]
        chunk_start = full_text.find(chunk_text)
        chunk_end = chunk_start + len(chunk_text)

        # Determine the page numbers for the current chunk
        chunk_pages = []
        while current_page_index < len(pages):
            page_text = pages[current_page_index]
            page_num = current_page_index + 1
            #page_start = current_page_start
            page_end = current_page_start + len(page_text)

            if chunk_start < page_end:
                chunk_pages.append(page_num)
                if chunk_end <= page_end:
                    break

            current_page_start = page_end
            current_page_index += 1

        formatted_chunks.append({
            "chunk_page_start": chunk_pages[0],
            "chunk_page_end": chunk_pages[-1],
            **chunk
        })
    
    return formatted_chunks