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

def parse_file_no_vlm(file_path: str) -> str:
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