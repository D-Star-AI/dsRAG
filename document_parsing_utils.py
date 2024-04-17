import PyPDF2
import docx2txt

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store the extracted text
        extracted_text = ""

        # Loop through all the pages in the PDF file
        for page_num in range(len(pdf_reader.pages)):
            # Extract the text from the current page
            page_text = pdf_reader.pages[page_num].extract_text()

            # Add the extracted text to the final text
            extracted_text += page_text

    return extracted_text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)