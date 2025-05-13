import os
import sys
import time
from pprint import pprint
from pypdf import PdfReader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsrag.dsparse.sectioning_and_chunking.semantic_sectioning import get_sections_from_str

def main():
    # Load a test document
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #test_file_path = os.path.join(current_dir, "..", "tests", "data", "levels_of_agi.pdf")
    test_file_path = os.path.join(current_dir, "..", "tests", "data", "nike_2023_annual_report.txt")

    # Extract text based on file type
    file_ext = os.path.splitext(test_file_path)[1].lower()
    if file_ext == '.pdf':
        # Extract text from PDF file
        pdf_reader = PdfReader(test_file_path)
        test_document = ""
        for page in pdf_reader.pages:
            test_document += page.extract_text() + "\n"
    elif file_ext == '.txt':
        # Read text file directly
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_document = f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Only .pdf and .txt files are supported.")
    
    print(f"Document length: {len(test_document)} characters")
    
    # Configuration for both approaches
    semantic_sectioning_config = {
        "use_semantic_sectioning": True,
        "llm_provider": "gemini",
        "model": "gemini-2.0-flash",
        "language": "en",
    }
    
    # Configure parallel processing
    semantic_sectioning_config["llm_max_concurrent_requests"] = 100

    print("\nRunning semantic sectioning (with parallel processing)...")
    start_time = time.time()
    sections, _ = get_sections_from_str(
        document=test_document,
        max_characters_per_window=20000,
        semantic_sectioning_config=semantic_sectioning_config
    )
    processing_time = time.time() - start_time
    print(f"Processing took {processing_time:.2f} seconds")
    print(f"Found {len(sections)} sections using up to {semantic_sectioning_config['llm_max_concurrent_requests']} concurrent LLM calls")

    # Print section titles
    print("\nSectioning results:")
    for i, section in enumerate(sections):
        print(f"  Section {i+1}: {section['title']}")

if __name__ == "__main__":
    main()