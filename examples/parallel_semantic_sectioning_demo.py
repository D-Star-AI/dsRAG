import os
import sys
import time
from pprint import pprint
from pypdf import PdfReader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsrag.dsparse.sectioning_and_chunking.semantic_sectioning import get_sections_from_str
from dsrag.dsparse.sectioning_and_chunking.ss_parallel import get_sections_from_str_parallel

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
        "llm_provider": "openai",
        "model": "gpt-4.1-mini",
        "language": "en",
    }
    
    # Run with serial processing
    print("\nRunning with serial processing...")
    start_time = time.time()
    serial_sections, _ = get_sections_from_str(
        document=test_document,
        max_characters=20000,
        semantic_sectioning_config=semantic_sectioning_config
    )
    serial_time = time.time() - start_time
    print(f"Serial processing took {serial_time:.2f} seconds")
    print(f"Found {len(serial_sections)} sections")
    
    # Run with parallel processing (specify maximum concurrent LLM calls)
    parallel_config = semantic_sectioning_config.copy()
    parallel_config["max_concurrent_llm_calls"] = 10
    
    print("\nRunning with parallel processing (10 concurrent LLM calls)...")
    start_time = time.time()
    parallel_sections, _ = get_sections_from_str_parallel(
        document=test_document,
        max_characters_per_window=20000,
        semantic_sectioning_config=parallel_config
    )
    parallel_time = time.time() - start_time
    print(f"Parallel processing took {parallel_time:.2f} seconds")
    print(f"Found {len(parallel_sections)} sections")
    
    # Print performance improvement
    speedup = serial_time / parallel_time if parallel_time > 0 else float('inf')
    print(f"\nPerformance speedup: {speedup:.2f}x")
    
    # Print section titles for comparison
    print("\nSerial sectioning results:")
    for i, section in enumerate(serial_sections):
        print(f"  Section {i+1}: {section['title']}")
    
    print("\nParallel sectioning results:")
    for i, section in enumerate(parallel_sections):
        print(f"  Section {i+1}: {section['title']}")

if __name__ == "__main__":
    main()