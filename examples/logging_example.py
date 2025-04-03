"""
Example demonstrating the logging framework in dsRAG.

This script shows how to configure logging for dsRAG and how different logging levels
affect the output.
"""

import logging
import sys
from dsrag.knowledge_base import KnowledgeBase
import os

# Configure basic logging to see INFO level messages
def configure_basic_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    print("Configured basic logging at INFO level")

# Configure detailed logging to see DEBUG level messages
def configure_detailed_logging():
    # Set up a more detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'
    )
    
    # Create a handler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.addHandler(console_handler)
    print("Configured detailed logging at DEBUG level")

# Configure JSON logging (requires python-json-logger)
def configure_json_logging():
    try:
        from pythonjsonlogger import jsonlogger
    except ImportError:
        print("pythonjsonlogger not installed. Run: pip install python-json-logger")
        return False
    
    # Create a JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d'
    )
    
    # Create a handler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # Configure the dsrag logger
    logger = logging.getLogger("dsrag")
    logger.setLevel(logging.DEBUG)
    
    # Ensure propagation is enabled
    logger.propagate = True
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(console_handler)
    print("Configured JSON logging at DEBUG level")
    return True

def main():
    # Choose one of the logging configurations
    # configure_basic_logging()
    configure_detailed_logging()
    # if not configure_json_logging():
    #     configure_basic_logging()
    
    # Remove the knowledge base directory if it exists
    kb_dir = os.path.expanduser("~/dsRAG_logging_example")
    if os.path.exists(kb_dir):
        import shutil
        shutil.rmtree(kb_dir)
    
    # Create a knowledge base
    print("\nCreating knowledge base...")
    kb = KnowledgeBase(
        kb_id="logging_example_kb",
        title="Logging Example KB",
        description="A knowledge base for demonstrating logging",
        storage_directory=kb_dir
    )
    
    # Add a document
    print("\nAdding document...")
    sample_text = """
    # dsRAG: Document-Section RAG
    
    dsRAG is a Python library for building and querying knowledge bases with state-of-the-art Retrieval-Augmented Generation (RAG).
    This library makes it easy to ingest documents, search them, and use them to improve LLM outputs with Relevant Segment Extraction (RSE).
    
    ## Key Features
    
    - **Section-aware document processing**: Ingest documents as semantically meaningful sections, not arbitrary chunks.
    - **Relevant Segment Extraction (RSE)**: Revolutionary approach that finds and extracts relevant segments, not just chunks.
    - **Multi-modal support**: Process documents with images (PDF files) using Vision Language Models (VLMs).
    - **Comprehensive databases**: Support for various vector databases and chunk storage options.
    - **Ready-to-use chat**: Built-in chat interfaces with citation support.
    """
    
    kb.add_document(
        doc_id="sample_doc",
        text=sample_text,
        document_title="dsRAG Documentation"
    )
    
    # Query the knowledge base
    print("\nQuerying knowledge base...")
    results = kb.query(
        search_queries=["What is Relevant Segment Extraction?"],
        rse_params="balanced"
    )
    
    # Display results
    print("\nQuery results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['doc_id']}")
        print(f"Score: {result['score']}")
        print(f"Content: {result['content'][:100]}...\n")
    
    print("Logging example completed. Check the output above to see the log messages.")

if __name__ == "__main__":
    main()