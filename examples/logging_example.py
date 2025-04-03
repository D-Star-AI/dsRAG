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

# Ensure python-json-logger is installed: pip install python-json-logger
try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    print("ERROR: python-json-logger not installed. Run: pip install python-json-logger")
    print("Falling back to basic console logging.")
    # Fallback to basic logging if json logger isn't available
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    jsonlogger = None # Flag that json logging is not available

def configure_json_logging_to_file(
    log_file_path: str = "dsrag_app.log.json",
    log_level_dsrag: int = logging.DEBUG,
    log_level_others: int = logging.INFO
):
    """
    Configures logging to output JSON formatted logs to a local file.

    Args:
        log_file_path: Path to the output log file.
        log_level_dsrag: Minimum logging level for 'dsrag' loggers.
        log_level_others: Minimum logging level for all other loggers (root).
    """
    if not jsonlogger: # Check if import failed
        print("Cannot configure JSON logging to file as python-json-logger is missing.")
        return False

    # --- Create JSON Formatter ---
    # Include standard fields. python-json-logger automatically adds 'extra'.
    formatter = jsonlogger.JsonFormatter(
        datefmt='%Y-%m-%dT%H:%M:%S%z'
    )

    # --- Create File Handler ---
    # Use 'a' for append mode. Ensure directory exists.
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # Handler processes all levels down to DEBUG
    file_handler.setFormatter(formatter)

    # --- Configure Loggers ---
    # Configure the root logger (for libraries)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root to DEBUG to allow all messages through

    # --- Attach Handler ---
    # Clear existing handlers from the root logger to avoid duplicates
    # and ensure only our file handler is used for the primary output.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Important part: Add a filter to the handler that allows DEBUG from dsrag
    # but only INFO and above from other loggers
    class DsragDebugFilter(logging.Filter):
        def filter(self, record):
            if record.name.startswith('dsrag'):
                return record.levelno >= log_level_dsrag
            return record.levelno >= log_level_others
    
    file_handler.addFilter(DsragDebugFilter())

    # Add the JSON file handler to the root logger.
    root_logger.addHandler(file_handler)

    # --- Optional: Set higher levels for noisy libraries if needed ---
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    #logging.getLogger("openai").setLevel(logging.WARNING)
    # ... etc.

    # --- Optional: Add Console Handler for basic feedback ---
    # You might still want basic INFO messages on the console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    console_handler.addFilter(DsragDebugFilter())  # Apply the same filter to console output
    root_logger.addHandler(console_handler) # Add alongside file handler

    logging.info(f"Configured JSON logging to file: {log_file_path}")
    logging.info(f"dsrag log level: {logging.getLevelName(log_level_dsrag)}, Other loggers level: {logging.getLevelName(log_level_others)}")
    
    return True

def reset_kb(kb_dir: str, kb_id: str):
    kb = KnowledgeBase(
        kb_id=kb_id,
        storage_directory=kb_dir
    )
    kb.delete()

def main():
    # Choose one of the logging configurations
    # configure_basic_logging()
    # configure_detailed_logging()

    # Define where you want the log file
    log_directory = os.path.expanduser("~/dsRAG_logging_example/logs")
    log_file = os.path.join(log_directory, "application_log.json")

    # Configure logging to file
    json_logging_configured = configure_json_logging_to_file(
        log_file_path=log_file,
        log_level_dsrag=logging.DEBUG, # Capture detailed dsrag logs
        log_level_others=logging.INFO   # Keep library logs less verbose
    )
    
    # Ensure the knowledge base directory exists
    kb_dir = os.path.expanduser("~/dsRAG_logging_example")
    kb_id = "logging_example_kb"
    os.makedirs(kb_dir, exist_ok=True)

    reset_kb(kb_dir, kb_id)
    
    # Create a knowledge base
    print("\nCreating knowledge base...")
    kb = KnowledgeBase(
        kb_id=kb_id,
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