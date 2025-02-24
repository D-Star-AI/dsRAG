# VLM File Parsing

dsRAG supports Vision Language Model (VLM) integration for enhanced PDF parsing capabilities. This feature is particularly useful for documents with complex layouts, tables, diagrams, or other visual elements that traditional text extraction might miss.

## Overview

When VLM parsing is enabled:

1. The PDF is converted to images (one per page)
2. Each page image is analyzed by the VLM to identify and extract text and visual elements
3. The extracted elements are converted to a structured format with page numbers preserved

## Supported Providers

Currently supported VLM providers:

- Google's Gemini (default)
- Google Cloud Vertex AI

## Configuration

To use VLM for file parsing, configure the `file_parsing_config` when adding documents:

```python
file_parsing_config = {
    "use_vlm": True,  # Enable VLM parsing
    "vlm_config": {
        # VLM Provider (required)
        "provider": "gemini",  # or "vertex_ai"
        
        # Model name (optional - defaults based on provider)
        "model": "gemini-2.0-flash",  # default for Gemini
        
        # For Vertex AI, additional required fields:
        "project_id": "your-gcp-project-id",
        "location": "us-central1",
        
        # Elements to exclude from parsing (optional)
        "exclude_elements": ["Header", "Footer"],
        
        # Whether images are pre-extracted (optional)
        "images_already_exist": False
    },
    
    # Save page images even if VLM unused (optional)
    "always_save_page_images": False
}

# Add document with VLM parsing (must be a PDF file)
kb.add_document(
    doc_id="my_document",
    file_path="path/to/document.pdf",  # Only PDF files are supported with VLM
    file_parsing_config=file_parsing_config
)
```

## Element Types

The VLM is prompted to categorize page content into the following element types:

Text Elements:

- NarrativeText: Main text content including paragraphs, lists, and titles
- Header: Page header content (typically at top of page)
- Footnote: References or notes at bottom of content
- Footer: Page footer content (at very bottom of page)

Visual Elements:

- Figure: Charts, graphs, and diagrams with associated titles and legends
- Image: Photos, illustrations, and other visual content
- Table: Tabular data arrangements with titles and captions
- Equation: Mathematical formulas and expressions

By default, Header and Footer elements are excluded from parsing as they rarely contain valuable information and can break the flow between pages. You can modify which elements to exclude using the `exclude_elements` configuration option.

## Best Practices

1. Use VLM parsing for documents with:
    - Complex layouts
    - Important visual elements
    - Tables that need precise formatting
    - Mathematical formulas
    - Scanned documents that need OCR

2. Consider traditional parsing for:
    - Simple text documents
    - Documents where visual layout isn't critical
    - Large volumes of documents (VLM parsing is slower and more expensive)

3. Configure `exclude_elements` to ignore irrelevant elements like headers/footers