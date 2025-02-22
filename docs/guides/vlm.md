# VLM Integration Guide

dsRAG supports Vision Language Model (VLM) integration for enhanced file parsing capabilities. Currently, this feature supports Google Cloud's Vertex AI platform.

## Configuration

To use VLM for file parsing, you need to configure the `file_parsing_config` when adding documents:

```python
file_parsing_config = {
    "use_vlm": True,  # Enable VLM parsing
    "vlm_config": {
        "provider": "vertex_ai",  # Currently the only supported provider
        "model": "your_model_name",  # Specify the VLM model
        "project_id": "your_gcp_project_id",  # Required for Vertex AI
        "location": "your_gcp_location",  # Required for Vertex AI
        "save_path": "path/to/save/files",  # For intermediate files
        "exclude_elements": ["Header", "Footer"]  # Elements to exclude from parsing
    }
}

# Add document with VLM parsing
kb.add_document(
    doc_id="my_document",
    file_path="path/to/document.pdf",
    file_parsing_config=file_parsing_config
)
```

## Note on Additional Content

The README doesn't contain detailed information about:
- Supported file types for VLM parsing
- Best practices for VLM usage
- Performance considerations
- Example outputs
- Troubleshooting guidelines

These sections should be added manually to provide a complete guide. 