# Configuration

dsRAG uses several configuration dictionaries to organize its many parameters. These configs can be passed to different methods of the KnowledgeBase class.

## Document Addition Configs

The following configuration dictionaries can be passed to the `add_document` method:

### AutoContext Config

```python
auto_context_config = {
    "use_generated_title": bool,  # whether to use an LLM-generated title if no title is provided (default: True)
    "document_title_guidance": str,  # Additional guidance for generating the document title (default: "")
    "get_document_summary": bool,  # whether to get a document summary (default: True)
    "document_summarization_guidance": str,  # Additional guidance for summarizing the document (default: "")
    "get_section_summaries": bool,  # whether to get section summaries (default: False)
    "section_summarization_guidance": str,  # Additional guidance for summarizing the sections (default: "")
    "llm_max_concurrent_requests": int  # Maximum concurrent requests for section summarization (default: 5)
}
```

### File Parsing Config

```python
file_parsing_config = {
    "use_vlm": bool,  # whether to use VLM for parsing the file (default: False)
    "vlm_config": {
        "provider": str,  # the VLM provider to use (default: "gemini", "vertex_ai" is also supported)
        "model": str,  # the VLM model to use (default: "gemini-2.0-flash")
        "project_id": str,  # GCP project ID (only required for "vertex_ai")
        "location": str,  # GCP location (only required for "vertex_ai")
        "save_path": str,  # path to save intermediate files during VLM processing
        "exclude_elements": list,  # element types to exclude (default: ["Header", "Footer"])
        "images_already_exist": bool  # whether images are pre-extracted (default: False)
    },
    "always_save_page_images": bool  # save page images even if VLM is not used (default: False)
}
```

### Semantic Sectioning Config

```python
semantic_sectioning_config = {
    "llm_provider": str,  # LLM provider (default: "openai", "anthropic" and "gemini" are also supported)
    "model": str,  # LLM model to use (default: "gpt-4o-mini")
    "use_semantic_sectioning": bool,  # if False, skip semantic sectioning (default: True)
    "llm_max_concurrent_requests": int  # Maximum concurrent requests for semantic sectioning (default: 5)
}
```

### Chunking Config

```python
chunking_config = {
    "chunk_size": int,  # maximum characters per chunk (default: 800)
    "min_length_for_chunking": int  # minimum text length to allow chunking (default: 2000)
}
```

## Query Config

The following configuration dictionary can be passed to the `query` method:

### RSE Parameters

```python
rse_params = {
    "max_length": int,  # maximum segment length in chunks (default: 15)
    "overall_max_length": int,  # maximum total length of all segments (default: 30)
    "minimum_value": float,  # minimum relevance value for segments (default: 0.5)
    "irrelevant_chunk_penalty": float,  # penalty for irrelevant chunks (0-1) (default: 0.18)
    "overall_max_length_extension": int,  # length increase per additional query (default: 5)
    "decay_rate": float,  # rate at which relevance decays (default: 30)
    "top_k_for_document_selection": int,  # maximum number of documents to consider (default: 10)
    "chunk_length_adjustment": bool  # whether to scale by chunk length (default: True)
}
```

## Metadata Query Filters

Some vector databases (currently only ChromaDB) support metadata filtering during queries. This allows for more controlled document selection.

Example metadata filter format:
```python
metadata_filter = {
    "field": str,  # The metadata field to filter by
    "operator": str,  # One of: 'equals', 'not_equals', 'in', 'not_in', 
                     # 'greater_than', 'less_than', 'greater_than_equals', 'less_than_equals'
    "value": str | int | float | list  # If list, all items must be same type
}
```

Example usage:
```python
# Filter with "equals" operator
metadata_filter = {
    "field": "doc_id",
    "operator": "equals",
    "value": "test_id_1"
}

# Filter with "in" operator
metadata_filter = {
    "field": "doc_id",
    "operator": "in",
    "value": ["test_id_1", "test_id_2"]
}
```