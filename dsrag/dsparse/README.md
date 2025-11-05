# dsParse
dsParse is a sub-module of dsRAG that does multimodal file parsing, semantic sectioning, and chunking. You provide a file path (and some config params) and receive nice clean chunks.

```python
sections, chunks = parse_and_chunk(
    kb_id = "sample_kb",
    doc_id = "sample_doc",
    file_parsing_config={
        "use_vlm": True,
        "vlm_config": {
            "provider": "gemini",
            "model": "gemini-2.0-flash",
        }
    }
    file_path="path/to/file.pdf",
)
```

dsParse can be used on its own, as shown above, or in conjunction with a dsRAG knowledge base. To use it with dsRAG, you just use the `add_document` function like normal, but set `use_vlm` to True in the `file_parsing_config` dictionary, and include a `vlm_config`.

```python
kb = KnowledgeBase(kb_id="mck_energy_test")
kb.add_document(
    doc_id="mck_energy_report",
    file_path=file_path,
    document_title="McKinsey Energy Report",
    file_parsing_config={
        "use_vlm": True,
        "vlm_config": {
            "provider": "gemini",
            "model": "gemini-2.0-flash",
        }
    }
)
```

## VLM clients
VLMs now support a class-based client abstraction (similar to LLM/Embedding/Reranker) that you can pass either at the KB level or per document. Legacy dict-based `vlm_config` remains fully supported.

- Quickstart with class-based client (serialized) and LocalFileSystem
```python
from dsrag.dsparse.main import parse_and_chunk
from dsrag.dsparse.file_parsing.vlm_clients import GeminiVLM
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem

sections, chunks = parse_and_chunk(
    kb_id="sample_kb",
    doc_id="sample_doc",
    file_path="/path/to/file.pdf",
    file_parsing_config={
        "use_vlm": True,
        "vlm": GeminiVLM(model="gemini-2.0-flash").to_dict(),
        "vlm_config": {"max_pages": 5, "vlm_max_concurrent_requests": 2},
    },
    file_system=LocalFileSystem(base_path="~/dsParse"),
)
```

- Fallback (preferred, class-based):
```python
from dsrag.dsparse.main import parse_and_chunk
from dsrag.dsparse.file_parsing.vlm_clients import GeminiVLM

primary = GeminiVLM(model="gemini-2.0-flash").to_dict()
fallback = GeminiVLM(model="gemini-2.5-flash").to_dict()

sections, chunks = parse_and_chunk(
    kb_id="kb",
    doc_id="doc",
    file_path="/path/to/file.pdf",
    file_parsing_config={
        "use_vlm": True,
        "vlm": primary,
        "vlm_fallback": fallback,
        "vlm_config": {"max_pages": 5},
    },
)
```

- Legacy path (still valid):
```python
from dsrag.dsparse.main import parse_and_chunk

sections, chunks = parse_and_chunk(
    kb_id="kb",
    doc_id="doc",
    file_path="/path/to/file.pdf",
    file_parsing_config={
        "use_vlm": True,
        "vlm_config": {
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "max_pages": 5,
            # Optional legacy fallback
            "fallback_provider": "gemini",
            "fallback_model": "gemini-2.5-flash",
        },
    },
)
```

- Images already exist
If you’ve pre-extracted page images into the configured FileSystem directory structure, you can reuse them:
```python
sections, chunks = parse_and_chunk(
    kb_id="kb",
    doc_id="doc",
    file_path="/path/to/file.pdf",  # path still required for metadata, but images won’t be regenerated
    file_parsing_config={
        "use_vlm": True,
        "vlm": GeminiVLM(model="gemini-2.0-flash").to_dict(),
        "vlm_config": {"images_already_exist": True},
    },
)
```

- Notes
  - Parallelism controls and DPI are in `vlm_config` (e.g., `vlm_max_concurrent_requests`, `dpi`).
  - Page images and `elements.json` are saved via the configured `FileSystem`.
  - Environment variable `GEMINI_API_KEY` is required for `GeminiVLM`. Clear errors are raised if missing.

## Installation
If you want to use dsParse on its own, without installing the full `dsrag` package, there is a standalone Python package available for dsParse, which can be installed with `pip install dsparse`. If you already have `dsrag` installed, you DO NOT need to separately install `dsparse`.

To use the VLM file parsing functionality, you'll need to install one external dependency: poppler. This is used to convert PDFs to page images. On a Mac you can install it with `brew install poppler`.

## Multimodal file parsing
dsParse uses a vision language model (VLM) to parse documents. This has a few advantages:
- It can provide descriptions for visual elements, like images and figures.
- It can parse documents that don't have extractable text (i.e. those that require OCR).
- It can accurately parse documents with complex structures.
- It can accurately categorize page content into element types.

When it comes across an element on the page that can't be accurately represented with text alone, like an image or figure (chart, graph, diagram, etc.), it provides a text description of it. This can then be used in the embedding and retrieval pipeline. 

The default model, `gemini-2.0-flash`, is a fast and cost-effective option with roughly state-of-the-art performance.

### Element types
Page content is categorized into the following eight categories by default:
- NarrativeText
- Figure
- Image
- Table
- Header
- Footnote
- Footer
- Equation

You can also choose to define your own categories and the VLM will be prompted accordingly.

You can choose to exclude certain element types. By default, Header and Footer elements are excluded, as they rarely contain valuable information and they break up the flow between pages. For example, if you wanted to exclude footnotes, in addition to headers and footers, you would do: `exclude_elements = ["Header", "Footer", "Footnote"]`.

## Using page images for full multimodal RAG functionality
While modern VLMs, like Gemini and Claude 3.5, are now better than traditional OCR and bounding box extraction methods at converting visual elements on a page to text or bounding boxes, they still aren’t perfect. For fully visual elements, like images or charts, getting an accurate bounding box that includes all necessary surrounding context, like legends and axis titles, is only about 90% reliable with even the best VLM models. For semi-visual content, like tables and equations, converting to plain text is also not quite perfect yet. The problem with errors at the file parsing stage is that they propagate all the way to the generation stage.

For all of these element types, it’s more reliable to just send in the original page images to the generative model as context. That ensures that no context is lost, and that OCR and other parsing errors don’t propagate to the final response generated by the model. Images are no more expensive to process than extracted text (with the exception of a few models, like GPT-4o Mini, with weird image input pricing). In fact, for pages with dense text, a full page image might actually be cheaper than using the text itself.

## Semantic sectioning and chunking
Semantic sectioning uses an LLM to break a document into sections. It works by annotating the document with line numbers and then prompting an LLM to identify the starting lines for each “semantically cohesive section.” These sections should be anywhere from a few paragraphs to a few pages long. The sections then get broken into smaller chunks if needed. The LLM also generates descriptive titles for each section. When using dsParse with a dsRAG knowledge base, these section titles get used in the contextual chunk headers created by AutoContext, which provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

The default model for semantic sectioning is `gpt-4o-mini`, but similar or stronger models like `gemini-2.0-flash` will also work well.

## Cost and latency/throughput estimation

### VLM file parsing
An obvious concern with using a VLM to parse documents is the cost. Let's run the numbers:

VLM file parsing cost calculation (`gemini-2.0-flash`)
- Input tokens for images are calculated based on the number of 768x768 tiles needed. At the standard dpi of 100 (or even up to around 150), this usually means 4 tiles. Each tile is counted as 258 tokens.
- Text input (prompt) + image input: 500 (text) + 4x258 (image) tokens x $0.10/10^6 per token = $0.0001532
- Text output: 700 tokens x $0.40/10^6 per token = $0.0002800
- Total: $0.0004332/page or **$0.43 per 1000 pages**

This is substantially cheaper than commercially available OCR/PDF parsing services. Unstructured and Azure Document Intelligence, for example, both cost $10 per 1000 pages. Reducto is generally $10-20 per 1000 pages.

What about latency and throughput? Since each page is processed independently, this is a highly parallelizable problem. The main limiting factor then is the rate limits imposed by the VLM provider. The current rate limit for `gemini-2.0-flash` on the highest tier is 30k requests per minute. Since dsParse uses one request per page, that means the limit is 30k pages per minute. Processing a single page takes around 15-20 seconds, so that's the minimum latency for processing a document.

### Semantic sectioning
Semantic sectioning produces far fewer output tokens, so it ends up being a bit cheaper than the file parsing step.

Semantic sectioning cost calculation (`gpt-4o-mini`)
- Input: 800 tokens x $0.15/10^6 per token = $0.00012
- Output: 50 tokens x $0.60/10^6 per token = $0.00003
- Total: $0.00015/page or **$0.15 per 1000 pages**

Document text is processed in ~5000 token mega-chunks, which is roughly ten pages on average. These mega-chunks are processed in parallel for each document. Processing each mega-chunk only takes a few seconds, though, so even a large document of a few hundred pages should only take 5-10 seconds. Rate limits for the OpenAI API are heavily dependent on the usage tier you're in.
