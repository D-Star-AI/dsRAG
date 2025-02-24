# Installation

## Installing dsRAG

Install the package using pip:

```bash
pip install dsrag
```

If you want to use VLM file parsing, you will need one non-Python dependency: poppler. This is used for converting PDFs to images. On MacOS, you can install it using Homebrew:

```bash
brew install poppler
```

## Third-party dependencies

To get the most out of dsRAG, you'll need a few third-party API keys set as environment variables. Here are the most important ones:

- `OPENAI_API_KEY`: Recommended for embeddings, AutoContext, and semantic sectioning
- `CO_API_KEY`: Recommended for reranking
- `GEMINI_API_KEY`: Recommended for VLM file parsing