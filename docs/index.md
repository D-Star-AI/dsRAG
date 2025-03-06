# Welcome to dsRAG

dsRAG is a retrieval engine for unstructured data. It is especially good at handling challenging queries over dense text, like financial reports, legal documents, and academic papers. dsRAG achieves substantially higher accuracy than vanilla RAG baselines on complex open-book question answering tasks. On one especially challenging benchmark, [FinanceBench](https://arxiv.org/abs/2311.11944), dsRAG gets accurate answers 96.6% of the time, compared to the vanilla RAG baseline which only gets 32% of questions correct.

## Key Methods

dsRAG uses three key methods to improve performance over vanilla RAG systems:

### 1. Semantic Sectioning
Semantic sectioning uses an LLM to break a document into sections. It works by annotating the document with line numbers and then prompting an LLM to identify the starting and ending lines for each "semantically cohesive section." These sections should be anywhere from a few paragraphs to a few pages long. The sections then get broken into smaller chunks if needed. The LLM is also prompted to generate descriptive titles for each section. These section titles get used in the contextual chunk headers created by AutoContext, which provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

### 2. AutoContext
AutoContext creates contextual chunk headers that contain document-level and section-level context, and prepends those chunk headers to the chunks prior to embedding them. This gives the embeddings a much more accurate and complete representation of the content and meaning of the text. In our testing, this feature leads to a dramatic improvement in retrieval quality. In addition to increasing the rate at which the correct information is retrieved, AutoContext also substantially reduces the rate at which irrelevant results show up in the search results. This reduces the rate at which the LLM misinterprets a piece of text in downstream chat and generation applications.

### 3. Relevant Segment Extraction (RSE)
Relevant Segment Extraction (RSE) is a query-time post-processing step that takes clusters of relevant chunks and intelligently combines them into longer sections of text that we call segments. These segments provide better context to the LLM than any individual chunk can. For simple factual questions, the answer is usually contained in a single chunk; but for more complex questions, the answer usually spans a longer section of text. The goal of RSE is to intelligently identify the section(s) of text that provide the most relevant information, without being constrained to fixed length chunks.

## Quick Example

```python
from dsrag.create_kb import create_kb_from_file

# Create a knowledge base from a file
file_path = "path/to/your/document.pdf"
kb_id = "my_knowledge_base"
kb = create_kb_from_file(kb_id, file_path)

# Query the knowledge base
search_queries = ["What are the main topics covered?"]
results = kb.query(search_queries)
for segment in results:
    print(segment)
```

## Evaluation Results

### FinanceBench
On [FinanceBench](https://arxiv.org/abs/2311.11944), which uses a corpus of hundreds of 10-Ks and 10-Qs with challenging queries that often require combining multiple pieces of information:

- Baseline retrieval pipeline: 32% accuracy
- dsRAG (with default parameters and Claude 3.5 Sonnet): 96.6% accuracy

### KITE Benchmark
On the [KITE](https://github.com/D-Star-AI/KITE) benchmark, which includes diverse datasets (AI papers, company 10-Ks, company handbooks, and Supreme Court opinions), dsRAG shows significant improvements:

|                         | Top-k    | RSE    | CCH+Top-k    | CCH+RSE    |
|-------------------------|----------|--------|--------------|------------|
| AI Papers               | 4.5      | 7.9    | 4.7          | 7.9        |
| BVP Cloud               | 2.6      | 4.4    | 6.3          | 7.8        |
| Sourcegraph             | 5.7      | 6.6    | 5.8          | 9.4        |
| Supreme Court Opinions  | 6.1      | 8.0    | 7.4          | 8.5        |
| **Average**             | 4.72     | 6.73   | 6.04         | 8.42       |

## Getting Started

Check out our [Quick Start Guide](getting-started/quickstart.md) to begin using dsRAG in your projects.

## Community and Support

- Join our [Discord](https://discord.gg/NTUVX9DmQ3) for community support
- Fill out our [use case form](https://forms.gle/RQ5qFVReonSHDcCu5) if using dsRAG in production
- Need professional help? Contact our [team](https://forms.gle/zbQwDJp7pBQKtqVT8) 