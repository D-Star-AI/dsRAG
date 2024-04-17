# spRAG
spRAG is a high-performance RAG framework for unstructured data. There are two key methods used to improve performance:
1. AutoContext
2. Relevant Segment Extraction (RSE)

### AutoContext
AutoContext automatically injects document-level context into individual chunks prior to embedding them. This gives the embeddings a much more accurate and complete representation of the content and meaning of the text. In our testing, this feature leads to a dramatic improvement in retrieval quality. In addition to increasing the rate at which the correct information is retrieved, AutoContext also substantially reduces the rate at which irrelevant results show up in the search results. This reduces the rate at which the LLM misinterprets a piece of text in downstream chat and generation applications.

### Relevant Segment Extraction
Relevant Segment Extraction (RSE) is a post-processing step that takes clusters of relevant chunks and intelligently combines them into longer sections of text that we call segments. These segments provide better context to the LLM than any individual chunk can. For simple factual questions, the answer is usually contained in a single chunk; but for more complex questions, the answer usually spans a longer section of text. The goal of RSE is to intelligently identify the section(s) of text that provide the most relevant information, without being constrained to fixed length chunks.

For example, suppose you have a bunch of SEC filings in a knowledge base and you ask “What were Apple’s key financial results in the most recent fiscal year?” RSE will identify the most relevant segment as the entire “Consolidated Statement of Operations” section, which will be 5-10 chunks long. Whereas if you ask “Who is Apple’s CEO?” the most relevant segment will be identified as a single chunk that mentions “Tim Cook, CEO.”

## Benchmarking
In our benchmarking, the combination of these two technologies dramatically improves accuracy on complex open-book question answering tasks. On one especially challenging benchmark, FinanceBench, spRAG gets accurate answers 43% of the time, compared to the vanilla RAG baseline which only gets 19% of questions correct.

## spRAG object
An spRAG object (commonly referred to as a knowledge base) takes in documents (which could be files or raw text) and does chunking and embedding on them, along with a few other preprocessing operations. Then at query time you feed in queries and it returns a relevant knowledge string.
