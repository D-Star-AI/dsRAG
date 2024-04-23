# spRAG
spRAG is a high-performance RAG framework for unstructured data. It is especially good at handling complex queries over dense text, like financial reports and legal documents.

There are two key methods used to improve performance:
1. AutoContext
2. Relevant Segment Extraction (RSE)

#### AutoContext
AutoContext automatically injects document-level context into individual chunks prior to embedding them. This gives the embeddings a much more accurate and complete representation of the content and meaning of the text. In our testing, this feature leads to a dramatic improvement in retrieval quality. In addition to increasing the rate at which the correct information is retrieved, AutoContext also substantially reduces the rate at which irrelevant results show up in the search results. This reduces the rate at which the LLM misinterprets a piece of text in downstream chat and generation applications.

#### Relevant Segment Extraction
Relevant Segment Extraction (RSE) is a post-processing step that takes clusters of relevant chunks and intelligently combines them into longer sections of text that we call segments. These segments provide better context to the LLM than any individual chunk can. For simple factual questions, the answer is usually contained in a single chunk; but for more complex questions, the answer usually spans a longer section of text. The goal of RSE is to intelligently identify the section(s) of text that provide the most relevant information, without being constrained to fixed length chunks.

For example, suppose you have a bunch of SEC filings in a knowledge base and you ask “What were Apple’s key financial results in the most recent fiscal year?” RSE will identify the most relevant segment as the entire “Consolidated Statement of Operations” section, which will be 5-10 chunks long. Whereas if you ask “Who is Apple’s CEO?” the most relevant segment will be identified as a single chunk that mentions “Tim Cook, CEO.”

## Benchmark results
In our benchmarking, the combination of these two techniques dramatically improves accuracy on complex open-book question answering tasks. On one especially challenging benchmark, FinanceBench, spRAG gets accurate answers 43% of the time, compared to the vanilla RAG baseline which only gets 19% of questions correct.

# Architecture

## KnowledgeBase object
A KnowledgeBase object takes in documents (in the form of raw text) and does chunking and embedding on them, along with a few other preprocessing operations. Then at query time you feed in queries and it returns the most relevant segments of text.

KnowledgeBase objects are persistent by default. The full configuration needed to reconstruct the object gets saved as a JSON file upon creation and updating.

## Components
There are five key components that define the configuration of a KnowledgeBase, each of which are customizable:
1. VectorDB
2. ChunkDB
3. Embedding
4. Reranker
5. LLM

There are defaults for each of these components, as well as alternative options included in the repo. You can also define fully custom components by subclassing the base classes and passing in an instance of that subclass to the KnowledgeBase constructor. 

#### VectorDB
The VectorDB component stores the embedding vectors, as well as a small amount of metadata.

#### ChunkDB
The ChunkDB stores the content of text chunks in a nested dictionary format, keyed on `doc_id` and `chunk_index`. This is used by RSE to retrieve the full text associated with specific chunks.

#### Embedding
The Embedding component defines the embedding model.

#### Reranker
The Reranker components define the reranker. This is used after the vector database search (and before RSE) to provide a more accurate ranking of chunks.

#### LLM
This defines the LLM to be used for document summarization, which is only used in AutoContext.

## Document upload flow
Documents -> chunking -> embedding -> chunk and vector database upsert

## Query flow
Queries -> vector database search -> reranking -> RSE -> results
