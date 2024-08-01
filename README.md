# dsRAG
[![Discord](https://img.shields.io/discord/1234629280755875881.svg?label=Discord&logo=discord&color=7289DA)](https://discord.gg/NTUVX9DmQ3)

**Note:** If you’re using (or planning to use) dsRAG in production, please fill out this short [form](https://forms.gle/RQ5qFVReonSHDcCu5) telling us about your use case. This helps us prioritize new features. In return I’ll give you my personal email address, which you can use for priority email support.

## What is dsRAG?
dsRAG is a retrieval engine for unstructured data. It is especially good at handling challenging queries over dense text, like financial reports, legal documents, and academic papers. dsRAG achieves substantially higher accuracy than vanilla RAG baselines on complex open-book question answering tasks. On one especially challenging benchmark, [FinanceBench](https://arxiv.org/abs/2311.11944), dsRAG gets accurate answers 83% of the time, compared to the vanilla RAG baseline which only gets 19% of questions correct.

There are three key methods used to improve performance over vanilla RAG systems:
1. Semantic sectioning
2. AutoContext
3. Relevant Segment Extraction (RSE)

#### Semantic sectioning
Semantic sectioning uses an LLM to break a document into sections. It works by annotating the document with line numbers and then prompting an LLM to identify the starting and ending lines for each “semantically cohesive section.” These sections should be anywhere from a few paragraphs to a few pages long. The sections then get broken into smaller chunks if needed. The LLM is also prompted to generate descriptive titles for each section. These section titles get used in the contextual chunk headers created by AutoContext, which provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

#### AutoContext
AutoContext creates contextual chunk headers that contain document-level and section-level context, and prepends those chunk headers to the chunks prior to embedding them. This gives the embeddings a much more accurate and complete representation of the content and meaning of the text. In our testing, this feature leads to a dramatic improvement in retrieval quality. In addition to increasing the rate at which the correct information is retrieved, AutoContext also substantially reduces the rate at which irrelevant results show up in the search results. This reduces the rate at which the LLM misinterprets a piece of text in downstream chat and generation applications.

#### Relevant Segment Extraction
Relevant Segment Extraction (RSE) is a query-time post-processing step that takes clusters of relevant chunks and intelligently combines them into longer sections of text that we call segments. These segments provide better context to the LLM than any individual chunk can. For simple factual questions, the answer is usually contained in a single chunk; but for more complex questions, the answer usually spans a longer section of text. The goal of RSE is to intelligently identify the section(s) of text that provide the most relevant information, without being constrained to fixed length chunks.

For example, suppose you have a bunch of SEC filings in a knowledge base and you ask “What were Apple’s key financial results in the most recent fiscal year?” RSE will identify the most relevant segment as the entire “Consolidated Statement of Operations” section, which will be 5-10 chunks long. Whereas if you ask “Who is Apple’s CEO?” the most relevant segment will be identified as a single chunk that mentions “Tim Cook, CEO.”

# Tutorial

#### Installation
To install the python package, run
```console
pip install dsrag
```

#### Quickstart
By default, dsRAG uses OpenAI for embeddings and AutoContext, and Cohere for reranking, so to run the code below you'll need to make sure you have API keys for those providers set as environmental variables with the following names: `OPENAI_API_KEY` and `CO_API_KEY`. **If you want to run dsRAG with different models, take a look at the "Basic customization" section below.**

You can create a new KnowledgeBase directly from a file using the `create_kb_from_file` helper function:
```python
from dsrag.create_kb import create_kb_from_file

file_path = "dsRAG/tests/data/levels_of_agi.pdf"
kb_id = "levels_of_agi"
kb = create_kb_from_file(kb_id, file_path)
```
KnowledgeBase objects persist to disk automatically, so you don't need to explicitly save it at this point.

Now you can load the KnowledgeBase by its `kb_id` (only necessary if you run this from a separate script) and query it using the `query` method:
```python
from dsrag.knowledge_base import KnowledgeBase

kb = KnowledgeBase("levels_of_agi")
search_queries = ["What are the levels of AGI?", "What is the highest level of AGI?"]
results = kb.query(search_queries)
for segment in results:
    print(segment)
```

#### Basic customization
Now let's look at an example of how we can customize the configuration of a KnowledgeBase. In this case, we'll customize it so that it only uses OpenAI (useful if you don't have API keys for Anthropic and Cohere). To do so, we need to pass in a subclass of `LLM` and a subclass of `Reranker`. We'll use `gpt-4o-mini` for the LLM (this is what gets used for document and section summarization in AutoContext) and since OpenAI doesn't offer a reranker, we'll use the `NoReranker` class for that.
```python
from dsrag.llm import OpenAIChatAPI
from dsrag.reranker import NoReranker

llm = OpenAIChatAPI(model='gpt-4o-mini')
reranker = NoReranker()

kb = KnowledgeBase(kb_id="levels_of_agi", reranker=reranker, auto_context_model=llm)
```

Now we can add documents to this KnowledgeBase using the `add_document` method. Note that the `add_document` method takes in raw text, not files, so we'll have to extract the text from our file first. There are some utility functions for doing this in the `document_parsing.py` file.
```python
from dsrag.document_parsing import extract_text_from_pdf

file_path = "dsRAG/tests/data/levels_of_agi.pdf"
text = extract_text_from_pdf(file_path)
kb.add_document(doc_id=file_path, text=text)
```

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

The currently available options are:
- `BasicVectorDB`
- `WeaviateVectorDB`

#### ChunkDB
The ChunkDB stores the content of text chunks in a nested dictionary format, keyed on `doc_id` and `chunk_index`. This is used by RSE to retrieve the full text associated with specific chunks.

The currently available options are:
- `BasicChunkDB`

#### Embedding
The Embedding component defines the embedding model.

The currently available options are:
- `OpenAIEmbedding`
- `CohereEmbedding`
- `VoyageAIEmbedding`
- `OllamaEmbedding`

#### Reranker
The Reranker components define the reranker. This is used after the vector database search (and before RSE) to provide a more accurate ranking of chunks.

The currently available options are:
- `CohereReranker`
- `VoyageReranker`

#### LLM
This defines the LLM to be used for document title generation, document summarization, and section summarization in AutoContext.

The currently available options are:
- `OpenAIChatAPI`
- `AnthropicChatAPI`
- `OllamaChatAPI`

## Config dictionaries
There are two config dictionaries that can be passed in to `add_document` (`auto_context_config` and `semantic_sectioning_config`) and one that can be passed in to `query` (`rse_params`).

Default values will be used for any parameters not provided in these dictionaries, so if you just want to alter one or two parameters there's no need to send in the full dictionary.

auto_context_config
- use_generated_title: bool - whether to use an LLM-generated title if no title is provided (default is True)
- document_title_guidance: str - guidance for generating the document title
- get_document_summary: bool - whether to get a document summary (default is True)
- document_summarization_guidance: str
- get_section_summaries: bool - whether to get section summaries (default is False)
- section_summarization_guidance: str

semantic_sectioning_config
- llm_provider: the LLM provider to use for semantic sectioning - only "openai" and "anthropic" are supported at the moment
- model: the LLM model to use for semantic sectioning
- use_semantic_sectioning: if False, semantic sectioning will be skipped (default is True)

rse_params
- max_length: maximum length of a segment, measured in number of chunks
- overall_max_length: maximum length of all segments combined, measured in number of chunks
- minimum_value: minimum value of a segment, measured in relevance value
- irrelevant_chunk_penalty: float between 0 and 1
- overall_max_length_extension: the maximum length of all segments combined will be increased by this amount for each additional query beyond the first
- decay_rate
- top_k_for_document_selection: the number of documents to consider

## Document upload flow
Documents -> semantic sectioning -> AutoContext -> chunking -> embedding -> chunk and vector database upsert

## Query flow
Queries -> vector database search -> reranking -> RSE -> results

# Community and support
You can join our [Discord](https://discord.gg/NTUVX9DmQ3) to ask questions, make suggestions, and discuss contributions.

If you’re using (or planning to use) dsRAG in production, please fill out this short [form](https://forms.gle/RQ5qFVReonSHDcCu5) telling us about your use case. This helps us prioritize new features. In return I’ll give you my personal email address, which you can use for priority email support.
