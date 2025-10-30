from dsrag.auto_context import (
    get_document_title,
    get_document_summary,
    get_section_summary,
    get_chunk_header,
)
from dsrag.llm import LLM
from dsrag.embedding import Embedding
from dsrag.database.chunk import ChunkDB
from dsrag.database.vector import VectorDB
from dsrag.custom_term_mapping import annotate_chunks
import logging
import time
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_section_summary(section, auto_context_model, document_title, auto_context_config, language, base_extra, i):
    ingestion_logger = logging.getLogger("dsrag.ingestion")
    if auto_context_config.get("get_section_summaries", False):
        section_summarization_guidance = auto_context_config.get("section_summarization_guidance", "")
        
        section_summary_start_time = time.perf_counter()
        section_summary = get_section_summary(
            auto_context_model=auto_context_model,
            section_text=section["content"],
            document_title=document_title,
            section_title=section["title"],
            section_summarization_guidance=section_summarization_guidance,
            language=language
        )
        section_summary_duration = time.perf_counter() - section_summary_start_time
        
        # Log duration and summary text
        ingestion_logger.debug("Generated section summary", extra={
            **base_extra,
            "step": "section_summary",
            "section_index": i,
            "section_title": section.get("title", "N/A"),
            "duration_s": round(section_summary_duration, 4),
            "summary_text": section_summary
        })
        
        return section_summary
    else:
        return ""

def auto_context(kb_id: str, auto_context_model: LLM, sections, chunks, text, doc_id, document_title, auto_context_config, language):
    ingestion_logger = logging.getLogger("dsrag.ingestion")
    base_extra = {"kb_id": kb_id, "doc_id": doc_id}
    step_start_time = time.perf_counter()
    
    ingestion_logger.debug("Starting AutoContext step", extra={**base_extra, "step": "auto_context_start"})

    # document title and summary
    if not document_title and auto_context_config.get("use_generated_title", True):
        document_title_guidance = auto_context_config.get(
            "document_title_guidance", ""
        )
        document_title = get_document_title(
            auto_context_model=auto_context_model,
            document_text=text,
            document_title_guidance=document_title_guidance,
            language=language
        )
    elif not document_title:
        document_title = doc_id

    if auto_context_config.get("get_document_summary", True):
        document_summarization_guidance = auto_context_config.get("document_summarization_guidance", "")
        document_summary = get_document_summary(
            auto_context_model,
            text,
            document_title=document_title,
            document_summarization_guidance=document_summarization_guidance,
            language=language
        )
    else:
        document_summary = ""

    # get section summaries in parallel
    if auto_context_config.get("get_section_summaries", False):
        # Get concurrent workers, default to 5 if not specified
        max_concurrent_workers = auto_context_config.get("llm_max_concurrent_requests", 5)
        with ThreadPoolExecutor(max_workers=min(max_concurrent_workers, len(sections))) as executor:
            future_to_section = {
                executor.submit(
                    process_section_summary, 
                    section, 
                    auto_context_model, 
                    document_title, 
                    auto_context_config, 
                    language, 
                    base_extra, 
                    i
                ): (i, section) for i, section in enumerate(sections)
            }
            
            # As futures complete, store results
            for future in as_completed(future_to_section):
                i, section = future_to_section[future]
                try:
                    section_summary = future.result()
                    sections[i]["summary"] = section_summary
                except Exception as e:
                    ingestion_logger.error(f"Error processing section {i}: {str(e)}", 
                                          extra={**base_extra, "section_index": i})
                    sections[i]["summary"] = ""
    else:
        # If not generating summaries, just set empty summaries
        for section in sections:
            section["summary"] = ""

    # add document title, document summary, and section summaries to the chunks
    for chunk in chunks:
        chunk["document_title"] = document_title
        chunk["document_summary"] = document_summary
        section_index = chunk["section_index"]
        if section_index is not None:
            chunk["section_title"] = sections[section_index]["title"]
            chunk["section_summary"] = sections[section_index]["summary"]

    # custom term mapping
    if auto_context_config.get("custom_term_mapping", None):
        raw_chunks = [chunk["content"] for chunk in chunks] # need to convert chunks to list of strings
        annotated_chunks = annotate_chunks(raw_chunks, auto_context_config["custom_term_mapping"])

    # prepare the chunks for embedding by prepending the chunk headers
    chunks_to_embed = []
    for i, chunk in enumerate(chunks):
        chunk_header = get_chunk_header(
            document_title=chunk["document_title"],
            document_summary=chunk["document_summary"],
            section_title=chunk["section_title"],
            section_summary=chunk["section_summary"],
        )
        if auto_context_config.get("custom_term_mapping", None):
            chunk["content"] = annotated_chunks[i] # override the chunk content with the annotated content if custom term mapping is used
        chunk_to_embed = f"{chunk_header}\n\n{chunk['content']}"
        chunks_to_embed.append(chunk_to_embed)

    step_duration = time.perf_counter() - step_start_time
    ingestion_logger.debug("AutoContext complete", extra={
        **base_extra, 
        "step": "auto_context", 
        "duration_s": round(step_duration, 4),
        "model": auto_context_model.__class__.__name__
    })

    return chunks, chunks_to_embed

rate_limits = {
    "openai": {
        "tpm": {
            "text-embedding-3-small": 1_000_000,
            "text-embedding-3-large": 1_000_000,
            "text-embedding-ada-002": 1_000_000
        },
        "rpm": {
            "text-embedding-3-small": 3000,
            "text-embedding-3-large": 3000,
            "text-embedding-ada-002": 3000
        },
        "max_tokens_per_request": 300_000
    },
    "cohere": {
        "tpm": {
            "embed-english-v3.0": 1_000_000,
            "embed-multilingual-v3.0": 1_000_000,
            "embed-english-light-v3.0": 1_000_000,
            "embed-multilingual-light-v3.0": 1_000_000
        },
        "rpm": {
            "embed-english-v3.0": 6000,
            "embed-multilingual-v3.0": 6000,
            "embed-english-light-v3.0": 6000,
            "embed-multilingual-light-v3.0": 6000
        },
        "max_tokens_per_request": 512
    },
    "voyageai": {
        "tpm": {
            "voyage-3-large": 3_000_000,
            "voyage-3.5": 8_000_000,
            "voyage-3.5-lite": 16_000_000,
            "voyage-finance-2": 3_000_000,
            "voyage-law-2": 3_000_000,
            "voyage-code-3": 3_000_000
        },
        "rpm": {
            "voyage-3-large": 2000,
            "voyage-3.5": 2000,
            "voyage-3.5-lite": 2000,
            "voyage-finance-2": 2000,
            "voyage-law-2": 2000,
            "voyage-code-3": 2000
        }
    },
    "ollama": {
        "tpm": {
            "llama2": 1_000_000,
            "llama3": 1_000_000,
            "all-minilm": 1_000_000,
            "nomic-embed-text": 1_000_000,
        },
        "rpm": {
            "llama2": 5000,
            "llama3": 5000,
            "all-minilm": 5000,
            "nomic-embed-text": 5000,
        }
    }
}

token_rate_limit = {
    "text-embedding-3-small": 100_000_000,
    "text-embedding-3-large": 100_000_000,
    "text-embedding-ada-002": 100_000_000,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "voyage-large-2": 1536,
    "voyage-law-2": 1024,
    "voyage-code-2": 1536,
    "llama2": 4096,
    "llama3": 4096,
    "all-minilm": 384,
    "nomic-embed-text": 768,
}

def get_token_encoder(embedding_model):
    if embedding_model.__class__.__name__ == "OpenAIEmbedding":
        return tiktoken.encoding_for_model(embedding_model.model)
    elif embedding_model.__class__.__name__ == "CohereEmbedding":
        return tiktoken.get_encoding("cl100k_base")
    elif embedding_model.__class__.__name__ == "VoyageAIEmbedding":
        return tiktoken.get_encoding("cl100k_base")
    elif embedding_model.__class__.__name__ == "OllamaEmbedding":
        return tiktoken.get_encoding("cl100k_base")

def get_embeddings_in_batches(kb_id, doc_id, embedding_model, chunks_to_embed):
    """
    Generate embeddings for chunks in batches while respecting token limits and rate limits.
    """
    ingestion_logger = logging.getLogger("dsrag.ingestion")
    context = {"kb_id": kb_id, "doc_id": doc_id}
    ingestion_logger.debug("Starting embedding step", extra={**context, "step": "embedding_start"})

    enc = tiktoken.encoding_for_model(embedding_model.model)
    max_tokens = int(300_000 * 0.75)
    chunk_embeddings, batch = [], []
    token_counter = 0

    def send_batch(batch_data):
        """Send a batch to the embedding model with retry logic for rate limits."""
        nonlocal context

        for retry in range(10):
            try:
                return embedding_model.get_embeddings(batch_data, input_type="document")
            except Exception as e:
                if "429" in str(e):  # Rate limit
                    wait_time = min(30 * (2 ** retry), 300)
                    ingestion_logger.warning(
                        f"Rate limit hit (attempt {retry + 1}). Retrying in {wait_time}s...",
                        extra={**context, "retry_attempt": retry + 1}
                    )
                    time.sleep(wait_time)
                else:
                    ingestion_logger.error(f"Embedding failed: {e}", extra=context)
                    raise
        ingestion_logger.error("Max retries reached for batch", extra=context)
        raise RuntimeError("Failed to embed batch after maximum retries")

    for idx, chunk in enumerate(chunks_to_embed):
        token_count = len(enc.encode(chunk))

        if token_counter + token_count > max_tokens and batch:
            # Send current batch before adding this chunk
            embeddings = send_batch(batch)
            chunk_embeddings.extend(embeddings)
            ingestion_logger.debug(
                f"Processed batch of {len(batch)} chunks (up to index {idx})",
                extra=context,
            )
            batch.clear()
            token_counter = 0
            time.sleep(60)  # respect rate limit window

        batch.append(chunk)
        token_counter += token_count

    # Send any remaining batch
    if batch:
        embeddings = send_batch(batch)
        chunk_embeddings.extend(embeddings)

    ingestion_logger.debug(
        f"Completed embeddings: {len(chunk_embeddings)} total chunks processed",
        extra={**context, "step": "embedding_complete"},
    )

    return chunk_embeddings

        
def get_embeddings(kb_id, doc_id, embedding_model, chunks_to_embed):
    ingestion_logger = logging.getLogger("dsrag.ingestion")
    base_extra = {"kb_id": kb_id, "doc_id": doc_id}

    ingestion_logger.debug("Starting embedding step", extra={**base_extra, "step": "embedding_start"})

    max_retries = 10
    batch_size = 800
    chunk_embeddings = []

    for batch_start in range(0, len(chunks_to_embed), batch_size):
        batch = chunks_to_embed[batch_start:batch_start + batch_size]
        retry_count = 0

        while retry_count < max_retries:
            try:
                embeddings = embedding_model.get_embeddings(batch, input_type="document")
                chunk_embeddings.extend(embeddings)
                break  # success
            except Exception as e:
                if "429" in str(e):  # rate limit
                    wait_time = min(30 * (2 ** retry_count), 300)  # exponential backoff up to 5 min
                    ingestion_logger.warning(
                        f"Rate limit exceeded. Retrying in {wait_time} seconds...",
                        extra={**base_extra, "retry_attempt": retry_count + 1}
                    )
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    ingestion_logger.error(f"Embedding error: {e}", extra=base_extra)
                    raise  # stop on non-rate-limit errors

        else:
            ingestion_logger.error(f"Failed after {max_retries} retries for batch starting at index {batch_start}", extra=base_extra)
            raise RuntimeError("Max retries exceeded for embedding batch")

    return chunk_embeddings


def add_chunks_to_db(chunk_db: ChunkDB, chunks, chunks_to_embed, chunk_embeddings, metadata, doc_id, supp_id):
    # add the chunks to the chunk database
    assert len(chunks) == len(chunk_embeddings) == len(chunks_to_embed)
    chunk_db.add_document(
        doc_id,
        {
            i: {
                "chunk_text": chunk["content"],
                "document_title": chunk["document_title"],
                "document_summary": chunk["document_summary"],
                "section_title": chunk["section_title"],
                "section_summary": chunk["section_summary"],
                "chunk_page_start": chunk.get("page_start", None),
                "chunk_page_end": chunk.get("page_end", None),
                "is_visual": chunk.get("is_visual", False),
            }
            for i, chunk in enumerate(chunks)
        },
        supp_id,
        metadata
    )

def add_vectors_to_db(vector_db: VectorDB, chunks, chunk_embeddings, metadata, doc_id):
    # create metadata list to add to the vector database
    vector_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_page_start = chunk.get("page_start", "")
        chunk_page_end = chunk.get("page_end", "")
        # Some vector dbs don't accept None as a value, so we need to convert it to an empty string
        if chunk_page_start is None:
            chunk_page_start = ""
        if chunk_page_end is None:
            chunk_page_end = ""
        vector_metadata.append(
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_header": get_chunk_header(
                    document_title=chunk["document_title"],
                    document_summary=chunk["document_summary"],
                    section_title=chunk["section_title"],
                    section_summary=chunk["section_summary"],
                ),
                "chunk_page_start": chunk_page_start,
                "chunk_page_end": chunk_page_end,
                # Add the rest of the metadata to the vector metadata
                **metadata
            }
        )

    # add the vectors and metadata to the vector database
    vector_db.add_vectors(vectors=chunk_embeddings, metadata=vector_metadata)