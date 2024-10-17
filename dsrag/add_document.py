"""
Split up the add_document function into smaller functions in a separate file
"""

from dsrag.dsparse.parse_and_chunk import parse_and_chunk_vlm, parse_and_chunk_no_vlm
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

def parse_and_chunk(file_path: str, text: str, file_parsing_config: dict, semantic_sectioning_config: dict, chunking_config: dict):
    # file parsing, semantic sectioning, and chunking
    use_vlm = file_parsing_config.get("use_vlm", False)
    if use_vlm:
        # make sure a file_path is provided
        if not file_path:
            raise ValueError("VLM parsing requires a file_path, not text. Please provide a file_path instead.")
        vlm_config = file_parsing_config.get("vlm_config", {})
        sections, chunks = parse_and_chunk_vlm(
            file_path=file_path,
            vlm_config=vlm_config,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config,
        )
    else:
        if file_path:
            sections, chunks = parse_and_chunk_no_vlm(
                semantic_sectioning_config=semantic_sectioning_config,
                chunking_config=chunking_config,
                file_path=file_path,
            )
        else:
            sections, chunks = parse_and_chunk_no_vlm(
                semantic_sectioning_config=semantic_sectioning_config,
                chunking_config=chunking_config,
                text=text,
            )

    return sections, chunks

def auto_context(auto_context_model: LLM, sections, chunks, text, doc_id, document_title, auto_context_config, language):
    # document title and summary
    if not document_title and auto_context_config.get("use_generated_title", True):
        document_title_guidance = auto_context_config.get(
            "document_title_guidance", ""
        )
        document_title = get_document_title(
            auto_context_model,
            text,
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

    # get section summaries
    for section in sections:
        if auto_context_config.get("get_section_summaries", False):
            section_summarization_guidance = auto_context_config.get("section_summarization_guidance", "")
            section["summary"] = get_section_summary(
                auto_context_model=auto_context_model,
                section_text=section["content"],
                document_title=document_title,
                section_title=section["title"],
                section_summarization_guidance=section_summarization_guidance,
                language=language
            )
        else:
            section["summary"] = ""

    # add document title, document summary, and section summaries to the chunks
    for chunk in chunks:
        chunk["document_title"] = document_title
        chunk["document_summary"] = document_summary
        section_index = chunk["section_index"]
        if section_index is not None:
            chunk["section_title"] = sections[section_index]["title"]
            chunk["section_summary"] = sections[section_index]["summary"]

    print(f"Adding {len(chunks)} chunks to the database")

    # prepare the chunks for embedding by prepending the chunk headers
    chunks_to_embed = []
    for i, chunk in enumerate(chunks):
        chunk_header = get_chunk_header(
            document_title=chunk["document_title"],
            document_summary=chunk["document_summary"],
            section_title=chunk["section_title"],
            section_summary=chunk["section_summary"],
        )
        chunk_to_embed = f"{chunk_header}\n\n{chunk['content']}"
        chunks_to_embed.append(chunk_to_embed)

    return chunks, chunks_to_embed

def get_embeddings(embedding_model: Embedding, chunks_to_embed):
    # embed the chunks - if the document is long, we need to get the embeddings in chunks
    chunk_embeddings = []
    for i in range(0, len(chunks_to_embed), 50):
        chunk_embeddings += embedding_model.get_embeddings(chunks_to_embed[i:i+50], input_type="document")

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
                "image_path": chunk.get("image_path", None),
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
        vector_metadata.append(
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_text": chunk["content"],
                "chunk_header": get_chunk_header(
                    document_title=chunk["document_title"],
                    document_summary=chunk["document_summary"],
                    section_title=chunk["section_title"],
                    section_summary=chunk["section_summary"],
                ),
                "chunk_page_start": chunk.get("page_start", ""),
                "chunk_page_end": chunk.get("page_end", ""),
                "image_path": chunk.get("image_path", ""), # not sure if this needs to be added
                # Add the rest of the metadata to the vector metadata
                **metadata
            }
        )

    # add the vectors and metadata to the vector database
    vector_db.add_vectors(vectors=chunk_embeddings, metadata=vector_metadata)