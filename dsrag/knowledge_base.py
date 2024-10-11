import numpy as np
import os
import time
import json
from typing import Optional, Union, Dict
import concurrent.futures
from dsrag.auto_context import (
    get_document_title,
    get_document_summary,
    get_section_summary,
    get_chunk_header,
    get_segment_header,
)
from dsrag.rse import (
    get_relevance_values,
    get_best_segments,
    get_meta_document,
    RSE_PARAMS_PRESETS,
)
from dsrag.database.vector import Vector, VectorDB, BasicVectorDB
from dsrag.database.vector.types import MetadataFilter
from dsrag.database.chunk import ChunkDB, BasicChunkDB
from dsrag.embedding import Embedding, OpenAIEmbedding
from dsrag.reranker import Reranker, CohereReranker
from dsrag.llm import LLM, OpenAIChatAPI
from dsrag.dsparse.parse_and_chunk import parse_and_chunk_vlm, parse_and_chunk_no_vlm


class KnowledgeBase:
    def __init__(
        self,
        kb_id: str,
        title: str = "",
        supp_id: str = "",
        description: str = "",
        language: str = "en",
        storage_directory: str = "~/dsRAG",
        embedding_model: Optional[Embedding] = None,
        reranker: Optional[Reranker] = None,
        auto_context_model: Optional[LLM] = None,
        vector_db: Optional[VectorDB] = None,
        chunk_db: Optional[ChunkDB] = None,
        exists_ok: bool = True,
        save_metadata_to_disk: bool = True,
    ):
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)

        if save_metadata_to_disk:
            # load the KB if it exists; otherwise, initialize it and save it to disk
            metadata_path = self.get_metadata_path()
            if os.path.exists(metadata_path) and exists_ok:
                self.load(
                    auto_context_model, reranker
                )  # allow the user to override the auto_context_model and reranker
            elif os.path.exists(metadata_path) and not exists_ok:
                raise ValueError(
                    f"Knowledge Base with ID {kb_id} already exists. Use exists_ok=True to load it."
                )
            else:
                created_time = time.time()
                # We don't care about the milliseconds
                created_time = int(created_time)
                self.kb_metadata = {
                    "title": title,
                    "description": description,
                    "language": language,
                    "supp_id": supp_id,
                    "created_on": created_time,
                }
                self.initialize_components(
                    embedding_model, reranker, auto_context_model, vector_db, chunk_db
                )
                self.save()  # save the config for the KB to disk
        else:
            self.kb_metadata = {
                "title": title,
                "description": description,
                "language": language,
                "supp_id": supp_id,
            }
            self.initialize_components(
                embedding_model, reranker, auto_context_model, vector_db, chunk_db
            )

    def get_metadata_path(self):
        return os.path.join(self.storage_directory, "metadata", f"{self.kb_id}.json")

    def initialize_components(
        self,
        embedding_model: Optional[Embedding],
        reranker: Optional[Reranker],
        auto_context_model: Optional[LLM],
        vector_db: Optional[VectorDB],
        chunk_db: Optional[ChunkDB],
    ):
        self.embedding_model = embedding_model if embedding_model else OpenAIEmbedding()
        self.reranker = reranker if reranker else CohereReranker()
        self.auto_context_model = (
            auto_context_model if auto_context_model else OpenAIChatAPI()
        )
        self.vector_db = (
            vector_db
            if vector_db
            else BasicVectorDB(self.kb_id, self.storage_directory)
        )
        self.chunk_db = (
            chunk_db if chunk_db else BasicChunkDB(self.kb_id, self.storage_directory)
        )
        self.vector_dimension = self.embedding_model.dimension

    def save(self):
        # Serialize components
        components = {
            "embedding_model": self.embedding_model.to_dict(),
            "reranker": self.reranker.to_dict(),
            "auto_context_model": self.auto_context_model.to_dict(),
            "vector_db": self.vector_db.to_dict(),
            "chunk_db": self.chunk_db.to_dict(),
        }
        # Combine metadata and components
        full_data = {**self.kb_metadata, "components": components}

        metadata_dir = os.path.join(self.storage_directory, "metadata")
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)

        with open(self.get_metadata_path(), "w") as f:
            json.dump(full_data, f, indent=4)

    def load(self, auto_context_model=None, reranker=None):
        """
        Note: auto_context_model and reranker can be passed in to override the models in the metadata file. The other components are not overridable because that would break things.
        """
        with open(self.get_metadata_path(), "r") as f:
            data = json.load(f)
            self.kb_metadata = {
                key: value for key, value in data.items() if key != "components"
            }
            components = data.get("components", {})
            # Deserialize components
            self.embedding_model = Embedding.from_dict(
                components.get("embedding_model", {})
            )
            self.reranker = (
                reranker
                if reranker
                else Reranker.from_dict(components.get("reranker", {}))
            )
            self.auto_context_model = (
                auto_context_model
                if auto_context_model
                else LLM.from_dict(components.get("auto_context_model", {}))
            )
            self.vector_db = VectorDB.from_dict(components.get("vector_db", {}))
            self.chunk_db = ChunkDB.from_dict(components.get("chunk_db", {}))
            self.vector_dimension = self.embedding_model.dimension

    def delete(self):
        # delete all documents in the KB
        doc_ids_to_delete = self.chunk_db.get_all_doc_ids()
        for doc_id in doc_ids_to_delete:
            self.delete_document(doc_id)

        self.chunk_db.delete()
        self.vector_db.delete()

        # delete the metadata file
        os.remove(self.get_metadata_path())

    def add_document(
        self,
        doc_id: str,
        text: str = "",
        file_path: str = "",
        document_title: str = "",
        auto_context_config: dict = {},
        file_parsing_config: dict = {},
        semantic_sectioning_config: dict = {},
        chunking_config: dict = {},
        supp_id: str = "",
        metadata: dict = {},
    ):
        """
        Inputs:
        - doc_id: unique identifier for the document; a file name or path is a good choice
        - text: the full text of the document
        - file_path: the path to the file to be uploaded. Supported file types are .txt, .md, .pdf, and .docx
        - document_title: the title of the document (if not provided, either the doc_id or an LLM-generated title will be used, depending on the auto_context_config)
        - auto_context_config: a dictionary with configuration parameters for AutoContext
            - use_generated_title: bool - whether to use an LLM-generated title if no title is provided (default is True)
            - document_title_guidance: str - guidance for generating the document title
            - get_document_summary: bool - whether to get a document summary (default is True)
            - document_summarization_guidance: str
            - get_section_summaries: bool - whether to get section summaries (default is False)
            - section_summarization_guidance: str
        - file_parsing_config: a dictionary with configuration parameters for parsing the file (ignored if text is provided instead of a file_path)
            - use_vlm: bool - whether to use VLM (visual language model) for parsing the file (default is False)
            - vlm_config: a dictionary with configuration parameters for VLM (ignored if use_vlm is False)
                - provider: the VLM provider to use - only "vertex_ai" is supported at the moment
                - model: the VLM model to use
                - project_id: the GCP project ID (required if provider is "vertex_ai")
                - location: the GCP location (required if provider is "vertex_ai")
                - save_path: the path to save intermediate files created during VLM processing
                - exclude_elements: a list of element types to exclude from the parsed text. Default is ["Header", "Footer"].
        - semantic_sectioning_config: a dictionary with configuration for the semantic sectioning model (defaults will be used if not provided)
            - llm_provider: the LLM provider to use for semantic sectioning - only "openai" and "anthropic" are supported at the moment
            - model: the LLM model to use for semantic sectioning
            - use_semantic_sectioning: if False, semantic sectioning will be skipped (default is True)
        - chunking_config: a dictionary with configuration for chunking the document/sections into smaller pieces (defaults will be used if not provided)
            - chunk_size: the maximum number of characters to include in each chunk
            - min_length_for_chunking: the minimum length of text to allow chunking (measured in number of characters); if the text is shorter than this, it will be added as a single chunk. If semantic sectioning is used, this parameter will be applied to each section. Setting this to a higher value than the chunk_size can help avoid unnecessary chunking of short documents or sections.
        - supp_id: supplementary ID for the document (Can be any string you like. Useful for filtering documents later on.)
        - metadata: a dictionary of metadata to associate with the document - can use whatever keys you like
        """

        if text == "" and file_path == "":
            raise ValueError("Either text or file_path must be provided")

        # verify that the document does not already exist in the KB - the doc_id should be unique
        if doc_id in self.chunk_db.get_all_doc_ids():
            print(f"Document with ID {doc_id} already exists in the KB. Skipping...")
            return

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

        # AUTOCONTEXT: create contextual chunk headers   

        # document title and summary
        if not document_title and auto_context_config.get("use_generated_title", True):
            document_title_guidance = auto_context_config.get(
                "document_title_guidance", ""
            )
            document_title = get_document_title(
                self.auto_context_model,
                text,
                document_title_guidance=document_title_guidance,
                language=self.kb_metadata["language"]
            )
        elif not document_title:
            document_title = doc_id

        if auto_context_config.get("get_document_summary", True):
            document_summarization_guidance = auto_context_config.get("document_summarization_guidance", "")
            document_summary = get_document_summary(
                self.auto_context_model,
                text,
                document_title=document_title,
                document_summarization_guidance=document_summarization_guidance,
                language=self.kb_metadata["language"]
            )
        else:
            document_summary = ""

        # get section summaries
        for section in sections:
            if auto_context_config.get("get_section_summaries", False):
                section_summarization_guidance = auto_context_config.get("section_summarization_guidance", "")
                section["summary"] = get_section_summary(
                    auto_context_model=self.auto_context_model,
                    section_text=section["content"],
                    document_title=document_title,
                    section_title=section["title"],
                    section_summarization_guidance=section_summarization_guidance,
                    language=self.kb_metadata["language"]
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

        # ADD CHUNKS TO DATABASE
        
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

        # embed the chunks - if the document is long, we need to get the embeddings in chunks
        chunk_embeddings = []
        for i in range(0, len(chunks), 50):
            chunk_embeddings += self.get_embeddings(chunks_to_embed[i : i + 50], input_type="document")

        # add the chunks to the chunk database
        assert len(chunks) == len(chunk_embeddings) == len(chunks_to_embed)
        self.chunk_db.add_document(
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
        self.vector_db.add_vectors(vectors=chunk_embeddings, metadata=vector_metadata)

        self.save()  # save to disk after adding a document

    def delete_document(self, doc_id: str):
        self.chunk_db.remove_document(doc_id)
        self.vector_db.remove_document(doc_id)

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        return self.chunk_db.get_chunk_text(doc_id, chunk_index)
    
    def get_chunk_content(self, doc_id: str, chunk_index: int) -> tuple[str, str]:
        return self.chunk_db.get_chunk_text(doc_id, chunk_index)

    def get_segment_header(self, doc_id: str, chunk_index: int) -> str:
        document_title = self.chunk_db.get_document_title(doc_id, chunk_index) or ""
        document_summary = self.chunk_db.get_document_summary(doc_id, chunk_index) or ""
        return get_segment_header(
            document_title=document_title, document_summary=document_summary
        )

    def get_embeddings(self, text: list[str], input_type: str = "") -> list[Vector]:
        return self.embedding_model.get_embeddings(text, input_type)

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2)  # since the embeddings are normalized

    def search(self, query: str, top_k: int, metadata_filter: Optional[MetadataFilter] = None) -> list:
        """
        Get top k most relevant chunks for a given query. This is where we interface with the vector database.
        - returns a list of dictionaries, where each dictionary has the following keys: `metadata` (which contains 'doc_id', 'chunk_index', 'chunk_text', and 'chunk_header') and `similarity`
        """
        query_vector = self.get_embeddings(
            [query], input_type="query"
        )[0]  # embed the query, and access the first element of the list since the query is a single string
        search_results = self.vector_db.search(
            query_vector, top_k, metadata_filter
        )  # do a vector database search
        if len(search_results) == 0:
            return []
        search_results = self.reranker.rerank_search_results(
            query, search_results
        )  # rerank search results using a reranker
        return search_results

    def get_all_ranked_results(self, search_queries: list[str], metadata_filter: Optional[MetadataFilter] = None):
        """
        - search_queries: list of search queries
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.search, query, 200, metadata_filter) for query in search_queries]
            
            all_ranked_results = []
            for future in futures:
                ranked_results = future.result()
                all_ranked_results.append(ranked_results)
        
        return all_ranked_results
    
    def get_segment_page_numbers(self, doc_id: str, chunk_start: int, chunk_end: int) -> tuple:
        start_page_number, _ = self.chunk_db.get_chunk_page_numbers(doc_id, chunk_start)
        _, end_page_number = self.chunk_db.get_chunk_page_numbers(doc_id, chunk_end - 1)
        return start_page_number, end_page_number

    def get_segment_text_from_database(self, doc_id: str, chunk_start: int, chunk_end: int) -> str:
        segment = f"{self.get_segment_header(doc_id=doc_id, chunk_index=chunk_start)}\n\n"  # initialize the segment with the segment header
        for chunk_index in range(chunk_start, chunk_end):  # end index is non-inclusive
            chunk_text = self.get_chunk_text(doc_id, chunk_index) or ""
            segment += chunk_text
        return segment.strip()
    
    def get_segment_content_from_database(self, doc_id: str, chunk_start: int, chunk_end: int) -> list[dict]:
        """
        Instead of returning a single string, like get_segment_text_from_database, this function returns a list of dictionaries, which enables
        the return of image paths when available. For adjacent chunks that are text, the text is concatenated into a single dictionary. If there 
        are no images in the segment, then the list will just contain a single dictionary with the type "text" and the content as the entire segment text.
        The segment header is treated just like any other text chunk and will be combined with the first chunk if it is text.

        Returns a list of dictionaries, where each dictionary has the following keys: `type` and `content`
        - `type` is either "text" or "image"
        - `content` is the text content (if 'type' is "text") or image path (if 'type' is "image")
        """
        segment = []

        # Retrieve and add the segment header as the initial text content
        header_text = self.get_segment_header(doc_id=doc_id, chunk_index=chunk_start)
        if header_text:
            segment.append({
                "type": "text",
                "content": header_text
            })

        for chunk_index in range(chunk_start, chunk_end):  # end index is non-inclusive
            chunk_text, image_path = self.get_chunk_content(doc_id, chunk_index)
            if image_path:
                segment.append({
                    "type": "image",
                    "content": image_path
                })
            else:
                # If current chunk is text, concatenate it with the previous text chunk if possible
                if segment and segment[-1]["type"] == "text":
                    # Concatenate the text
                    segment[-1]["content"] += chunk_text
                else:
                    # Append as a new text dictionary
                    segment.append({
                        "type": "text",
                        "content": chunk_text
                    })

        return segment

    def query(
        self,
        search_queries: list[str],
        rse_params: Union[Dict, str] = "balanced",
        latency_profiling: bool = False,
        metadata_filter: Optional[MetadataFilter] = None,
        return_images: bool = False, # whether to return image paths when available
    ) -> list[dict]:
        """
        Inputs:
        - search_queries: list of search queries
        - rse_params: dictionary containing the following parameters:
            - max_length: maximum length of a segment, measured in number of chunks
            - overall_max_length: maximum length of all segments combined, measured in number of chunks
            - minimum_value: minimum value of a segment, measured in relevance value
            - irrelevant_chunk_penalty: float between 0 and 1
            - overall_max_length_extension: the maximum length of all segments combined will be increased by this amount for each additional query beyond the first
            - decay_rate
            - top_k_for_document_selection: the number of documents to consider

        Returns relevant_segment_info, a list of segment_info dictionaries, ordered by relevance, that each contain:
        - doc_id: the document ID of the document that the segment is from
        - chunk_start: the start index of the segment in the document
        - chunk_end: the (non-inclusive) end index of the segment in the document
        - text OR content: 
            if return_images is False, text: a string that contains the full text of the segment
            if return_images is True, content: a list of dictionaries, where each dictionary has the following keys: `type` and `content`
                - `type` is either "text" or "image"
                - `content` is the text content (if 'type' is "text") or image path (if 'type' is "image")
        """
        # check if the rse_params is a preset name and convert it to a dictionary if it is
        if isinstance(rse_params, str) and rse_params in RSE_PARAMS_PRESETS:
            rse_params = RSE_PARAMS_PRESETS[rse_params]
        elif isinstance(rse_params, str):
            raise ValueError(f"Invalid rse_params preset name: {rse_params}")

        # set the RSE parameters
        default_rse_params = RSE_PARAMS_PRESETS[
            "balanced"
        ]  # use the 'balanced' preset as the default for any missing parameters
        max_length = rse_params.get("max_length", default_rse_params["max_length"])
        overall_max_length = rse_params.get(
            "overall_max_length", default_rse_params["overall_max_length"]
        )
        minimum_value = rse_params.get(
            "minimum_value", default_rse_params["minimum_value"]
        )
        irrelevant_chunk_penalty = rse_params.get(
            "irrelevant_chunk_penalty", default_rse_params["irrelevant_chunk_penalty"]
        )
        overall_max_length_extension = rse_params.get(
            "overall_max_length_extension",
            default_rse_params["overall_max_length_extension"],
        )
        decay_rate = rse_params.get("decay_rate", default_rse_params["decay_rate"])
        top_k_for_document_selection = rse_params.get(
            "top_k_for_document_selection",
            default_rse_params["top_k_for_document_selection"],
        )
        chunk_length_adjustment = rse_params.get(
            "chunk_length_adjustment", default_rse_params["chunk_length_adjustment"]
        )

        overall_max_length += (
            len(search_queries) - 1
        ) * overall_max_length_extension  # increase the overall max length for each additional query

        start_time = time.time()
        all_ranked_results = self.get_all_ranked_results(search_queries=search_queries, metadata_filter=metadata_filter)
        if latency_profiling:
            print(
                f"get_all_ranked_results took {time.time() - start_time} seconds to run for {len(search_queries)} queries"
            )

        document_splits, document_start_points, unique_document_ids = get_meta_document(
            all_ranked_results=all_ranked_results,
            top_k_for_document_selection=top_k_for_document_selection,
        )

        # verify that we have a valid meta-document - otherwise return an empty list of segments
        if len(document_splits) == 0:
            return []

        # get the length of the meta-document so we don't have to pass in the whole list of splits
        meta_document_length = document_splits[-1]

        # get the relevance values for each chunk in the meta-document and use those to find the best segments
        all_relevance_values = get_relevance_values(
            all_ranked_results=all_ranked_results,
            meta_document_length=meta_document_length,
            document_start_points=document_start_points,
            unique_document_ids=unique_document_ids,
            irrelevant_chunk_penalty=irrelevant_chunk_penalty,
            decay_rate=decay_rate,
            chunk_length_adjustment=chunk_length_adjustment,
        )
        best_segments, scores = get_best_segments(
            all_relevance_values=all_relevance_values,
            document_splits=document_splits,
            max_length=max_length,
            overall_max_length=overall_max_length,
            minimum_value=minimum_value,
        )

        # convert the best segments into a list of dictionaries that contain the document id and the start and end of the chunk
        relevant_segment_info = []
        for segment_index, (start, end) in enumerate(best_segments):
            # find the document that this segment starts in
            for i, split in enumerate(document_splits):
                if start < split:  # splits represent the end of each document
                    doc_start = document_splits[i - 1] if i > 0 else 0
                    relevant_segment_info.append(
                        {
                            "doc_id": unique_document_ids[i],
                            "chunk_start": start - doc_start,
                            "chunk_end": end - doc_start,
                        }
                    )  # NOTE: end index is non-inclusive
                    break

            score = scores[segment_index]
            relevant_segment_info[-1]["score"] = score

        if return_images:
            # retrieve the text and image paths (including segment header) for each of the segments
            for segment_info in relevant_segment_info:
                segment_info["content"] = self.get_segment_content_from_database(
                    segment_info["doc_id"],
                    segment_info["chunk_start"],
                    segment_info["chunk_end"],
                )
                start_page_number, end_page_number = self.get_segment_page_numbers(
                    segment_info["doc_id"],
                    segment_info["chunk_start"],
                    segment_info["chunk_end"]
                )
                segment_info["chunk_page_start"] = start_page_number
                segment_info["chunk_page_end"] = end_page_number
        else:
            # retrieve the text (including segment header) for each of the segments
            for segment_info in relevant_segment_info:
                segment_info["text"] = self.get_segment_text_from_database(
                    segment_info["doc_id"],
                    segment_info["chunk_start"],
                    segment_info["chunk_end"],
                )
                start_page_number, end_page_number = self.get_segment_page_numbers(
                    segment_info["doc_id"],
                    segment_info["chunk_start"],
                    segment_info["chunk_end"]
                )
                segment_info["chunk_page_start"] = start_page_number
                segment_info["chunk_page_end"] = end_page_number

        return relevant_segment_info