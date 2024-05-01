import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
import json
from sprag.auto_context import get_document_context, get_chunk_header
from sprag.rse import get_relevance_values, get_best_segments, get_meta_document
from sprag.vector_db import VectorDB, BasicVectorDB
from sprag.chunk_db import ChunkDB, BasicChunkDB
from sprag.embedding import Embedding, OpenAIEmbedding
from sprag.reranker import Reranker, CohereReranker
from sprag.llm import LLM, AnthropicChatAPI


class KnowledgeBase:
    def __init__(self, kb_id: str, title: str = "", description: str = "", language: str = "en", storage_directory: str = '~/spRAG', embedding_model: Embedding = None, reranker: Reranker = None, auto_context_model: LLM = None, vector_db: VectorDB = None, chunk_db: ChunkDB = None, exists_ok: bool = True):
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)

        # load the KB if it exists; otherwise, initialize it and save it to disk
        metadata_path = self.get_metadata_path()
        if os.path.exists(metadata_path) and exists_ok:
            self.load()
        elif os.path.exists(metadata_path) and not exists_ok:
            raise ValueError(f"Knowledge Base with ID {kb_id} already exists. Use exists_ok=True to load it.")
        else:
            self.kb_metadata = {
                'title': title,
                'description': description,
                'language': language,
                'chunk_size': 800, # max number of characters in a chunk - should be replaced with a Chunking class for more flexibility
            }
            self.initialize_components(embedding_model, reranker, auto_context_model, vector_db, chunk_db)
            self.save() # save the config for the KB to disk

    def get_metadata_path(self):
        return os.path.join(self.storage_directory, 'metadata', f'{self.kb_id}.json')

    def initialize_components(self, embedding_model, reranker, auto_context_model, vector_db, chunk_db):
        self.embedding_model = embedding_model if embedding_model else OpenAIEmbedding()
        self.reranker = reranker if reranker else CohereReranker()
        self.auto_context_model = auto_context_model if auto_context_model else AnthropicChatAPI()
        self.vector_db = vector_db if vector_db else BasicVectorDB(self.kb_id, self.storage_directory)
        self.chunk_db = chunk_db if chunk_db else BasicChunkDB(self.kb_id, self.storage_directory)
        self.vector_dimension = self.embedding_model.dimension

    def save(self):
        # Serialize components
        components = {
            'embedding_model': self.embedding_model.to_dict(),
            'reranker': self.reranker.to_dict(),
            'auto_context_model': self.auto_context_model.to_dict(),
            'vector_db': self.vector_db.to_dict(),
            'chunk_db': self.chunk_db.to_dict()
        }
        # Combine metadata and components
        full_data = {**self.kb_metadata, 'components': components}

        metadata_dir = os.path.join(self.storage_directory, 'metadata')
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)

        with open(self.get_metadata_path(), 'w') as f:
            json.dump(full_data, f, indent=4)

    def load(self):
        with open(self.get_metadata_path(), 'r') as f:
            data = json.load(f)
            self.kb_metadata = {key: value for key, value in data.items() if key != 'components'}
            components = data.get('components', {})
            # Deserialize components
            self.embedding_model = Embedding.from_dict(components.get('embedding_model', {}))
            self.reranker = Reranker.from_dict(components.get('reranker', {}))
            self.auto_context_model = LLM.from_dict(components.get('auto_context_model', {}))
            self.vector_db = VectorDB.from_dict(components.get('vector_db', {}))
            self.chunk_db = ChunkDB.from_dict(components.get('chunk_db', {}))
            self.vector_dimension = self.embedding_model.dimension

    def delete(self):
        # delete all documents in the KB
        doc_ids_to_delete = self.chunk_db.get_all_doc_ids()
        for doc_id in doc_ids_to_delete:
            self.delete_document(doc_id)

        # delete the metadata file
        os.remove(self.get_metadata_path())

    def add_document(self, doc_id: str, text: str, auto_context: bool = True, chunk_header: str = None, auto_context_guidance: str = ""):
        # verify that only one of auto_context and chunk_header is set
        try:
            assert auto_context != (chunk_header is not None)
        except:
            print ("Error in add_document: only one of auto_context and chunk_header can be set")

        # verify that the document does not already exist in the KB
        if doc_id in self.chunk_db.get_all_doc_ids():
            print (f"Document with ID {doc_id} already exists in the KB. Skipping...")
            return
        
        # AutoContext
        if auto_context:
            document_context = get_document_context(self.auto_context_model, text, document_title=doc_id, auto_context_guidance=auto_context_guidance)
            chunk_header = get_chunk_header(file_name=doc_id, document_context=document_context)
        elif chunk_header:
            pass
        else:
            chunk_header = ""

        chunks = self.split_into_chunks(text)
        print (f'Adding {len(chunks)} chunks to the database')

        # add chunk headers to the chunks before embedding them
        chunks_to_embed = []
        for i, chunk in enumerate(chunks):
            chunk_to_embed = f'[{chunk_header}]\n{chunk}'
            chunks_to_embed.append(chunk_to_embed)

        # embed the chunks
        if len(chunks) <= 50:
            # if the document is short, we can get all the embeddings at once
            chunk_embeddings = self.get_embeddings(chunks_to_embed, input_type="document")
        else:
            # if the document is long, we need to get the embeddings in chunks
            chunk_embeddings = []
            for i in range(0, len(chunks), 50):
                chunk_embeddings += self.get_embeddings(chunks_to_embed[i:i+50], input_type="document")

        assert len(chunks) == len(chunk_embeddings) == len(chunks_to_embed)
        self.chunk_db.add_document(doc_id, {i: {'chunk_text': chunk, 'chunk_header': chunk_header} for i, chunk in enumerate(chunks)})

        # create metadata list
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({'doc_id': doc_id, 'chunk_index': i, 'chunk_header': chunk_header, 'chunk_text': chunk})

        # add the vectors and metadata to the vector database
        self.vector_db.add_vectors(vectors=chunk_embeddings, metadata=metadata)

        self.save() # save the database to disk after adding a document

    def delete_document(self, doc_id: str):
        self.chunk_db.remove_document(doc_id)
        self.vector_db.remove_document(doc_id)

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> str:
        return self.chunk_db.get_chunk_text(doc_id, chunk_index)
    
    def get_chunk_header(self, doc_id: str, chunk_index: int) -> str:
        return self.chunk_db.get_chunk_header(doc_id, chunk_index)

    def get_embeddings(self, text: str or list[str], input_type: str = ""):
        return self.embedding_model.get_embeddings(text, input_type)
    
    def split_into_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.kb_metadata['chunk_size'], chunk_overlap = 0, length_function = len)
        texts = text_splitter.create_documents([text])
        chunks = [text.page_content for text in texts]
        return chunks

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) # since the embeddings are normalized

    def search(self, query: str, top_k: int) -> list:
        """
        Get top k most relevant chunks for a given query. This is where we interface with the vector database.
        - returns a list of dictionaries, where each dictionary has the following keys: `metadata` (which contains 'doc_id', 'chunk_index', 'chunk_text', and 'chunk_header') and `similarity`
        """
        query_vector = self.get_embeddings(query, input_type="query") # embed the query
        search_results = self.vector_db.search(query_vector, top_k) # do a vector database search
        search_results = self.reranker.rerank_search_results(query, search_results) # rerank search results using a reranker
        return search_results
    
    def get_all_ranked_results(self, search_queries: list[str]):
        """
        - search_queries: list of search queries
        """
        all_ranked_results = []
        for query in search_queries:
            ranked_results = self.search(query, 200)
            all_ranked_results.append(ranked_results)
        return all_ranked_results
    
    def get_segment_text_from_database(self, doc_id: str, chunk_start: int, chunk_end: int) -> str:
        segment = f"[{self.get_chunk_header(doc_id, chunk_start)}]\n" # initialize the segment with the chunk header
        for chunk_index in range(chunk_start, chunk_end): # NOTE: end index is non-inclusive
            chunk_text = self.get_chunk_text(doc_id, chunk_index)
            segment += chunk_text
        return segment.strip()
    
    def query(self, search_queries: list[str], rse_params: dict = {}, latency_profiling: bool = False) -> list[dict]:
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
        - text: the full text of the segment
        """

        default_rse_params = {
            'max_length': 10,
            'overall_max_length': 20,
            'minimum_value': 0.7,
            'irrelevant_chunk_penalty': 0.2,
            'overall_max_length_extension': 5,
            'decay_rate': 20,
            'top_k_for_document_selection': 7
        }

        # set the RSE parameters
        max_length = rse_params.get('max_length', default_rse_params['max_length'])
        overall_max_length = rse_params.get('overall_max_length', default_rse_params['overall_max_length'])
        minimum_value = rse_params.get('minimum_value', default_rse_params['minimum_value'])
        irrelevant_chunk_penalty = rse_params.get('irrelevant_chunk_penalty', default_rse_params['irrelevant_chunk_penalty'])
        overall_max_length_extension = rse_params.get('overall_max_length_extension', default_rse_params['overall_max_length_extension'])
        decay_rate = rse_params.get('decay_rate', default_rse_params['decay_rate'])
        top_k_for_document_selection = rse_params.get('top_k_for_document_selection', default_rse_params['top_k_for_document_selection'])

        overall_max_length += (len(search_queries) - 1) * overall_max_length_extension # increase the overall max length for each additional query

        start_time = time.time()
        all_ranked_results = self.get_all_ranked_results(search_queries=search_queries)
        if latency_profiling:
            print(f"get_all_ranked_results took {time.time() - start_time} seconds to run for {len(search_queries)} queries")

        document_splits, document_start_points, unique_document_ids = get_meta_document(all_ranked_results=all_ranked_results, top_k_for_document_selection=top_k_for_document_selection)

        # verify that we have a valid meta-document - otherwise return an empty list of segments
        if len(document_splits) == 0:
            return []
        
        # get the length of the meta-document so we don't have to pass in the whole list of splits
        meta_document_length = document_splits[-1]

        # get the relevance values for each chunk in the meta-document and use those to find the best segments
        all_relevance_values = get_relevance_values(all_ranked_results=all_ranked_results, meta_document_length=meta_document_length, document_start_points=document_start_points, unique_document_ids=unique_document_ids, irrelevant_chunk_penalty=irrelevant_chunk_penalty, decay_rate=decay_rate)
        best_segments = get_best_segments(all_relevance_values=all_relevance_values, document_splits=document_splits, max_length=max_length, overall_max_length=overall_max_length, minimum_value=minimum_value)
        
        # convert the best segments into a list of dictionaries that contain the document id and the start and end of the chunk
        relevant_segment_info = []
        for start, end in best_segments:
            # find the document that this segment starts in
            for i, split in enumerate(document_splits):
                if start < split: # splits represent the end of each document
                    doc_start = document_splits[i-1] if i > 0 else 0
                    relevant_segment_info.append({"doc_id": unique_document_ids[i], "chunk_start": start - doc_start, "chunk_end": end - doc_start}) # NOTE: end index is non-inclusive
                    break
        
        # retrieve the actual text for the segments from the database
        for segment_info in relevant_segment_info:
            segment_info["text"] = (self.get_segment_text_from_database(segment_info["doc_id"], segment_info["chunk_start"], segment_info["chunk_end"])) # NOTE: this is where the chunk header is added to the segment text

        return relevant_segment_info