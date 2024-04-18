import numpy as np
from auto_context import get_document_context, get_chunk_header
from document_parsing_utils import extract_text_from_pdf, extract_text_from_docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import pickle
import os
import time
from embeddings import get_embeddings, dimensionality
from reranker import rerank_search_results
from rse import get_relevance_values, get_best_segments, get_meta_document
from database import VectorDB, BasicVectorDB


def truncate_content(content: str, max_tokens: int):
    TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)

class KnowledgeBase:
    def __init__(self, kb_id: str, title: str = "", description: str = "", language: str = "en", embedding_model: str = "text-embedding-3-small-768", vector_db: VectorDB = None, storage_directory: str = '~/spRAG'):
        self.kb_id = kb_id
        self.database = {} # store text chunks and chunk headers
        self.vector_db = vector_db # to store embeddings
        self.metadata = {} # to store title, description, etc.
        self.chunk_size = 800 # max number of characters in a chunk
        self.storage_directory = f'{storage_directory}/knowledge_bases/'

        if self.vector_db is None:
            self.vector_db = BasicVectorDB(kb_id, self.storage_directory)

        # load the database from disk if it exists
        if os.path.exists(f'{self.storage_directory}{self.kb_id}_metadata.pkl'):
            self.load()
        else:
            self.metadata['title'] = title
            self.metadata['description'] = description
            self.metadata['language'] = language
            self.metadata['embedding_model'] = embedding_model
            self.metadata['embedding_dimensions'] = dimensionality(embedding_model)
            self.save() # save the metadata

    def save(self):
        with open(f'{self.storage_directory}{self.kb_id}_database.pkl', 'wb') as f:
            pickle.dump(self.database, f)
        with open(f'{self.storage_directory}{self.kb_id}_metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self):
        with open(f'{self.storage_directory}{self.kb_id}_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        try:
            with open(f'{self.storage_directory}{self.kb_id}_database.pkl', 'rb') as f:
                self.database = pickle.load(f)
        except:
            self.database = {}

    def delete(self, retain_metadata: bool = False):
        # delete all documents in the KB so they get removed from the doc_id_to_kb_id mapping
        doc_ids_to_delete = list(self.database.keys())
        for doc_id in doc_ids_to_delete:
            self.delete_document(doc_id)

        # delete the database and metadata files
        os.remove(f'{self.storage_directory}{self.kb_id}_database.pkl')

        if not retain_metadata:
            os.remove(f'{self.storage_directory}{self.kb_id}_metadata.pkl')

    def add_document(self, doc_id: str, text: str, auto_context: bool = True, chunk_header: str = None, auto_context_guidance: str = ""):
        # verify that only one of auto_context and chunk_header is set
        try:
            assert auto_context != (chunk_header is not None)
        except:
            print ("Error in add_document: only one of auto_context and chunk_header can be set")

        # verify that the document doesn't already exist in this KB
        with open(f'{self.storage_directory}doc_id_to_kb_id.pkl', 'rb') as f:
            mapping = pickle.load(f)

        if doc_id in mapping:
            if mapping[doc_id] == self.kb_id:
                # document already exists in this KB - skip it
                print (f'Document with id {doc_id} already exists in this KB')
                return
        
        # AutoContext
        if auto_context:
            auto_context_model_name = "gpt-3.5-turbo-0125" # "mistral-tiny" "gpt-3.5-turbo-0125"
            document_context = get_document_context(text, document_title=doc_id, auto_context_guidance=auto_context_guidance, model_name=auto_context_model_name)
            chunk_header = get_chunk_header(file_name=doc_id, document_context=document_context)
        elif chunk_header:
            pass
        else:
            chunk_header = ""

        chunks = self.split_into_chunks(text)
        print (f'Adding {len(chunks)} chunks to the database')

        # add chunk headers and section summaries to the chunks before embedding them
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
        self.database[doc_id] = {i: {'text': chunk, 'vector': chunk_embeddings[i], 'chunk_header': chunk_header} for i, chunk in enumerate(chunks)}

        self.save() # save the database to disk after adding a document

    def delete_document(self, doc_id: str):
        del self.database[doc_id]
        self.save() # save the database to disk after deleting a document

        # delete the mapping from doc_id to kb_id
        with open(f'{self.storage_directory}doc_id_to_kb_id.pkl', 'rb') as f:
            mapping = pickle.load(f)
        del mapping[doc_id]
        with open(f'{self.storage_directory}doc_id_to_kb_id.pkl', 'wb') as f:
            pickle.dump(mapping, f)

    def get_document(self, doc_id: str) -> str:
        if doc_id in self.database:
            return self.database[doc_id]
        return None

    def get_chunk(self, doc_id: str, chunk_index: int) -> str:
        if doc_id in self.database and chunk_index in self.database[doc_id]:
            return self.database[doc_id][chunk_index]['text']
        return None
    
    def get_chunk_header(self, doc_id: str, chunk_index: int) -> str:
        if doc_id in self.database and chunk_index in self.database[doc_id]:
            return self.database[doc_id][chunk_index]['chunk_header']
        return None

    def get_embeddings(self, text: str or list[str], input_type: str = ""):
        model = self.metadata['embedding_model']
        return get_embeddings(text, model, input_type)
    
    def split_into_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = 0, length_function = len)
        texts = text_splitter.create_documents([text])
        chunks = [text.page_content for text in texts]
        return chunks

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) # since the embeddings are normalized

    def search(self, query: str, top_k: int, use_reranker: bool = True) -> list:
        """
        Get top k most relevant chunks for a given query. This is where we interface with the vector database.
        - returns a list of dictionaries, where each dictionary has the following keys: `metadata` (which contains 'doc_id' and 'chunk_index'), `similarity`, and `content`
        """
        query_vector = self.get_embeddings(query, input_type="query")

        # do a brute force search
        similarities = []
        for doc_id, doc in self.database.items():
            for chunk_index, chunk in doc.items():
                vector = chunk['vector']
                chunk_header = chunk['chunk_header']
                sim = self.cosine_similarity(query_vector, vector)
                similarities.append({'metadata': {'doc_id': doc_id, 'chunk_index': chunk_index, 'chunk_header': chunk_header}, 'similarity': sim, 'content': chunk['text']})

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        search_results = similarities[:top_k]

        if use_reranker:
            search_results = rerank_search_results(query, search_results)
        
        return search_results
    
    def get_all_ranked_results(self, search_queries: list[str]):
        """
        - search_queries: list of search queries
        """
        all_ranked_results = []
        for query in search_queries:
            ranked_results = self.search(query, 100, use_reranker=True)
            all_ranked_results.append(ranked_results)
        return all_ranked_results
    
    def get_segment_text_from_database(self, doc_id: str, chunk_start: int, chunk_end: int) -> str:
        segment = f"[{self.get_chunk_header(doc_id, chunk_start)}]\n" # initialize the segment with the chunk header
        for chunk_index in range(chunk_start, chunk_end): # NOTE: end index is non-inclusive
            chunk_text = self.get_chunk(doc_id, chunk_index)
            segment += chunk_text
        return segment.strip()
    
    def query(self, search_queries: list[str], max_length: int = 10, overall_max_length: int = 20, minimum_value: float = 0.7, irrelevant_chunk_penalty: float = 0.2, overall_max_length_extension: int = 5, decay_rate: int = 20, top_k_for_document_selection: int = 7, latency_profiling: bool = False) -> list[dict]:
        """
        Inputs:
        - search_queries: list of search queries
        - max_length: maximum length of a segment, measured in number of chunks
        - overall_max_length: maximum length of all segments combined, measured in number of chunks
        - minimum_value: minimum value of a segment, measured in relevance value
        - irrelevant_chunk_penalty: float between 0 and 1
        - overall_max_length_extension: the maximum length of all segments combined will be increased by this amount for each additional query beyond the first

        Returns relevant_segment_info, a list of segment_info dictionaries, ordered by relevance, that each contain:
        - doc_id: the document ID of the document that the segment is from
        - chunk_start: the start index of the segment in the document
        - chunk_end: the (non-inclusive) end index of the segment in the document
        - text: the full text of the segment
        """

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
    
    def get_kb_info(self):
        kb_info = {
            'kb_id': self.kb_id,
            'language': self.metadata['language'],
            'title': self.metadata['title'],
            'description': self.metadata['description'],
        }
        return kb_info


def create_kb_from_directory(kb_id: str, directory: str, title: str = None, description: str = "", language: str = 'en', auto_context: bool = True, micro_context: bool = False, embedding_model: str = 'cohere-english', use_unstructured: bool = False, use_spdb: bool = False, auto_context_guidance: str = ""):
    """
    - kb_id is the name of the knowledge base
    - directory is the absolute path to the directory containing the documents
    - no support for manually defined chunk headers here, because they would have to be defined for each file in the directory

    Supported file types: .docx, .md, .txt, .pdf
    """
    if not title:
        title = kb_id
    
    # create a new KB
    kb = KnowledgeBase(kb_id, title=title, description=description, language=language, embedding_model=embedding_model)

    # verify that the new KB doesn't already exist by making sure kb.database is an empty dictionary
    if kb.database:
        print (f'KB with id {kb_id} already exists. No documents were added.')
        return

    # add documents
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(('.docx', '.md', '.txt', '.pdf')):
                try:
                    file_path = os.path.join(root, file_name)
                    clean_file_path = file_path.replace(directory, "")
                    
                    if file_name.endswith('.docx'):
                        text = extract_text_from_docx(file_path)
                    elif file_name.endswith('.pdf') and not use_unstructured:
                        text = extract_text_from_pdf(file_path)
                    elif file_name.endswith('.md') or file_name.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            text = f.read()

                    kb.add_document(clean_file_path, text, auto_context=auto_context, auto_context_guidance=auto_context_guidance)
                    time.sleep(1) # pause for 1 second to avoid hitting API rate limits
                except:
                    print (f"Error reading {file_name}")
                    continue
            else:
                print (f"Unsupported file type: {file_name}")
                continue
    
    return kb

def create_kb_from_file(kb_id: str, file_path: str, title: str = None, description: str = "", language: str = 'en', auto_context: bool = True, micro_context: bool = False, embedding_model: str = 'cohere-english', use_unstructured: bool = False, use_spdb: bool = False, auto_context_guidance: str = ""):
    """
    - kb_id is the name of the knowledge base
    - file_path is the absolute path to the file containing the documents

    Supported file types: .docx, .md, .txt, .pdf
    """
    if not title:
        title = kb_id
    
    # create a new KB
    kb = KnowledgeBase(kb_id, title=title, description=description, language=language, embedding_model=embedding_model)

    # verify that the new KB doesn't already exist by making sure kb.database is an empty dictionary
    if kb.database:
        print (f'KB with id {kb_id} already exists. No documents were added.')
        return
    
    print (f'Creating KB with id {kb_id}...')

    file_name = os.path.basename(file_path)

    # add document
    if file_path.endswith(('.docx', '.md', '.txt', '.pdf')):
        # define clean file path as just the file name here since we're not using a directory
        clean_file_path = file_name
        
        if file_path.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_name.endswith('.pdf') and not use_unstructured:
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith('.md') or file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()

        kb.add_document(clean_file_path, text, auto_context=auto_context, auto_context_guidance=auto_context_guidance)
    else:
        print (f"Unsupported file type: {file_name}")
        return
    
    return kb

def load_kb(kb_id: str, storage_directory: str = '~/spRAG/knowledge_bases/'):
    """
    Load a knowledge base
    """
    kb = KnowledgeBase(kb_id)
    return kb

if __name__ == "__main__":
    kb_id = "bvp_cloud_no_ss"
    kb = KnowledgeBase(kb_id)
    documents = list(kb.database.keys())
    print (documents)