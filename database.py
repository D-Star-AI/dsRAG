from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import pickle
import os


class VectorDB(ABC):
    @abstractmethod
    def add_vectors(self, vector, metadata):
        """
        Store a list of vectors with associated metadata.
        """
        pass

    @abstractmethod
    def remove_document(self, doc_id):
        """
        Remove all vectors and metadata associated with a given document ID.
        """
        pass

    @abstractmethod
    def search(self, query_vector, top_k=10):
        """
        Retrieve the top-k closest vectors to a given query vector.
        - needs to return results as list of dictionaries in this format: 
        {
            'metadata': 
                {
                    'doc_id': doc_id, 
                    'chunk_index': chunk_index, 
                    'chunk_header': chunk_header,
                    'chunk_text': chunk_text
                }, 
            'similarity': similarity, 
        }
        """
        pass


class BasicVectorDB(VectorDB):
    def __init__(self, kb_id: str, storage_directory: str = '~/spRAG'):
        self.storage_path = f'{storage_directory}/vector_storage/{kb_id}.pkl'
        self.load()

    def add_vectors(self, vectors, metadata):
        try:
            assert len(vectors) == len(metadata)
        except AssertionError:
            raise ValueError('Error in add_vectors: the number of vectors and metadata items must be the same.')
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)
        self.save()

    def search(self, query_vector, top_k=10):
        if not self.vectors:
            return []
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        indexed_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        results = []
        for i, similarity in indexed_similarities[:top_k]:
            result = {
                'metadata': self.metadata[i],
                'similarity': similarity,
            }
            results.append(result)
        return results

    def remove_document(self, doc_id):
        i = 0
        while i < len(self.metadata):
            if self.metadata[i]['doc_id'] == doc_id:
                del self.vectors[i]
                del self.metadata[i]
            else:
                i += 1
        self.save()

    def save(self):
        with open(self.storage_path, 'wb') as f:
            pickle.dump((self.vectors, self.metadata), f)

    def load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'rb') as f:
                self.vectors, self.metadata = pickle.load(f)
        else:
            self.vectors = []
            self.metadata = []