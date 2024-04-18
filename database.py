from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import pickle
import os


class VectorDB(ABC):
    @abstractmethod
    def add_vector(self, vector, metadata):
        """
        Store a vector with associated metadata.
        """
        pass

    @abstractmethod
    def remove_vector(self, index):
        """
        Remove a vector from the database by its index.
        """
        pass

    @abstractmethod
    def update_vector(self, index, new_vector):
        """
        Update an existing vector.
        """
        pass

    @abstractmethod
    def search(self, query_vector, top_k=10):
        """
        Retrieve the top-k closest vectors to a given query vector.
        """
        pass


class BasicVectorDB(VectorDB):
    def __init__(self, kb_id: str, storage_directory: str = '~/spRAG'):
        self.storage_path = f'{storage_directory}/vector_storage/{kb_id}.pkl'
        self.vectors = []
        self.metadata = []
        self.load()

    def add_vector(self, vector, metadata):
        self.vectors.append(vector)
        self.metadata.append(metadata)
        self.save()

    def search(self, query_vector, top_k=10):
        if not self.vectors:
            return []
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        indexed_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return [(self.metadata[i], sim) for i, sim in indexed_similarities[:top_k]]

    def remove_vector(self, index):
        del self.vectors[index]
        del self.metadata[index]
        self.save()

    def update_vector(self, index, new_vector):
        self.vectors[index] = new_vector
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


class PersistentFaissVectorDB(VectorDB):
    def __init__(self, dimension, storage_path='faiss_index.idx'):
        self.dimension = dimension
        self.storage_path = storage_path
        self.index = None
        self.metadata = []
        self.load_or_initialize_index()

    def add_vector(self, vector, meta):
        if len(vector) != self.dimension:
            raise ValueError("Vector dimension mismatch.")
        self.index.add(np.array([vector], dtype='float32'))
        self.metadata.append(meta)
        self.save_index()

    def search(self, query_vector, top_k=10):
        if len(query_vector) != self.dimension:
            raise ValueError("Query vector dimension mismatch.")
        distances, indices = self.index.search(np.array([query_vector], dtype='float32'), top_k)
        return [(self.metadata[idx], 1 - dist) for idx, dist in zip(indices[0], distances[0])]

    def remove_vector(self, index):
        # Faiss does not support removing vectors by default, handle accordingly
        pass

    def update_vector(self, index, new_vector):
        # Faiss does not support updating vectors by default, handle accordingly
        pass

    def save_index(self):
        faiss.write_index(self.index, self.storage_path)

    def load_or_initialize_index(self):
        if os.path.exists(self.storage_path):
            self.index = faiss.read_index(self.storage_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)