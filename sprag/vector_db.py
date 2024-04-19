from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
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
        self.storage_directory = os.path.expanduser(storage_directory)  # Expand the user path
        self.storage_path = os.path.join(self.storage_directory, 'vector_storage', f'{kb_id}.pkl')
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
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)  # Ensure the directory exists
        with open(self.storage_path, 'wb') as f:
            pickle.dump((self.vectors, self.metadata), f)

    def load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'rb') as f:
                self.vectors, self.metadata = pickle.load(f)
        else:
            self.vectors = []
            self.metadata = []


import numpy as np

def teardown():
    storage_directory = "/tmp"
    storage_path = os.path.join(storage_directory, 'vector_storage', 'test_db.pkl')
    if os.path.exists(storage_path):
        os.remove(storage_path)

def test_add_vectors_and_search():
    db = BasicVectorDB("test_db", "/tmp")
    vectors = [np.array([1, 0]), np.array([0, 1])]
    metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
    
    db.add_vectors(vectors, metadata)
    query_vector = np.array([1, 0])
    
    results = db.search(query_vector, top_k=1)
    
    assert len(results) == 1
    assert results[0]['metadata']['doc_id'] == '1'
    assert results[0]['similarity'] >= 0.99
    teardown()

def test_remove_document():
    db = BasicVectorDB("test_db", "/tmp")
    vectors = [np.array([1, 0]), np.array([0, 1])]
    metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
    
    db.add_vectors(vectors, metadata)
    db.remove_document('1')
    
    assert len(db.metadata) == 1
    assert db.metadata[0]['doc_id'] == '2'
    teardown()

def test_empty_search():
    db = BasicVectorDB("test_db", "/tmp")
    query_vector = np.array([1, 0])
    results = db.search(query_vector)
    
    assert len(results) == 0
    teardown()

def test_save_and_load():
    db = BasicVectorDB("test_db", "/tmp")
    vectors = [np.array([1, 0]), np.array([0, 1])]
    metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
    
    db.add_vectors(vectors, metadata)
    db.save()
    
    new_db = BasicVectorDB("test_db", "/tmp")
    new_db.load()
    
    assert len(new_db.metadata) == 2
    assert new_db.metadata[0]['doc_id'] == '1'
    assert new_db.metadata[1]['doc_id'] == '2'
    teardown()

def test_assertion_error_on_mismatched_input_lengths():
    db = BasicVectorDB("test_db", "/tmp")
    vectors = [np.array([1, 0])]
    metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
    
    try:
        db.add_vectors(vectors, metadata)
    except ValueError as e:
        assert str(e) == 'Error in add_vectors: the number of vectors and metadata items must be the same.'
    teardown()


if __name__ == '__main__':
    teardown()
    test_add_vectors_and_search()
    test_remove_document()
    test_empty_search()
    test_save_and_load()
    test_assertion_error_on_mismatched_input_lengths()
    print("All tests passed!")