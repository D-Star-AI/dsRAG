import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sprag.vector_db import BasicVectorDB, VectorDB

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

def test_load_from_dict():
    config = {
        'subclass_name': 'BasicVectorDB',
        'kb_id': 'test_db',
        'storage_directory': '/tmp'
    }
    vector_db_instance = VectorDB.from_dict(config)
    assert isinstance(vector_db_instance, BasicVectorDB)
    assert vector_db_instance.kb_id == 'test_db'
    teardown()

def test_save_and_load_from_dict():
    db = BasicVectorDB("test_db", "/tmp")
    config = db.to_dict()
    vector_db_instance = VectorDB.from_dict(config)
    assert isinstance(vector_db_instance, BasicVectorDB)
    assert vector_db_instance.kb_id == 'test_db'
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

def test_faiss_search():
    db = BasicVectorDB("test_db", "/tmp", use_faiss=True)
    vectors = [np.array([1, 0]), np.array([0, 1])]
    metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
    
    db.add_vectors(vectors, metadata)
    query_vector = np.array([1, 0])
    
    faiss_results = db.search(query_vector, top_k=1)

    db.use_faiss = False
    non_faiss_results = db.search(query_vector, top_k=1)

    assert faiss_results == non_faiss_results
    teardown()


if __name__ == '__main__':
    teardown()
    test_add_vectors_and_search()
    test_remove_document()
    test_empty_search()
    test_save_and_load()
    test_assertion_error_on_mismatched_input_lengths()
    test_load_from_dict()
    test_save_and_load_from_dict()
    test_faiss_search()
    print("All tests passed!")