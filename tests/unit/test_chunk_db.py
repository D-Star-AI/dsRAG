import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.chunk_db import BasicChunkDB, ChunkDB
import shutil


def setup_test_environment():
    test_storage_directory = '~/test_spRAG'
    test_kb_id = 'test_kb'
    resolved_test_storage_directory = os.path.expanduser(test_storage_directory)
    if os.path.exists(resolved_test_storage_directory):
        shutil.rmtree(resolved_test_storage_directory)
    return test_kb_id, test_storage_directory

def test_add_and_get_chunk_text():
    kb_id, storage_directory = setup_test_environment()
    db = BasicChunkDB(kb_id, storage_directory)
    doc_id = 'doc1'
    chunks = {
        0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'},
        1: {'chunk_header': 'Header 2', 'chunk_text': 'Content of chunk 2'}
    }
    db.add_document(doc_id, chunks)
    retrieved_chunk = db.get_chunk_text(doc_id, 0)
    assert retrieved_chunk == chunks[0]['chunk_text'], "Failed to retrieve the correct chunk."
    print("test_add_and_get_chunk passed.")

def test_get_chunk_header():
    kb_id, storage_directory = setup_test_environment()
    db = BasicChunkDB(kb_id, storage_directory)
    doc_id = 'doc1'
    chunks = {
        0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'}
    }
    db.add_document(doc_id, chunks)
    header = db.get_chunk_header(doc_id, 0)
    assert header == 'Header 1', "Failed to retrieve the correct chunk header."
    print("test_get_chunk_header passed.")

def test_remove_document():
    kb_id, storage_directory = setup_test_environment()
    db = BasicChunkDB(kb_id, storage_directory)
    doc_id = 'doc1'
    chunks = {
        0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'}
    }
    db.add_document(doc_id, chunks)
    db.remove_document(doc_id)
    assert doc_id not in db.data, "Document was not removed."
    print("test_remove_document passed.")

def test_persistence():
    kb_id, storage_directory = setup_test_environment()
    db = BasicChunkDB(kb_id, storage_directory)
    doc_id = 'doc1'
    chunks = {
        0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'}
    }
    db.add_document(doc_id, chunks)
    db2 = BasicChunkDB(kb_id, storage_directory)
    assert doc_id in db2.data, "Data was not persisted correctly."
    print("test_persistence passed.")

def test_save_and_load_from_dict():
    kb_id, storage_directory = setup_test_environment()
    db = BasicChunkDB(kb_id, storage_directory)
    config = db.to_dict()
    db2 = ChunkDB.from_dict(config)
    assert db2.kb_id == db.kb_id, "Failed to load kb_id from dict."

def run_all_tests():
    test_add_and_get_chunk_text()
    test_get_chunk_header()
    test_remove_document()
    test_persistence()
    test_save_and_load_from_dict()
    cleanup_test_environment()

def cleanup_test_environment():
    test_storage_directory = '~/test_spRAG'
    resolved_test_storage_directory = os.path.expanduser(test_storage_directory)
    if os.path.exists(resolved_test_storage_directory):
        shutil.rmtree(resolved_test_storage_directory)
    print("Cleaned up test environment.")

# Run all tests
if __name__ == '__main__':
    run_all_tests()