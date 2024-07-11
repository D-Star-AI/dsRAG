import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.chunk_db import BasicChunkDB, ChunkDB
import shutil


class TestChunkDB(unittest.TestCase):
    def setUp(self):
        self.storage_directory = '~/test_dsRAG'
        self.kb_id = 'test_kb'
        resolved_test_storage_directory = os.path.expanduser(self.storage_directory)
        if os.path.exists(resolved_test_storage_directory):
            shutil.rmtree(resolved_test_storage_directory)
        return super().setUp()

    @classmethod
    def tearDownClass(cls):
        test_storage_directory = '~/test_dsRAG'
        resolved_test_storage_directory = os.path.expanduser(test_storage_directory)
        if os.path.exists(resolved_test_storage_directory):
            shutil.rmtree(resolved_test_storage_directory)
        return super().tearDownClass()


    def test__add_and_get_chunk_text(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = 'doc1'
        chunks = {
            0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'},
            1: {'chunk_header': 'Header 2', 'chunk_text': 'Content of chunk 2'}
        }
        db.add_document(doc_id, chunks)
        retrieved_chunk = db.get_chunk_text(doc_id, 0)
        self.assertEqual(retrieved_chunk, chunks[0]['chunk_text'])

    def test__get_chunk_header(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = 'doc1'
        chunks = {
            0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'}
        }
        db.add_document(doc_id, chunks)
        header = db.get_chunk_header(doc_id, 0)
        self.assertEqual(header, 'Header 1')

    def test__remove_document(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = 'doc1'
        chunks = {
            0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'}
        }
        db.add_document(doc_id, chunks)
        db.remove_document(doc_id)
        self.assertNotIn(doc_id, db.data)

    def test__persistence(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = 'doc1'
        chunks = {
            0: {'chunk_header': 'Header 1', 'chunk_text': 'Content of chunk 1'}
        }
        db.add_document(doc_id, chunks)
        db2 = BasicChunkDB(self.kb_id, self.storage_directory)
        self.assertIn(doc_id, db2.data)

    def test__save_and_load_from_dict(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        config = db.to_dict()
        db2 = ChunkDB.from_dict(config)
        assert db2.kb_id == db.kb_id, "Failed to load kb_id from dict."
        self.assertEqual(db2.kb_id, db.kb_id)

# Run all tests
if __name__ == '__main__':
    unittest.main()