import os
import sys
import unittest
import shutil
import psycopg2
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.database.chunk.basic_db import BasicChunkDB
from dsrag.database.chunk.sqlite_db import SQLiteDB
from dsrag.database.chunk.db import ChunkDB
from dsrag.database.chunk.postgres_db import PostgresChunkDB
from dsrag.database.chunk import DynamoDB


class TestChunkDB(unittest.TestCase):
    def setUp(self):
        self.storage_directory = "~/test__chunk_db_dsRAG"
        self.kb_id = "test_kb"
        resolved_test_storage_directory = os.path.expanduser(self.storage_directory)
        if os.path.exists(resolved_test_storage_directory):
            shutil.rmtree(resolved_test_storage_directory)
        return super().setUp()

    @classmethod
    def tearDownClass(cls):
        test_storage_directory = "~/test__chunk_db_dsRAG"
        resolved_test_storage_directory = os.path.expanduser(test_storage_directory)
        if os.path.exists(resolved_test_storage_directory):
            shutil.rmtree(resolved_test_storage_directory)
        return super().tearDownClass()

    def test__add_and_get_chunk_text(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
            },
            1: {
                "chunk_text": "Content of chunk 2",
                "document_title": "Title of document 2",
                "document_summary": "Summary of document 2",
                "section_title": "Section title 2",
                "section_summary": "Section summary 2",
            },
        }
        db.add_document(doc_id, chunks)
        retrieved_chunk = db.get_chunk_text(doc_id, 0)
        self.assertEqual(retrieved_chunk, chunks[0]["chunk_text"])

    def test__get_chunk_page_numbers(self):

        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"

        chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
                "chunk_page_start": 1,
                "chunk_page_end": 2,
            },
            1: {
                "chunk_text": "Content of chunk 2",
                "document_title": "Title of document 2",
                "document_summary": "Summary of document 2",
                "section_title": "Section title 2",
                "section_summary": "Section summary 2",
            },
        }
        db.add_document(doc_id, chunks)
        page_numbers = db.get_chunk_page_numbers(doc_id, 0)
        self.assertEqual(page_numbers, (1, 2))

        page_numbers = db.get_chunk_page_numbers(doc_id, 1)
        self.assertEqual(page_numbers, (None, None))

    def test__get_document_title(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"document_title": "Title 1", "chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        title = db.get_document_title(doc_id, 0)
        self.assertEqual(title, "Title 1")

    def test__get_document_summary(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {"document_summary": "Summary 1", "chunk_text": "Content of chunk 1"}
        }
        db.add_document(doc_id, chunks)
        summary = db.get_document_summary(doc_id, 0)
        self.assertEqual(summary, "Summary 1")

    def test__get_section_title(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"section_title": "Title 1", "chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        title = db.get_section_title(doc_id, 0)
        self.assertEqual(title, "Title 1")

    def test__get_section_summary(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {"section_summary": "Summary 1", "chunk_text": "Content of chunk 1"}
        }
        db.add_document(doc_id, chunks)
        summary = db.get_section_summary(doc_id, 0)
        self.assertEqual(summary, "Summary 1")

    def test__remove_document(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        db.remove_document(doc_id)
        self.assertNotIn(doc_id, db.data)

    def test__persistence(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
            },
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

    def test__delete(self):
        db = BasicChunkDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        # Make sure the storage directory exists before deleting it
        self.assertTrue(os.path.exists(db.storage_path))
        db.delete()
        # Make sure the storage directory does not exist
        self.assertFalse(os.path.exists(db.storage_path))


class TestSQLiteDB(unittest.TestCase):

    def setUp(self):
        self.storage_directory = "~/test__sqlite_db_dsRAG"
        self.kb_id = "test_kb"
        resolved_test_storage_directory = os.path.expanduser(self.storage_directory)
        if os.path.exists(resolved_test_storage_directory):
            shutil.rmtree(resolved_test_storage_directory)
        return super().setUp()

    @classmethod
    def tearDownClass(cls):
        test_storage_directory = "~/test__chunk_db_dsRAG"
        resolved_test_storage_directory = os.path.expanduser(test_storage_directory)
        if os.path.exists(resolved_test_storage_directory):
            shutil.rmtree(resolved_test_storage_directory)
        return super().tearDownClass()

    def test__add_and_get_chunk_text(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
            },
            1: {
                "chunk_text": "Content of chunk 2",
                "document_title": "Title of document 2",
                "document_summary": "Summary of document 2",
                "section_title": "Section title 2",
                "section_summary": "Section summary 2",
            },
        }
        db.add_document(doc_id, chunks)
        retrieved_chunk = db.get_chunk_text(doc_id, 0)
        self.assertEqual(retrieved_chunk, chunks[0]["chunk_text"])

    def test__get_chunk_page_numbers(self):
            
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"

        chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
                "chunk_page_start": 1,
                "chunk_page_end": 2,
            },
            1: {
                "chunk_text": "Content of chunk 2",
                "document_title": "Title of document 2",
                "document_summary": "Summary of document 2",
                "section_title": "Section title 2",
                "section_summary": "Section summary 2",
            },
        }
        db.add_document(doc_id, chunks)
        page_numbers = db.get_chunk_page_numbers(doc_id, 0)
        self.assertEqual(page_numbers, (1, 2))

        page_numbers = db.get_chunk_page_numbers(doc_id, 1)
        self.assertEqual(page_numbers, (None, None))

    def test__get_document_title(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"document_title": "Title 1", "chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        title = db.get_document_title(doc_id, 0)
        self.assertEqual(title, "Title 1")

    def test__get_document_summary(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {"document_summary": "Summary 1", "chunk_text": "Content of chunk 1"}
        }
        db.add_document(doc_id, chunks)
        summary = db.get_document_summary(doc_id, 0)
        self.assertEqual(summary, "Summary 1")

    def test__get_document_content(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {"chunk_text": "Content of chunk 1"},
            1: {"chunk_text": "Content of chunk 2"},
        }
        db.add_document(doc_id, chunks)
        content = db.get_document(doc_id, include_content=True)
        self.assertEqual(content["content"], "Content of chunk 1\nContent of chunk 2")

    def test__get_section_title(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"section_title": "Title 1", "chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        title = db.get_section_title(doc_id, 0)
        self.assertEqual(title, "Title 1")

    def test__get_section_summary(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {
            0: {"section_summary": "Summary 1", "chunk_text": "Content of chunk 1"}
        }
        db.add_document(doc_id, chunks)
        summary = db.get_section_summary(doc_id, 0)
        self.assertEqual(summary, "Summary 1")

    def test__get_by_supp_id(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        supp_id = "Supp ID 1"
        chunks = {
            0: {"chunk_text": "Content of chunk 1"},
        }
        db.add_document(doc_id=doc_id, chunks=chunks, supp_id=supp_id)
        doc_id = "doc2"
        chunks = {
            0: {"chunk_text": "Content of chunk 2"},
        }
        db.add_document(doc_id=doc_id, chunks=chunks)
        docs = db.get_all_doc_ids("Supp ID 1")
        # There should only be one document with the supp_id 'Supp ID 1'
        self.assertEqual(len(docs), 1)

    def test__remove_document(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        db.remove_document(doc_id)
        results = db.get_document(doc_id)
        # Make sure the document does not exist, it should just be None
        self.assertIsNone(results)

    def test__save_and_load_from_dict(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        config = db.to_dict()
        db2 = ChunkDB.from_dict(config)
        assert db2.kb_id == db.kb_id, "Failed to load kb_id from dict."
        self.assertEqual(db2.kb_id, db.kb_id)

    def test__delete(self):
        db = SQLiteDB(self.kb_id, self.storage_directory)
        doc_id = "doc1"
        chunks = {0: {"chunk_text": "Content of chunk 1"}}
        db.add_document(doc_id, chunks)
        # Make sure the storage directory exists before deleting it
        self.assertTrue(os.path.exists(os.path.join(db.db_path, f"{self.kb_id}.db")))
        db.delete()
        # Make sure the storage directory does not exist
        self.assertFalse(os.path.exists(os.path.join(db.db_path, f"{self.kb_id}.db")))

class TestDynamoDB(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.table_name = "test_dynamo_db_chunks"
        self.kb_id = "test_dynamo_db"
        self.db = DynamoDB(
            self.kb_id
        )
        #return super().setUp()

    @classmethod
    def tearDownClass(self):
        resp = self.db.delete()
        print ("resp", resp)
        #return super().tearDownClass()

    def test__add_and_get_chunk_text(self):
        doc_id = "doc1"
        chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
            },
            1: {
                "chunk_text": "Content of chunk 2",
                "document_title": "Title of document 2",
                "document_summary": "Summary of document 2",
                "section_title": "Section title 2",
                "section_summary": "Section summary 2",
            },
        }
        self.db.add_document(doc_id, chunks)
        retrieved_chunk = self.db.get_chunk_text(doc_id, 0)
        self.assertEqual(retrieved_chunk, chunks[0]["chunk_text"])

    def test__get_chunk_page_numbers(self):
            
        #db = self.db
        doc_id = "doc1"

        chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
                "chunk_page_start": 1,
                "chunk_page_end": 2,
            },
            1: {
                "chunk_text": "Content of chunk 2",
                "document_title": "Title of document 2",
                "document_summary": "Summary of document 2",
                "section_title": "Section title 2",
                "section_summary": "Section summary 2",
            },
        }
        self.db.add_document(doc_id, chunks)
        page_numbers = self.db.get_chunk_page_numbers(doc_id, 0)
        self.assertEqual(page_numbers, (1, 2))

        page_numbers = self.db.get_chunk_page_numbers(doc_id, 1)
        self.assertEqual(page_numbers, (None, None))

    def test__get_document_title(self):
        doc_id = "doc1"
        chunks = {0: {"document_title": "Title 1", "chunk_text": "Content of chunk 1"}}
        self.db.add_document(doc_id, chunks)
        title = self.db.get_document_title(doc_id, 0)
        self.assertEqual(title, "Title 1")

    def test__get_document_summary(self):
        doc_id = "doc1"
        chunks = {
            0: {"document_summary": "Summary 1", "chunk_text": "Content of chunk 1"}
        }
        self.db.add_document(doc_id, chunks)
        summary = self.db.get_document_summary(doc_id, 0)
        self.assertEqual(summary, "Summary 1")

    def test__get_document_content(self):
        doc_id = "doc1"
        chunks = {
            0: {"chunk_text": "Content of chunk 1"},
            1: {"chunk_text": "Content of chunk 2"},
        }
        self.db.add_document(doc_id, chunks)
        content = self.db.get_document(doc_id, include_content=True)
        self.assertEqual(content["content"], "Content of chunk 1\nContent of chunk 2")

    def test__get_section_title(self):
        doc_id = "doc1"
        chunks = {0: {"section_title": "Title 1", "chunk_text": "Content of chunk 1"}}
        self.db.add_document(doc_id, chunks)
        title = self.db.get_section_title(doc_id, 0)
        self.assertEqual(title, "Title 1")

    def test__get_section_summary(self):
        doc_id = "doc1"
        chunks = {
            0: {"section_summary": "Summary 1", "chunk_text": "Content of chunk 1"}
        }
        self.db.add_document(doc_id, chunks)
        summary = self.db.get_section_summary(doc_id, 0)
        self.assertEqual(summary, "Summary 1")

    def test__get_by_supp_id(self):
        doc_id = "doc1"
        supp_id = "Supp ID 1"
        chunks = {
            0: {"chunk_text": "Content of chunk 1"},
        }
        self.db.add_document(doc_id=doc_id, chunks=chunks, supp_id=supp_id)
        doc_id = "doc2"
        chunks = {
            0: {"chunk_text": "Content of chunk 2"},
        }
        self.db.add_document(doc_id=doc_id, chunks=chunks)
        time.sleep(2) # Wait for the data to be consistent
        docs = self.db.get_all_doc_ids("Supp ID 1")
        # There should only be one document with the supp_id 'Supp ID 1'
        self.assertEqual(len(docs), 1)

    def test__remove_document(self):
        doc_id = "doc1"
        chunks = {0: {"chunk_text": "Content of chunk 1"}}
        self.db.add_document(doc_id, chunks)
        self.db.remove_document(doc_id)
        results = self.db.get_document(doc_id)
        # Make sure the document does not exist, it should just be None
        self.assertIsNone(results)

    def test__save_and_load_from_dict(self):
        db = DynamoDB(
            self.kb_id,
            table_name=self.table_name,
        )
        config = db.to_dict()
        db2 = ChunkDB.from_dict(config)
        assert db2.kb_id == db.kb_id, "Failed to load kb_id from dict."
        self.assertEqual(db2.kb_id, db.kb_id)

class TestPostgresChunkDB(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.kb_id = "test_kb"
        self.host = os.environ.get("POSTGRES_HOST")
        self.port = 5432
        self.username = os.environ.get("POSTGRES_USER")
        self.password = os.environ.get("POSTGRES_PASSWORD")
        self.database = os.environ.get("POSTGRES_DB")

        self.db = PostgresChunkDB(
            kb_id=self.kb_id,
            username=self.username,
            password=self.password,
            database=self.database,
            host=self.host,
            port=self.port,
        )
        self.doc_id = "test_doc_id"
        self.chunks = {
            0: {
                "chunk_text": "Content of chunk 1",
                "document_title": "Title of document 1",
                "document_summary": "Summary of document 1",
                "section_title": "Section title 1",
                "section_summary": "Section summary 1",
                "chunk_page_start": 1,
                "chunk_page_end": 2,
            },
            1: {
                "chunk_text": "Content of chunk 2",
                "document_title": "Title of document 2",
                "document_summary": "Summary of document 2",
                "section_title": "Section title 2",
                "section_summary": "Section summary 2",
            },
        }


    def test__001_add_and_get_chunk_text(self):
        self.db.add_document(self.doc_id, self.chunks)
        retrieved_chunk = self.db.get_chunk_text(self.doc_id, 0)
        self.assertEqual(retrieved_chunk, self.chunks[0]["chunk_text"])

    def test__002_get_chunk_page_numbers(self):
        page_numbers = self.db.get_chunk_page_numbers(self.doc_id, 0)
        self.assertEqual(page_numbers, (1, 2))

        page_numbers = self.db.get_chunk_page_numbers(self.doc_id, 1)
        self.assertEqual(page_numbers, (None, None))

    def test__003_get_document_title(self):
        title = self.db.get_document_title(self.doc_id, 0)
        self.assertEqual(title, "Title of document 1")

    def test__004_get_document_summary(self):
        summary = self.db.get_document_summary(self.doc_id, 0)
        self.assertEqual(summary, "Summary of document 1")

    def test__005_get_document_content(self):
        content = self.db.get_document(self.doc_id, include_content=True)
        self.assertEqual(content["content"], "Content of chunk 1\nContent of chunk 2")

    def test__006_get_section_title(self):
        title = self.db.get_section_title(self.doc_id, 0)
        self.assertEqual(title, "Section title 1")

    def test__007_get_section_summary(self):
        summary = self.db.get_section_summary(self.doc_id, 0)
        self.assertEqual(summary, "Section summary 1")

    def test__008_get_by_supp_id(self):
        doc_id = "doc1"
        supp_id = "Supp ID 1"
        chunks = {
            0: {"chunk_text": "Content of chunk 1"},
        }
        self.db.add_document(doc_id=doc_id, chunks=chunks, supp_id=supp_id)
        doc_id = "doc2"
        chunks = {
            0: {"chunk_text": "Content of chunk 2"},
        }
        self.db.add_document(doc_id=doc_id, chunks=chunks)
        docs = self.db.get_all_doc_ids("Supp ID 1")
        # There should only be one document with the supp_id 'Supp ID 1'
        self.assertEqual(len(docs), 1)

    def test__009_remove_document(self):
        self.db.remove_document(self.doc_id)
        results = self.db.get_document(self.doc_id)
        # Make sure the document does not exist, it should just be None
        self.assertIsNone(results)

    def test__010_save_and_load_from_dict(self):
        config = self.db.to_dict()
        db2 = PostgresChunkDB.from_dict(config)
        assert db2.kb_id == self.db.kb_id, "Failed to load kb_id from dict."
        self.assertEqual(db2.kb_id, self.db.kb_id)

    def test__011_delete(self):
        self.db.delete()
        try:
            _ = self.db.get_all_doc_ids()
        except Exception as e:
            # The table shouldn't exist anymore
            self.assertTrue("does not exist" in str(e))
    

    @classmethod
    def tearDownClass(self):
        # Delete the table
        try:
            self.db.delete()
        except:
            pass



# Run all tests
if __name__ == "__main__":
    unittest.main()