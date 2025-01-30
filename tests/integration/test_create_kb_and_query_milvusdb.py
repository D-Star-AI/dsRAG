import sys
import os
import unittest

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.knowledge_base import KnowledgeBase
from dsrag.database.vector import MilvusDB


class TestMilvusDB(unittest.TestCase):
    def setUp(self):
        # Set up paths and parameters
        self.kb_id = "levels_of_agi"
        self.storage_directory = './test_milvus_db'
        self.dimension = 768
        self.cleanup()  # Ensure a clean start

    def tearDown(self):
        self.cleanup()  # Clean up after tests

    def cleanup(self):
        # Delete the KnowledgeBase and related files if they exist
        try:
            kb = KnowledgeBase(kb_id=self.kb_id, exists_ok=True)
            kb.delete()
        except Exception as e:
            pass  # If it doesn't exist, do nothing

        # Ensure MilvusDB is cleaned up as well
        db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        db.delete()

    def test__001_create_kb_and_query(self):
        # Get the absolute path of the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the dataset file
        file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")


        # Initialize MilvusDB and KnowledgeBase
        vector_db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        kb = KnowledgeBase(kb_id=self.kb_id, vector_db=vector_db, exists_ok=False)
        kb.reranker.model = "rerank-english-v3.0"

        # Add document to the knowledge base
        kb.add_document(
            doc_id="levels_of_agi.pdf",
            document_title="Levels of AGI",
            file_path=file_path,
            semantic_sectioning_config={"use_semantic_sectioning": False},
            auto_context_config={
                "use_generated_title": False,
                "get_document_summary": False
            }
        )

        # Verify that the document is in the chunk database
        self.assertEqual(len(kb.chunk_db.get_all_doc_ids()), 1)

        # Run a query and verify results are returned
        search_queries = ["What are the levels of AGI?", "What is the highest level of AGI?"]
        segment_info = kb.query(search_queries)
        self.assertGreater(len(segment_info[0]), 0)
        # Assert that the chunk page start and end are correct
        self.assertEqual(segment_info[0]["chunk_page_start"], 5)
        self.assertEqual(segment_info[0]["chunk_page_end"], 8)

    def test__002_test_query_with_metadata_filter(self):
        vector_db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        kb = KnowledgeBase(kb_id=self.kb_id, vector_db=vector_db, exists_ok=False)
        document_text = "AGI has many levels. The highest level of AGI is level 5."
        kb.add_document(
            doc_id="test_doc",
            document_title="AGI Test Document",
            text=document_text,
            semantic_sectioning_config={"use_semantic_sectioning": False},
            auto_context_config={
                "use_generated_title": False, # No need to generate a title for this document for testing purposes
                "get_document_summary": False
            }
        )

        search_queries = [
            "What is the highest level of AGI?",
        ]
        metadata_filter = {"field": "doc_id", "operator": "equals", "value": "test_doc"}
        segment_info = kb.query(search_queries, metadata_filter=metadata_filter, rse_params={"minimum_value": 0})
        self.assertEqual(len(segment_info), 1)
        self.assertEqual(segment_info[0]["doc_id"], "test_doc")
        self.assertEqual(segment_info[0]["chunk_page_start"], None)
        self.assertEqual(segment_info[0]["chunk_page_end"], None)

    def test__003_remove_document(self):
        # Initialize KnowledgeBase and remove document
        vector_db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        kb = KnowledgeBase(kb_id=self.kb_id, vector_db=vector_db, exists_ok=False)
        document_text = "AGI has many levels. The highest level of AGI is level 5."
        kb.add_document(
            doc_id="test_doc",
            document_title="AGI Test Document",
            text=document_text,
            semantic_sectioning_config={"use_semantic_sectioning": False},
            auto_context_config={
                "use_generated_title": False,
                # No need to generate a title for this document for testing purposes
                "get_document_summary": False
            }
        )
        self.assertEqual(kb.vector_db.get_num_vectors(), 1)
        # Remove document and verify it's deleted
        kb.delete_document(doc_id="test_doc")
        self.assertEqual(kb.vector_db.get_num_vectors(), 0)

        # Clean up KnowledgeBase
        kb.delete()

    def test__004_query_empty_kb(self):
        # Initialize KnowledgeBase and query empty KB
        vector_db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        kb = KnowledgeBase(kb_id=self.kb_id, vector_db=vector_db, exists_ok=True)

        # Query an empty knowledge base and verify no results
        results = kb.query(["What are the levels of AGI?"])
        self.assertEqual(len(results), 0)
        kb.delete()


if __name__ == "__main__":
    unittest.main()
