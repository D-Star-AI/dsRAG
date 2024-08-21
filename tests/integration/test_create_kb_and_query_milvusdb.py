import sys
import os
import unittest
import numpy as np

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.knowledge_base import KnowledgeBase
from dsrag.document_parsing import extract_text_from_pdf
from dsrag.vector_db import MilvusDB


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

        # Extract text from the PDF file
        document_text = extract_text_from_pdf(file_path)

        # Initialize MilvusDB and KnowledgeBase
        vector_db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        kb = KnowledgeBase(kb_id=self.kb_id, vector_db=vector_db, exists_ok=False)

        # Add document to the knowledge base
        kb.add_document(doc_id="levels_of_agi.pdf", text=document_text,
                        semantic_sectioning_config={"use_semantic_sectioning": False})

        # Verify that the document is in the chunk database
        self.assertEqual(len(kb.chunk_db.get_all_doc_ids()), 1)

        # Run a query and verify results are returned
        search_queries = ["What are the levels of AGI?", "What is the highest level of AGI?"]
        segment_info = kb.query(search_queries)
        self.assertGreater(len(segment_info[0]), 0)

    def test__002_remove_document(self):
        # Initialize KnowledgeBase and remove document
        vector_db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        kb = KnowledgeBase(kb_id=self.kb_id, vector_db=vector_db, exists_ok=True)

        # Remove document and verify it's deleted
        kb.delete_document(doc_id="levels_of_agi.pdf")
        self.assertEqual(len(kb.chunk_db.get_all_doc_ids()), 0)

        # Clean up KnowledgeBase
        kb.delete()

    def test__003_query_empty_kb(self):
        # Initialize KnowledgeBase and query empty KB
        vector_db = MilvusDB(kb_id=self.kb_id, storage_directory=self.storage_directory, dimension=self.dimension)
        kb = KnowledgeBase(kb_id=self.kb_id, vector_db=vector_db, exists_ok=True)

        # Query an empty knowledge base and verify no results
        results = kb.query(["What are the levels of AGI?"])
        self.assertEqual(len(results), 0)
        kb.delete()


if __name__ == "__main__":
    unittest.main()
