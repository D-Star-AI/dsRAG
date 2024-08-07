import sys
import os
import unittest

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.knowledge_base import KnowledgeBase
from dsrag.document_parsing import extract_text_from_pdf
from dsrag.vector_db import ChromaDB


class TestCreateKB(unittest.TestCase):    
    def test__001_create_kb_and_query(self):
        self.cleanup() # delete the KnowledgeBase object if it exists so we can start fresh
        
        # Get the absolute path of the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the dataset file
        file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")
        
        kb_id = "levels_of_agi"
        document_text = extract_text_from_pdf(file_path)

        vector_db = ChromaDB(kb_id=kb_id)
        kb = KnowledgeBase(kb_id=kb_id, vector_db=vector_db, exists_ok=False)
        kb.add_document(doc_id="levels_of_agi.pdf", text=document_text, semantic_sectioning_config={"use_semantic_sectioning": False})

        # verify that the document is in the chunk db
        self.assertEqual(len(kb.chunk_db.get_all_doc_ids()), 1)

        # run a query and verify results are returned
        search_queries = ["What are the levels of AGI?", "What is the highest level of AGI?"]
        segment_info = kb.query(search_queries)
        self.assertGreater(len(segment_info[0]), 0)
    
    def test__002_remove_document(self):
        kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
        kb.delete_document(doc_id="levels_of_agi.pdf")
        self.assertEqual(len(kb.chunk_db.get_all_doc_ids()), 0)

        # delete the KnowledgeBase object
        kb.delete()

    def cleanup(self):
        kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
        kb.delete()

if __name__ == "__main__":
    unittest.main()