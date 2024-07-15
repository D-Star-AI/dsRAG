import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.create_kb import create_kb_from_file
from dsrag.knowledge_base import KnowledgeBase
from integrations.langchain_retriever import DsRAGLangchainRetriever
from langchain_core.documents import Document


class TestDsRAGLangchainRetriever(unittest.TestCase):

    def setUp(self):

        """ We have to create a knowledge base from a file before we can run the retriever tests."""

        # Get the absolute path of the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the dataset file
        file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")
        
        kb_id = "levels_of_agi"
        create_kb_from_file(kb_id, file_path)
        self.kb_id = kb_id

    def test_create_kb_and_query(self):

        retriever = DsRAGLangchainRetriever(self.kb_id)
        documents = retriever.invoke("What are the levels of AGI?")

        # Make sure results are returned
        self.assertTrue(len(documents) > 0)

        # Make sure the results are Document objects, and that they have page content and metadata
        for doc in documents:
            assert isinstance(doc, Document)
            assert doc.page_content
            assert doc.metadata

    def tearDown(self):
        # Delete the KnowledgeBase object after the test runs
        kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
        kb.delete()


if __name__ == "__main__":
    unittest.main()