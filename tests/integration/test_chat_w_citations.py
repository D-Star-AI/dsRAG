import os
import sys
import unittest

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.chat.chat import create_new_chat_thread, get_chat_thread_response, ChatResponseInput
from dsrag.database.chat_thread.basic_db import BasicChatThreadDB
from dsrag.knowledge_base import KnowledgeBase
from dsrag.database.vector import ChromaDB
from dsrag.reranker import NoReranker

class TestChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleanup()  # Clean up any existing test data
        
        # Set up the knowledge base with the levels of AGI test file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")
        
        kb_id = "levels_of_agi"
        vector_db = ChromaDB(kb_id=kb_id)
        reranker = NoReranker() 
        cls.kb = KnowledgeBase(
            kb_id=kb_id, 
            vector_db=vector_db, 
            reranker=reranker, 
            title="Levels of AGI",
            description="Paper about the levels of AGI, written by Google DeepMind",
            exists_ok=False
        )

        file_parsing_config = {
            "use_vlm": False
        }

        cls.kb.add_document(
            doc_id="levels_of_agi.pdf",
            document_title="Levels of AGI",
            file_path=file_path,
            file_parsing_config=file_parsing_config,
        )
        
        # Set up chat thread parameters
        cls.chat_thread_params = {
            "kb_ids": ["levels_of_agi"],
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
            "system_message": "",
            "auto_query_model": "claude-3-5-sonnet-20241022",
        }
        
        cls.knowledge_bases = {
            "levels_of_agi": cls.kb
        }
        
        cls.chat_thread_db = BasicChatThreadDB()

    def test_001_create_new_chat_thread(self):
        thread_id = create_new_chat_thread(
            chat_thread_params=self.chat_thread_params,
            chat_thread_db=self.chat_thread_db
        )
        self.assertIsNotNone(thread_id)
        return thread_id

    def test_002_get_chat_response_with_citations(self):
        thread_id = create_new_chat_thread(
            chat_thread_params=self.chat_thread_params,
            chat_thread_db=self.chat_thread_db
        )
        
        # Test initial question about levels of AGI
        chat_response_input = ChatResponseInput(
            user_input="What are the levels of AGI?",
            chat_thread_params=None,
            metadata_filter=None
        )
        
        response = get_chat_thread_response(
            thread_id=thread_id,
            get_response_input=chat_response_input,
            chat_thread_db=self.chat_thread_db,
            knowledge_bases=self.knowledge_bases
        )
        
        # Verify response structure
        self.assertIn("model_response", response)
        self.assertIn("content", response["model_response"])
        self.assertIn("citations", response["model_response"])
        
        # Verify citations
        citations = response["model_response"]["citations"]
        self.assertGreater(len(citations), 0)
        
        # Verify citation structure
        first_citation = citations[0]
        self.assertIn("doc_id", first_citation)
        self.assertEqual(first_citation["doc_id"], "levels_of_agi.pdf")
        self.assertIn("page_numbers", first_citation)
        
        # Test follow-up question
        chat_response_input = ChatResponseInput(
            user_input="What is the highest level of AGI?",
            chat_thread_params=None,
            metadata_filter=None
        )
        
        follow_up_response = get_chat_thread_response(
            thread_id=thread_id,
            get_response_input=chat_response_input,
            chat_thread_db=self.chat_thread_db,
            knowledge_bases=self.knowledge_bases
        )
        
        # Verify follow-up response and citations
        self.assertIn("citations", follow_up_response["model_response"])
        self.assertGreater(len(follow_up_response["model_response"]["citations"]), 0)

    @classmethod
    def cleanup(cls):
        try:
            kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
            kb.delete()
        except:
            pass

    @classmethod
    def tearDownClass(cls):
        cls.cleanup()

if __name__ == "__main__":
    unittest.main()