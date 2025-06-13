import os
import sys
import unittest

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.chat.chat import create_new_chat_thread, get_chat_thread_response, ChatResponseInput
from dsrag.database.chat_thread.basic_db import BasicChatThreadDB
from dsrag.knowledge_base import KnowledgeBase
from dsrag.reranker import NoReranker

class TestChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleanup()  # Clean up any existing test data
        
        # Set up the knowledge base with the levels of AGI test file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")
        
        kb_id = "levels_of_agi"
        reranker = NoReranker() 
        cls.kb = KnowledgeBase(
            kb_id=kb_id, 
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

        rse_params = {
            'max_length': 20,
            'overall_max_length': 60,
            'minimum_value': 0.0,
            'irrelevant_chunk_penalty': 0.15,
            'overall_max_length_extension': 10,
            'decay_rate': 50,
            'top_k_for_document_selection': 30,
            'chunk_length_adjustment': True,
        }
        
        # Set up chat thread parameters
        cls.chat_thread_params = {
            "kb_ids": ["levels_of_agi"],
            "model": "openai/o4-mini",
            "temperature": 0.0,
            "system_message": "",
            "auto_query_model": "openai/gpt-4.1-mini",
            "rse_params": rse_params
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

    def test_003_get_chat_response_streaming(self):
        thread_id = create_new_chat_thread(
            chat_thread_params=self.chat_thread_params,
            chat_thread_db=self.chat_thread_db
        )
        
        # Test streaming question about levels of AGI
        chat_response_input = ChatResponseInput(
            user_input="What are the levels of AGI?",
            chat_thread_params=None,
            metadata_filter=None
        )
        
        response_generator = get_chat_thread_response(
            thread_id=thread_id,
            get_response_input=chat_response_input,
            chat_thread_db=self.chat_thread_db,
            knowledge_bases=self.knowledge_bases,
            stream=True
        )
        
        accumulated_content = ""
        final_citations = []
        message_id = None

        for partial_response in response_generator:
            self.assertIn("model_response", partial_response)
            self.assertIn("content", partial_response["model_response"])
            self.assertIn("status", partial_response["model_response"])
            self.assertIn("message_id", partial_response)
            
            if message_id is None:
                message_id = partial_response["message_id"]
            else:
                self.assertEqual(message_id, partial_response["message_id"])

            accumulated_content = partial_response["model_response"]["content"]
            if "citations" in partial_response["model_response"]:
                final_citations = partial_response["model_response"]["citations"]
        
        # Verify final accumulated response
        self.assertGreater(len(accumulated_content), 0)
        
        # Verify citations received
        self.assertGreater(len(final_citations), 0)
        
        # Verify citation structure
        first_citation = final_citations[0]
        self.assertIn("doc_id", first_citation)
        self.assertEqual(first_citation["doc_id"], "levels_of_agi.pdf")

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

class TestChatWithNoKBs(unittest.TestCase):
    def test_001_create_new_chat_thread(self):
        chat_thread_params = {
            "kb_ids": [],
            "model": "gemini/gemini-2.5-flash-preview-05-20",
        }
        
        chat_thread_db = BasicChatThreadDB()
        thread_id = create_new_chat_thread(
            chat_thread_params=chat_thread_params,
            chat_thread_db=chat_thread_db
        )
        self.assertIsNotNone(thread_id)
        return thread_id
        
    def test_002_get_chat_response_without_kb(self):
        chat_thread_db = BasicChatThreadDB()
        chat_thread_params = {
            "kb_ids": [],
            "model": "anthropic/claude-3-5-sonnet-20241022",
            "temperature": 0.0,
        }
        
        thread_id = create_new_chat_thread(
            chat_thread_params=chat_thread_params,
            chat_thread_db=chat_thread_db
        )
        
        # Test sending a message without knowledge base
        chat_response_input = ChatResponseInput(
            user_input="What is the capital of France?",
            chat_thread_params=None,
            metadata_filter=None
        )
        
        response = get_chat_thread_response(
            thread_id=thread_id,
            get_response_input=chat_response_input,
            chat_thread_db=chat_thread_db,
            knowledge_bases={}
        )
        
        # Verify response structure
        self.assertIn("model_response", response)
        self.assertIn("content", response["model_response"])
        
        # No citations should be present or they should be empty
        if "citations" in response["model_response"]:
            self.assertEqual(len(response["model_response"]["citations"]), 0)

if __name__ == "__main__":    
    unittest.main()