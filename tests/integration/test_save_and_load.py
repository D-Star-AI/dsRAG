import sys
import os
import unittest

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.knowledge_base import KnowledgeBase
from sprag.llm import OpenAIChatAPI, OllamaAPI
from sprag.embedding import VoyageAIEmbedding

class TestSaveAndLoad(unittest.TestCase):
    def cleanup(self):
        kb = KnowledgeBase(kb_id="test_kb", exists_ok=True)
        kb.delete()

    def test_save_and_load(self):
        self.cleanup()

        # initialize a KnowledgeBase object
        auto_context_model = OllamaAPI(model="llama3")
        embedding_model = VoyageAIEmbedding(model="voyage-code-2")
        kb = KnowledgeBase(kb_id="test_kb", auto_context_model=auto_context_model, embedding_model=embedding_model, exists_ok=False)

        # load the KnowledgeBase object
        kb1 = KnowledgeBase(kb_id="test_kb")

        # verify that the KnowledgeBase object has the right parameters
        self.assertEqual(kb1.auto_context_model.model, "llama3")
        self.assertEqual(kb1.embedding_model.model, "voyage-code-2")

        # delete the KnowledgeBase object
        kb1.delete()

if __name__ == "__main__":
    unittest.main()