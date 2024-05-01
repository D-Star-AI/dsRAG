import sys
import os
from openai import OpenAI
import unittest

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.embedding import Embedding
from sprag.knowledge_base import KnowledgeBase

class CustomEmbedding(Embedding):
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 768):
        super().__init__(dimension)
        self.model = model
        self.client = OpenAI()

    def get_embeddings(self, text, input_type=None):
        response = self.client.embeddings.create(input=text, model=self.model, dimensions=int(self.dimension))
        embeddings = [embedding_item.embedding for embedding_item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model
        })
        return base_dict
    

class TestCustomEmbeddingSubclass(unittest.TestCase):
    def cleanup(self):
        kb = KnowledgeBase(kb_id="test_kb", exists_ok=True)
        kb.delete()

    def test_custom_embedding_subclass(self):
        # cleanup the knowledge base
        self.cleanup()

        # initialize a KnowledgeBase object with the custom embedding
        kb_id = "test_kb"
        kb = KnowledgeBase(kb_id, embedding_model=CustomEmbedding(model="text-embedding-3-large", dimension=1024), exists_ok=False)

        # load the knowledge base
        kb = KnowledgeBase(kb_id, exists_ok=True)

        # verify that the embedding model is set correctly
        self.assertEqual(kb.embedding_model.model, "text-embedding-3-large")
        self.assertEqual(kb.embedding_model.dimension, 1024)

        # delete the knowledge base
        self.cleanup()

if __name__ == "__main__":
    unittest.main()