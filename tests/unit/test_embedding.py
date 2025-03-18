import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dsrag.embedding import (
    OpenAIEmbedding,
    CohereEmbedding,
    OllamaEmbedding,
    Embedding,
)


class TestEmbedding(unittest.TestCase):
    def test__get_embeddings_openai(self):
        input_text = "Hello, world!"
        model = "text-embedding-3-small"
        dimension = 768
        embedding_provider = OpenAIEmbedding(model, dimension)
        embedding = embedding_provider.get_embeddings(input_text)
        self.assertEqual(len(embedding), dimension)

    def test__get_embeddings_cohere(self):
        input_text = "Hello, world!"
        model = "embed-english-v3.0"
        embedding_provider = CohereEmbedding(model)
        embedding = embedding_provider.get_embeddings(input_text, input_type="query")
        self.assertEqual(len(embedding), 1024)

    def test__get_embeddings_ollama(self):
        input_text = "Hello, world!"
        model = "llama3"
        dimension = 4096
        try:
            embedding_provider = OllamaEmbedding(model, dimension)
            embedding = embedding_provider.get_embeddings(input_text)
        except Exception as e:
            print (e)
            if e.__class__.__name__ == "ConnectError":
                print("Connection failed")
            return
        self.assertEqual(len(embedding), dimension)

    def test__get_embeddings_openai_with_list(self):
        input_texts = ["Hello, world!", "Goodbye, world!"]
        model = "text-embedding-3-small"
        dimension = 768
        embedding_provider = OpenAIEmbedding(model, dimension)
        embeddings = embedding_provider.get_embeddings(input_texts)
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(len(embed) == dimension for embed in embeddings))

    def test__get_embeddings_cohere_with_list(self):
        input_texts = ["Hello, world!", "Goodbye, world!"]
        model = "embed-english-v3.0"
        embedding_provider = CohereEmbedding(model)
        embeddings = embedding_provider.get_embeddings(input_texts, input_type="query")
        assert len(embeddings) == 2
        assert all(len(embed) == 1024 for embed in embeddings)
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(len(embed) == 1024 for embed in embeddings))

    def test__get_embeddings_ollama_with_list_llama2(self):
        input_texts = ["Hello, world!", "Goodbye, world!"]
        model = "llama2"
        dimension = 4096
        try:
            embedding_provider = OllamaEmbedding(model, dimension)
            embeddings = embedding_provider.get_embeddings(input_texts)
        except Exception as e:
            if e.__class__.__name__ == "ConnectError":
                print ("Connection failed")
                print (e)
                return
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(len(embed) == dimension for embed in embeddings))

    def test__get_embeddings_ollama_with_list_minilm(self):
        input_texts = ["Hello, world!", "Goodbye, world!"]
        model = "all-minilm"
        dimension = 384
        try:
            embedding_provider = OllamaEmbedding(model, dimension)
            embeddings = embedding_provider.get_embeddings(input_texts)
        except Exception as e:
            print (e)
            if e.__class__.__name__ == "ConnectError":
                print ("Connection failed")
            return
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(len(embed) == dimension for embed in embeddings))

    def test__get_embeddings_ollama_with_list_nomic(self):
        input_texts = ["Hello, world!", "Goodbye, world!"]
        model = "nomic-embed-text"
        dimension = 768
        try:
            embedding_provider = OllamaEmbedding(model, dimension)
            embeddings = embedding_provider.get_embeddings(input_texts)
        except Exception as e:
            if e.__class__.__name__ == "ConnectError":
                print ("Connection failed")
                print (e)
                return
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(len(embed) == dimension for embed in embeddings))

    def test__initialize_from_config(self):
        config = {
            'subclass_name': 'OpenAIEmbedding',
            'model': 'text-embedding-3-small',
            'dimension': 1024
        }
        embedding_instance = Embedding.from_dict(config)
        self.assertIsInstance(embedding_instance, OpenAIEmbedding)
        self.assertEqual(embedding_instance.model, 'text-embedding-3-small')
        self.assertEqual(embedding_instance.dimension, 1024)


if __name__ == "__main__":
    unittest.main()