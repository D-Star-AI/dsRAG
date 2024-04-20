import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sprag.embedding import OpenAIEmbedding, CohereEmbedding, VoyageAIEmbedding

def test_get_embeddings_openai():
    input_text = "Hello, world!"
    model = "text-embedding-3-small"
    dimension = 768
    embedding_provider = OpenAIEmbedding(model, dimension)
    embedding = embedding_provider.get_embeddings(input_text)
    assert len(embedding) == 768

def test_get_embeddings_cohere():
    input_text = "Hello, world!"
    model = "embed-english-v3.0"
    embedding_provider = CohereEmbedding(model)
    embedding = embedding_provider.get_embeddings(input_text, input_type="query")
    assert len(embedding) == 1024

def test_get_embeddings_voyage():
    input_text = "Hello, world!"
    model = "voyage-large-2"
    embedding_provider = VoyageAIEmbedding(model)
    embedding = embedding_provider.get_embeddings(input_text, input_type="query")
    assert len(embedding) == 1536

def test_get_embeddings_openai_with_list():
    input_texts = ["Hello, world!", "Goodbye, world!"]
    model = "text-embedding-3-small"
    dimension = 768
    embedding_provider = OpenAIEmbedding(model, dimension)
    embeddings = embedding_provider.get_embeddings(input_texts)
    assert len(embeddings) == 2
    assert all(len(embed) == dimension for embed in embeddings)

def test_get_embeddings_cohere_with_list():
    input_texts = ["Hello, world!", "Goodbye, world!"]
    model = "embed-english-v3.0"
    embedding_provider = CohereEmbedding(model)
    embeddings = embedding_provider.get_embeddings(input_texts, input_type="query")
    assert len(embeddings) == 2
    assert all(len(embed) == 1024 for embed in embeddings)

def test_get_embeddings_voyage_with_list():
    input_texts = ["Hello, world!", "Goodbye, world!"]
    model = "voyage-large-2"
    embedding_provider = VoyageAIEmbedding(model)
    embeddings = embedding_provider.get_embeddings(input_texts, input_type="query")
    assert len(embeddings) == 2
    assert all(len(embed) == 1536 for embed in embeddings)


if __name__ == "__main__":
    # run tests
    test_get_embeddings_openai()
    test_get_embeddings_cohere()
    test_get_embeddings_voyage()
    test_get_embeddings_openai_with_list()
    test_get_embeddings_cohere_with_list()
    test_get_embeddings_voyage_with_list()