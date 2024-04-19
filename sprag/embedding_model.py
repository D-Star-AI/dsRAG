import os
from abc import ABC, abstractmethod
from openai import OpenAI
import cohere
import voyageai


dimensionality = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "voyage-large-2": 1536,
    "voyage-law-2": 1024,
    "voyage-code-2": 1536,
}

class Embedding(ABC):
    def __init__(self, model, dimension=None):
        self.model = model
        self.dimension = dimension

    @abstractmethod
    def get_embeddings(self, text, input_type=None):
        pass

class OpenAIEmbedding(Embedding):
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 768):
        """
        Only v3 models are supported.
        """
        super().__init__(model, dimension)
        self.client = OpenAI()

    def get_embeddings(self, text, input_type=None):
        response = self.client.embeddings.create(input=text, model=self.model, dimensions=int(self.dimension))
        embeddings = [embedding_item.embedding for embedding_item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings

class CohereEmbedding(Embedding):
    def __init__(self, model: str = "embed-english-v3.0", dimension: int = None):
        super().__init__(model)
        self.client = cohere.Client(os.environ['COHERE_API_KEY'])
        
        # Set dimension if not provided
        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(f"Dimension for model {model} is unknown. Please provide the dimension manually.")

    def get_embeddings(self, text, input_type=None):
        if input_type == "query":
            input_type = "search_query"
        elif input_type == "document":
            input_type = "search_document"
        response = self.client.embed(texts=[text] if isinstance(text, str) else text, input_type=input_type, model=self.model)
        return response.embeddings[0] if isinstance(text, str) else response.embeddings

class VoyageAIEmbedding(Embedding):
    def __init__(self, model: str = "voyage-large-2", dimension: int = None):
        super().__init__(model)

        # Set dimension if not provided
        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(f"Dimension for model {model} is unknown. Please provide the dimension manually.")

    def get_embeddings(self, text, input_type=None):
        if isinstance(text, str):
            return voyageai.get_embedding(text, model=self.model, input_type=input_type)
        else:
            return voyageai.get_embeddings(text, model=self.model, input_type=input_type)


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