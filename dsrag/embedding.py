import os
from abc import ABC, abstractmethod
from typing import Optional
from dsrag.database.vector.types import Vector
from dsrag.utils.imports import openai, cohere, voyageai, ollama


dimensionality = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "voyage-large-2": 1536,
    "voyage-law-2": 1024,
    "voyage-code-2": 1536,
    "llama2": 4096,
    "llama3": 4096,
    "all-minilm": 384,
    "nomic-embed-text": 768,
}


class Embedding(ABC):
    subclasses = {}

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {"subclass_name": self.__class__.__name__, "dimension": self.dimension}

    @classmethod
    def from_dict(cls, config) -> "Embedding":
        subclass_name = config.pop(
            "subclass_name", None
        )  # Remove subclass_name from config
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @abstractmethod
    def get_embeddings(self, text: list[str], input_type: Optional[str]) -> list[Vector]:
        pass


class OpenAIEmbedding(Embedding):
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 768):
        """
        Only v3 models are supported.
        """
        super().__init__(dimension)
        self.model = model
        base_url = os.environ.get("DSRAG_OPENAI_BASE_URL", None)
        if base_url is not None:
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def get_embeddings(self, text: list[str], input_type: Optional[str] = None) -> list[Vector]:
        response = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self.dimension
        )
        embeddings = [embedding_item.embedding for embedding_item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict


class CohereEmbedding(Embedding):
    def __init__(self, model: str = "embed-english-v3.0", dimension: Optional[int] = None):
        super().__init__()

        self.model = model
        base_url = os.environ.get("DSRAG_COHERE_BASE_URL", None)
        if base_url is not None:
            self.client = cohere.Client(api_key=os.environ["CO_API_KEY"], base_url=base_url)
        else:
            self.client = cohere.Client(api_key=os.environ["CO_API_KEY"])

        # Set dimension if not provided
        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(
                    f"Dimension for model {model} is unknown. Please provide the dimension manually."
                )
        else:
            self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: Optional[str]):
        if input_type == "query":
            input_type = "search_query"
        elif input_type == "document":
            input_type = "search_document"
        response = self.client.embed(
            texts=[text] if isinstance(text, str) else text,
            input_type=input_type,
            model=self.model,
        )
        return response.embeddings[0] if isinstance(text, str) else response.embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict


class VoyageAIEmbedding(Embedding):
    def __init__(self, model: str = "voyage-large-2", dimension: Optional[int] = None):
        super().__init__()
        self.model = model
        self.client = voyageai.Client()

        # Set dimension if not provided
        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(
                    f"Dimension for model {model} is unknown. Please provide the dimension manually."
                )
        else:
            self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: Optional[str]):
        response = self.client.embed(
            texts=[text] if isinstance(text, str) else text,
            model=self.model,
            input_type=input_type,
        )
        return response.embeddings[0] if isinstance(text, str) else response.embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict


class OllamaEmbedding(Embedding):

    def __init__(
        self,
        model: str = "llama3",
        dimension: Optional[int] = None,
        client: "ollama.Client" = None,
    ):
        super().__init__(dimension)
        self.model = model
        self.client = client or ollama.Client()
        ollama.pull(model)

        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(
                    f"Dimension for model {model} is unknown. Please provide the dimension manually."
                )
        else:
            self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: Optional[str]):
        if isinstance(text, list):
            responses = []
            for text in text:
                response = self.client.embeddings(model=self.model, prompt=text)
                responses.append(response["embedding"])
            return responses
        else:
            response = self.client.embeddings(model=self.model, prompt=text)
            return response["embedding"]

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict