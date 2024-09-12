from abc import ABC, abstractmethod
from typing import Sequence, Optional
from dsrag.database.vector.types import ChunkMetadata, Vector, VectorSearchResult


class VectorDB(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            "subclass_name": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config):
        subclass_name = config.pop(
            "subclass_name", None
        )  # Remove subclass_name from config
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @abstractmethod
    def add_vectors(
        self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]
    ) -> None:
        """
        Store a list of vectors with associated metadata.
        """
        pass

    @abstractmethod
    def remove_document(self, doc_id) -> None:
        """
        Remove all vectors and metadata associated with a given document ID.
        """
        pass

    @abstractmethod
    def search(self, query_vector, top_k: int=10, metadata_filter: Optional[dict] = None) -> list[VectorSearchResult]:
        """
        Retrieve the top-k closest vectors to a given query vector.
        - needs to return results as list of dictionaries in this format:
        {
            'metadata':
                {
                    'doc_id': doc_id,
                    'chunk_index': chunk_index,
                    'chunk_header': chunk_header,
                    'chunk_text': chunk_text
                },
            'similarity': similarity,
        }
        """
        pass

    @abstractmethod
    def delete(self) -> None:
        """
        Delete the vector database.
        """
        pass
