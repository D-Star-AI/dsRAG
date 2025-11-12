from abc import ABC, abstractmethod
from typing import Any, Optional

from dsrag.database.chunk.types import FormattedDocument


class ChunkDB(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            "subclass_name": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config) -> "ChunkDB":
        subclass_name = config.pop(
            "subclass_name", None
        )  # Remove subclass_name from config
        cls._import_subclass(subclass_name)  # Attempt to import the subclass
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @classmethod
    def _import_subclass(cls, class_name: str):
        """Try to import a subclass by name."""
        import_map = {
            'BasicChunkDB': 'dsrag.database.chunk.basic_db',
            'DynamoDB': 'dsrag.database.chunk.dynamo_db',
            'PostgresChunkDB': 'dsrag.database.chunk.postgres_db',
            'SQLiteDB': 'dsrag.database.chunk.sqlite_db'
        }
        
        if class_name in import_map:
            module_path = import_map[class_name]
            try:
                __import__(module_path)
            except ImportError as e:
                raise ImportError(f"Failed to import {module_path} for {class_name}: {e}")
    
    @abstractmethod
    def add_document(self, doc_id: str, chunks: dict[int, dict[str, Any]], supp_id: str = "", metadata: dict = {}) -> None:
        """
        Store all chunks for a given document.
        """
        pass

    @abstractmethod
    def remove_document(self, doc_id: str) -> None:
        """
        Remove all chunks and metadata associated with a given document ID.
        """
        pass

    @abstractmethod
    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_is_visual(self, doc_id: str, chunk_index: int) -> Optional[bool]:
        """
        Retrieve the is_visual flag of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Optional[tuple[int, int]]:
        """
        Retrieve the page numbers of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[FormattedDocument]:
        """
        Retrieve all chunks from a given document ID.
        """
        pass

    @abstractmethod
    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the document title of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the document summary of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the section title of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the section summary of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> list[str]:
        """
        Retrieve all document IDs.
        """
        pass

    @abstractmethod
    def delete(self) -> None:
        """
        Delete the chunk database.
        """
        pass
