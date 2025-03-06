import os
import pickle
from typing import Any, Optional, cast

from dsrag.database.chunk.db import ChunkDB
from dsrag.database.chunk.types import FormattedDocument


class BasicChunkDB(ChunkDB):
    """
    This is a basic implementation of a ChunkDB that stores chunks in a nested dictionary and persists them to disk by pickling the dictionary.
    """

    def __init__(self, kb_id: str, storage_directory: str = "~/dsRAG") -> None:
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(
            storage_directory
        )  # Expand the user path
        # Ensure the base directory and the chunk storage directory exist
        os.makedirs(
            os.path.join(self.storage_directory, "chunk_storage"), exist_ok=True
        )
        self.storage_path = os.path.join(
            self.storage_directory, "chunk_storage", f"{kb_id}.pkl"
        )
        self.load()

    def add_document(self, doc_id: str, chunks: dict[int, dict[str, Any]], supp_id: str = "", metadata: dict = {}) -> None:
        self.data[doc_id] = chunks
        self.save()

    def remove_document(self, doc_id: str):
        self.data.pop(doc_id, None)
        self.save()

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            return self.data[doc_id][chunk_index]["chunk_text"]
        return None
    
    def get_is_visual(self, doc_id: str, chunk_index: int) -> Optional[bool]:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            return self.data[doc_id][chunk_index].get("is_visual", False) # default to False for backwards compatibility
        return None
    
    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Optional[tuple[int, int]]:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            return (
                self.data[doc_id][chunk_index].get("chunk_page_start", None),
                self.data[doc_id][chunk_index].get("chunk_page_end", None),
            )
        return None, None

    def get_document(
        self, doc_id: str, include_content: bool = False
    ) -> Optional[FormattedDocument]:
        if doc_id in self.data:
            document = self.data[doc_id]
            title = cast(str, document[0].get("document_title", ""))

            full_document_string = ""
            if include_content:
                # Concatenate the chunks into a single string
                for _, chunk in document.items():
                    # Join each chunk text with a new line character
                    full_document_string += chunk["chunk_text"] + "\n"

            return FormattedDocument(
                id=doc_id,
                title=title,
                content=full_document_string if include_content else None,
                summary=cast(str, document[0].get("document_summary", "")),
                created_on=None,
                chunk_count=len(document.items())
            )

        else:
            return None

    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if "document_title" in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]["document_title"]
            else:
                return None
        return None

    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if "document_summary" in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]["document_summary"]
            else:
                return None
        return None

    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if "section_title" in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]["section_title"]
            else:
                return None
        return None

    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if "section_summary" in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]["section_summary"]
            else:
                return None
        return None

    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> list[str]:
        doc_ids = list(self.data.keys())
        if supp_id:
            doc_ids = [
                doc_id
                for doc_id in doc_ids
                if self.data[doc_id][0].get("supp_id", "") == supp_id
            ]
        return doc_ids
    
    def get_document_count(self) -> int:
        # Retrieve the number of documents from the length of the data dictionary
        return len(self.data.keys())
    
    def get_total_num_characters(self) -> int:
        total_num_characters = 0
        for _, doc in self.data.items():
            for chunk in doc.values():
                total_num_characters += len(chunk["chunk_text"])
        return total_num_characters

    def load(self):
        try:
            with open(self.storage_path, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            self.data = {}

    def save(self):
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.data, f)

    def delete(self):
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            "storage_directory": self.storage_directory,
        }
