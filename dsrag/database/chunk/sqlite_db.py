import os
import time
import sqlite3
from typing import Any, Optional

from dsrag.database.chunk.db import ChunkDB
from dsrag.database.chunk.types import FormattedDocument


class SQLiteDB(ChunkDB):

    def __init__(self, kb_id: str, storage_directory: str = "~/dsRAG") -> None:
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)
        os.makedirs(
            os.path.join(self.storage_directory, "chunk_storage"), exist_ok=True
        )
        self.db_path = os.path.join(self.storage_directory, "chunk_storage")

        # Create a table for this kb_id if it doesn't exist
        conn = sqlite3.connect(os.path.join(self.db_path, f"{kb_id}.db"))
        c = conn.cursor()
        result = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        if not result.fetchone():
            # Create a table for this kb_id
            c.execute(
                "CREATE TABLE documents (doc_id TEXT, document_title TEXT, document_summary TEXT, section_title TEXT, section_summary TEXT, chunk_text TEXT, chunk_index INT, created_on TEXT, supp_id TEXT)"
            )
            conn.commit()
        else:
            # Check if we need to add the columns to the table for the supp_id and created_on fields
            c.execute("PRAGMA table_info(documents)")
            columns = c.fetchall()
            column_names = [column[1] for column in columns]
            if "supp_id" not in column_names:
                c.execute("ALTER TABLE documents ADD COLUMN supp_id TEXT")
            if "created_on" not in column_names:
                c.execute("ALTER TABLE documents ADD COLUMN created_on TEXT")
        conn.close()

    def add_document(self, doc_id: str, chunks: dict[int, dict[str, Any]]) -> None:
        # Add the docs to the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        # Create a created on timestamp
        created_on = str(int(time.time()))

        # Get the data from the dictionary
        for chunk_index, chunk in chunks.items():
            document_title = chunk.get("document_title", "")
            document_summary = chunk.get("document_summary", "")
            section_title = chunk.get("section_title", "")
            section_summary = chunk.get("section_summary", "")
            chunk_text = chunk.get("chunk_text", "")
            supp_id = chunk.get("supp_id", "")
            c.execute(
                "INSERT INTO documents (doc_id, document_title, document_summary, section_title, section_summary, chunk_text, chunk_index, created_on, supp_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    document_title,
                    document_summary,
                    section_title,
                    section_summary,
                    chunk_text,
                    chunk_index,
                    created_on,
                    supp_id,
                ),
            )

        conn.commit()
        conn.close()

    def remove_document(self, doc_id: str) -> None:
        # Remove the docs from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(f"DELETE FROM documents WHERE doc_id='{doc_id}'")
        conn.commit()
        conn.close()

    def get_document(
        self, doc_id: str, include_content: bool = False
    ) -> FormattedDocument | None:
        # Retrieve the document from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        columns = ["doc_id", "document_title", "document_summary", "created_on"]
        if include_content:
            columns += ["chunk_text", "chunk_index"]

        query_statement = (
            f"SELECT {', '.join(columns)} FROM documents WHERE doc_id='{doc_id}'"
        )
        c.execute(query_statement)
        results = c.fetchall()
        conn.close()

        # If there are no results, return an empty dictionary
        if not results:
            return None

        # Turn the results into an object where the columns are keys
        full_document_string = ""
        if include_content:
            # Concatenate the chunks into a single string
            for result in results:
                # Join each chunk text with a new line character
                full_document_string += result[4] + "\n"

        title = results[0][1]
        created_on = results[0][3]
        title = results[0][1]
        summary = results[0][2]

        return FormattedDocument(
            id=doc_id,
            title=title,
            content=full_document_string if include_content else None,
            summary=summary,
            created_on=created_on,
        )

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the chunk text from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(
            f"SELECT chunk_text FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}"
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the document title from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(
            f"SELECT document_title FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}"
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the document summary from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(
            f"SELECT document_summary FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}"
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the section title from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(
            f"SELECT section_title FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}"
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the section summary from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(
            f"SELECT section_summary FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}"
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> list[str]:
        # Retrieve all document IDs from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        query_statement = "SELECT DISTINCT doc_id FROM documents"
        if supp_id:
            query_statement += f" WHERE supp_id='{supp_id}'"
        c.execute(query_statement)
        results = c.fetchall()
        conn.close()
        return [result[0] for result in results]

    def delete(self) -> None:
        # Delete the sqlite database
        if os.path.exists(os.path.join(self.db_path, f"{self.kb_id}.db")):
            os.remove(os.path.join(self.db_path, f"{self.kb_id}.db"))

    def to_dict(self) -> dict[str, str]:
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            "storage_directory": self.storage_directory,
        }
