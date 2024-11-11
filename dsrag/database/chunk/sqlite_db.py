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
        self.columns = [
            {"name": "doc_id", "type": "TEXT"},
            {"name": "document_title", "type": "TEXT"},
            {"name": "document_summary", "type": "TEXT"},
            {"name": "section_title", "type": "TEXT"},
            {"name": "section_summary", "type": "TEXT"},
            {"name": "chunk_text", "type": "TEXT"},
            {"name": "chunk_index", "type": "INT"},
            {"name": "chunk_length", "type": "INT"},
            {"name": "chunk_page_start", "type": "INT"},
            {"name": "chunk_page_end", "type": "INT"},
            {"name": "is_visual", "type": "BOOLEAN"},
            {"name": "created_on", "type": "TEXT"},
            {"name": "supp_id", "type": "TEXT"},
            {"name": "metadata", "type": "TEXT"},
        ]

        # Create a table for this kb_id if it doesn't exist
        conn = sqlite3.connect(os.path.join(self.db_path, f"{kb_id}.db"))
        c = conn.cursor()
        result = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        if not result.fetchone():
            # Create a table for this kb_id
            query_statement = "CREATE TABLE documents ("
            for column in self.columns:
                query_statement += f"{column['name']} {column['type']}, "
            query_statement = query_statement[:-2] + ")"
            c.execute(query_statement)
            conn.commit()
        else:
            # Check if we need to add any columns to the table. This happens if the columns have been updated
            c.execute("PRAGMA table_info(documents)")
            columns = c.fetchall()
            column_names = [column[1] for column in columns]
            for column in self.columns:
                if column["name"] not in column_names:
                    # Add the column to the table
                    c.execute("ALTER TABLE documents ADD COLUMN {} {}".format(column["name"], column["type"]))
        conn.close()

    def add_document(self, doc_id: str, chunks: dict[int, dict[str, Any]], supp_id: str = "", metadata: dict = {}) -> None:
        # Add the docs to the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        # Create a created on timestamp
        created_on = str(int(time.time()))

        # Turn the metadata object into a string
        metadata = str(metadata)

        # Get the data from the dictionary
        for chunk_index, chunk in chunks.items():
            chunk_text = chunk.get("chunk_text", "")
            chunk_length = len(chunk_text)

            values_dict = {
                'doc_id': doc_id,
                'document_title': chunk.get("document_title", ""),
                'document_summary': chunk.get("document_summary", ""),
                'section_title': chunk.get("section_title", ""),
                'section_summary': chunk.get("section_summary", ""),
                'chunk_text': chunk.get("chunk_text", ""),
                'chunk_page_start': chunk.get("chunk_page_start", None),
                'chunk_page_end': chunk.get("chunk_page_end", None),
                'is_visual': chunk.get("is_visual", False),
                'chunk_index': chunk_index,
                'chunk_length': chunk_length,
                'created_on': created_on,
                'supp_id': supp_id,
                'metadata': metadata
            }

            # Generate the column names and placeholders
            columns = ', '.join(values_dict.keys())
            placeholders = ', '.join(['?'] * len(values_dict))

            sql = f"INSERT INTO documents ({columns}) VALUES ({placeholders})"

            c.execute(sql, tuple(values_dict.values()))

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
    ) -> Optional[FormattedDocument]:
        # Retrieve the document from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        columns = ["supp_id", "document_title", "document_summary", "created_on", "metadata"]
        if include_content:
            columns += ["chunk_text", "chunk_index"]

        query_statement = (
            f"SELECT {', '.join(columns)} FROM documents WHERE doc_id='{doc_id}'"
        )
        c.execute(query_statement)
        results = c.fetchall()
        conn.close()

        # If there are no results, return None
        if not results:
            return None

        # Turn the results into an object where the columns are keys
        full_document_string = ""
        if include_content:
            # Concatenate the chunks into a single string
            for result in results:
                # Join each chunk text with a new line character
                full_document_string += result[columns.index("chunk_text")] + "\n"
            # Remove the last new line character
            full_document_string = full_document_string[:-1]

        supp_id = results[0][columns.index("supp_id")]
        title = results[0][columns.index("document_title")]
        summary = results[0][columns.index("document_summary")]
        created_on = results[0][columns.index("created_on")]
        metadata = results[0][columns.index("metadata")]

        # Convert the metadata string back into a dictionary
        if metadata:
            metadata = eval(metadata)

        return FormattedDocument(
            id=doc_id,
            supp_id=supp_id,
            title=title,
            content=full_document_string if include_content else None,
            summary=summary,
            created_on=created_on,
            metadata=metadata
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
    
    def get_is_visual(self, doc_id: str, chunk_index: int) -> Optional[bool]:
        # Retrieve the is_visual param from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(
            f"SELECT is_visual FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}"
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None
    
    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Optional[tuple[int, int]]:
        # Retrieve the chunk page numbers from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f"{self.kb_id}.db"))
        c = conn.cursor()
        c.execute(
            f"SELECT chunk_page_start, chunk_page_end FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}"
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result
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
    
    def get_document_count(self) -> int:
        # Retrieve the number of documents in the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"SELECT COUNT(DISTINCT doc_id) FROM documents")
        result = c.fetchone()
        conn.close()
        if result is None:
            return 0
        return result[0]

    def get_total_num_characters(self) -> int:
        # Retrieve the total number of characters in the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"SELECT SUM(chunk_length) FROM documents")
        result = c.fetchone()
        conn.close()
        if result is None or result[0] is None:
            return 0
        return result[0]

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
