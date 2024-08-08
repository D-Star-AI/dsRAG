from abc import ABC, abstractmethod
import os
import pickle
import sqlite3

class ChunkDB(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            'subclass_name': self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config):
        subclass_name = config.pop('subclass_name', None)  # Remove subclass_name from config
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @abstractmethod
    def add_document(self, doc_id: str, chunks: dict[dict]):
        """
        Store all chunks for a given document.
        """
        pass

    @abstractmethod
    def remove_document(self, doc_id: str):
        """
        Remove all chunks and metadata associated with a given document ID.
        """
        pass

    @abstractmethod
    def get_chunk_text(self, doc_id: str, chunk_index: int) -> dict:
        """
        Retrieve a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_document_title(self, doc_id: str, chunk_index: int) -> str:
        """
        Retrieve the document title of a specific chunk from a given document ID.
        """
        pass

    def get_document_summary(self, doc_id: str, chunk_index: int) -> str:
        """
        Retrieve the document summary of a specific chunk from a given document ID.
        """
        pass

    def get_section_title(self, doc_id: str, chunk_index: int) -> str:
        """
        Retrieve the section title of a specific chunk from a given document ID.
        """
        pass

    def get_section_summary(self, doc_id: str, chunk_index: int) -> str:
        """
        Retrieve the section summary of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_all_doc_ids(self) -> list:
        """
        Retrieve all document IDs.
        """
        pass

    @abstractmethod
    def delete(self):
        """
        Delete the chunk database.
        """
        pass


class BasicChunkDB(ChunkDB):
    """
    This is a basic implementation of a ChunkDB that stores chunks in a nested dictionary and persists them to disk by pickling the dictionary.
    """
    def __init__(self, kb_id: str, storage_directory: str = '~/dsRAG'):
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)  # Expand the user path
        # Ensure the base directory and the chunk storage directory exist
        os.makedirs(os.path.join(self.storage_directory, 'chunk_storage'), exist_ok=True)
        self.storage_path = os.path.join(self.storage_directory, 'chunk_storage', f'{kb_id}.pkl')
        self.load()

    def add_document(self, doc_id: str, chunks: dict[dict]):
        self.data[doc_id] = chunks
        self.save()

    def remove_document(self, doc_id: str):
        self.data.pop(doc_id, None)
        self.save()

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> str:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            return self.data[doc_id][chunk_index]['chunk_text']
        return None

    def get_document_title(self, doc_id: str, chunk_index: int) -> str:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if 'document_title' in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]['document_title']
            else:
                return ""
        return None
    
    def get_document_summary(self, doc_id: str, chunk_index: int) -> str:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if 'document_summary' in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]['document_summary']
            else:
                return ""
        return None
    
    def get_section_title(self, doc_id: str, chunk_index: int) -> str:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if 'section_title' in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]['section_title']
            else:
                return ""
        return None
    
    def get_section_summary(self, doc_id: str, chunk_index: int) -> str:
        if doc_id in self.data and chunk_index in self.data[doc_id]:
            if 'section_summary' in self.data[doc_id][chunk_index]:
                return self.data[doc_id][chunk_index]['section_summary']
            else:
                return ""
        return None
    
    def get_all_doc_ids(self) -> list:
        return list(self.data.keys())

    def load(self):
        try:
            with open(self.storage_path, 'rb') as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            self.data = {}

    def save(self):
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.data, f)
    
    def delete(self):
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def to_dict(self):
        return {
            **super().to_dict(),
            'kb_id': self.kb_id,
            'storage_directory': self.storage_directory,
        }
    

class SQLiteDB(ChunkDB):

    def __init__(self, kb_id: str, title: str = "", description: str = "", language: str = "en", db_path: str = '~/dsRAG'):
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(db_path)
        os.makedirs(os.path.join(self.storage_directory, 'chunk_storage'), exist_ok=True)
        self.db_path = os.path.join(self.storage_directory, 'chunk_storage')

        # Make sure main knowledge bases table exists
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        result = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_bases'")
        if not result.fetchone():
            c.execute("CREATE TABLE knowledge_bases (kb_id VARCHAR(256) PRIMARY KEY, title VARCHAR(256), description TEXT, language VARCHAR(16))")
            conn.commit()
        conn.close()

        # Create a table for this kb_id if it doesn't exist
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        result = c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{kb_id}'")
        if not result.fetchone():
            # Insert this kb into the table
            c.execute("INSERT INTO knowledge_bases (kb_id, title, description, language) VALUES (?, ?, ?, ?)", (kb_id, title, description, language))
            c.execute(f"CREATE TABLE {kb_id} (doc_id VARCHAR(256), document_title VARCHAR(256), document_summary TEXT, section_title VARCHAR(256), section_summary TEXT, chunk_text TEXT, chunk_index INT)")
            conn.commit()
        conn.close()
        

    def add_document(self, doc_id: str, chunks: dict[dict]):
        # Add the docs to the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        # Get the data from the dictionary
        for chunk_index, chunk in chunks.items():
            document_title = chunk.get('document_title', "")
            document_summary = chunk.get('document_summary', "")
            section_title = chunk.get('section_title', "")
            section_summary = chunk.get('section_summary', "")
            chunk_text = chunk.get('chunk_text', "")
            c.execute(f"INSERT INTO {self.kb_id} (doc_id, document_title, document_summary, section_title, section_summary, chunk_text, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)", (doc_id, document_title, document_summary, section_title, section_summary, chunk_text, chunk_index))

        conn.commit()
        conn.close()

    def remove_document(self, doc_id: str):
        # Remove the docs from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        c.execute(f"DELETE FROM {self.kb_id} WHERE doc_id='{doc_id}'")
        conn.commit()
        conn.close()

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the chunk text from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        c.execute(f"SELECT chunk_text FROM {self.kb_id} WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_document_title(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the document title from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        c.execute(f"SELECT document_title FROM {self.kb_id} WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_document_summary(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the document summary from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        c.execute(f"SELECT document_summary FROM {self.kb_id} WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_section_title(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the section title from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        c.execute(f"SELECT section_title FROM {self.kb_id} WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_section_summary(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the section summary from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        c.execute(f"SELECT section_summary FROM {self.kb_id} WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_all_doc_ids(self) -> list:
        # Retrieve all document IDs from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, 'dsRAG.db'))
        c = conn.cursor()
        c.execute(f"SELECT DISTINCT doc_id FROM {self.kb_id}")
        results = c.fetchall()
        conn.close()
        return [result[0] for result in results]

    def to_dict(self):
        return {
            **super().to_dict(),
            'kb_id': self.kb_id,
            'db_path': self.db_path,
            'title': self.title,
            'description': self.description,
            'language': self.language,
        }