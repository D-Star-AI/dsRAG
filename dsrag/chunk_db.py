from abc import ABC, abstractmethod
import os
import time
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
    def get_document(self, doc_id: str) -> dict:
        """
        Retrieve all chunks from a given document ID.
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
    def get_all_doc_ids(self, supp_id: str = None) -> list:
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
    
    def get_document(self, doc_id: str, include_content: bool = False) -> dict:
        if doc_id in self.data:
            document = self.data[doc_id]
            formatted_document = {
                'id': doc_id,
                'title': document[0].get('document_title', "")
            }
            
            if include_content:
                # Concatenate the chunks into a single string
                full_document_string = ""
                for chunk_index, chunk in document.items():
                    # Join each chunk text with a new line character
                    full_document_string += chunk['chunk_text'] + "\n"
                formatted_document["content"] = full_document_string
            
            return formatted_document
        
        else:
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
    
    def get_all_doc_ids(self, supp_id: str = None) -> list:
        doc_ids = list(self.data.keys())
        if supp_id:
            doc_ids = [doc_id for doc_id in doc_ids if self.data[doc_id][0].get('supp_id', '') == supp_id]
        return doc_ids

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

    def __init__(self, kb_id: str, storage_directory: str = '~/dsRAG'):
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)
        os.makedirs(os.path.join(self.storage_directory, 'chunk_storage'), exist_ok=True)
        self.db_path = os.path.join(self.storage_directory, 'chunk_storage')

        # Create a table for this kb_id if it doesn't exist
        conn = sqlite3.connect(os.path.join(self.db_path, f'{kb_id}.db'))
        c = conn.cursor()
        result = c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        if not result.fetchone():
            # Create a table for this kb_id
            c.execute(f"CREATE TABLE documents (doc_id TEXT, document_title TEXT, document_summary TEXT, section_title TEXT, section_summary TEXT, chunk_text TEXT, chunk_index INT, created_on TEXT, supp_id TEXT)")
            conn.commit()
        else:
            # Check if we need to add the columns to the table for the supp_id and created_on fields
            c.execute("PRAGMA table_info(documents)")
            columns = c.fetchall()
            column_names = [column[1] for column in columns]
            if 'supp_id' not in column_names:
                c.execute("ALTER TABLE documents ADD COLUMN supp_id TEXT")
            if 'created_on' not in column_names:
                c.execute("ALTER TABLE documents ADD COLUMN created_on TEXT")
        conn.close()
        

    def add_document(self, doc_id: str, chunks: dict[dict]):
        # Add the docs to the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        # Create a created on timestamp
        created_on = str(int(time.time()))

        # Get the data from the dictionary
        for chunk_index, chunk in chunks.items():
            document_title = chunk.get('document_title', "")
            document_summary = chunk.get('document_summary', "")
            section_title = chunk.get('section_title', "")
            section_summary = chunk.get('section_summary', "")
            chunk_text = chunk.get('chunk_text', "")
            supp_id = chunk.get('supp_id', "")
            c.execute(f"INSERT INTO documents (doc_id, document_title, document_summary, section_title, section_summary, chunk_text, chunk_index, created_on, supp_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (doc_id, document_title, document_summary, section_title, section_summary, chunk_text, chunk_index, created_on, supp_id))

        conn.commit()
        conn.close()

    def remove_document(self, doc_id: str):
        # Remove the docs from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"DELETE FROM documents WHERE doc_id='{doc_id}'")
        conn.commit()
        conn.close()
    
    def get_document(self, doc_id: str, include_content: bool = False) -> dict:
        # Retrieve the document from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        columns = ["doc_id", "document_title", "document_summary", "created_on"]
        if include_content:
            columns += ["chunk_text", "chunk_index"]

        query_statement = f"SELECT {', '.join(columns)} FROM documents WHERE doc_id='{doc_id}'"
        c.execute(query_statement)
        results = c.fetchall()
        conn.close()

        # If there are no results, return an empty dictionary
        if not results:
            return None

        formatted_results = {}
        # Turn the results into an object where the columns are keys
        if include_content:
            # Concatenate the chunks into a single string
            full_document_string = ""
            for result in results:
                # Join each chunk text with a new line character
                full_document_string += result[4] + "\n"
            formatted_results["content"] = full_document_string

        formatted_results["id"] = doc_id
        formatted_results["created_on"] = results[0][3]
        formatted_results["document_title"] = results[0][1]
        formatted_results["document_summary"] = results[0][2]

        return formatted_results

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the chunk text from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"SELECT chunk_text FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_document_title(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the document title from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"SELECT document_title FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_document_summary(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the document summary from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"SELECT document_summary FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_section_title(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the section title from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"SELECT section_title FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_section_summary(self, doc_id: str, chunk_index: int) -> str:
        # Retrieve the section summary from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        c.execute(f"SELECT section_summary FROM documents WHERE doc_id='{doc_id}' AND chunk_index={chunk_index}")
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return ""

    def get_all_doc_ids(self, supp_id: str = None) -> list:
        # Retrieve all document IDs from the sqlite table
        conn = sqlite3.connect(os.path.join(self.db_path, f'{self.kb_id}.db'))
        c = conn.cursor()
        query_statement = f"SELECT DISTINCT doc_id FROM documents"
        if supp_id:
            query_statement += f" WHERE supp_id='{supp_id}'"
        c.execute(query_statement)
        results = c.fetchall()
        conn.close()
        return [result[0] for result in results]

    def delete(self):
        # Delete the sqlite database
        if os.path.exists(os.path.join(self.db_path, f'{self.kb_id}.db')):
            os.remove(os.path.join(self.db_path, f'{self.kb_id}.db'))

    def to_dict(self):
        return {
            **super().to_dict(),
            'kb_id': self.kb_id,
            'storage_directory': self.storage_directory
        }