import sqlite3
import os
import json
from .db import ChatThreadDB
import uuid

class SQLiteChatThreadDB(ChatThreadDB):

    def __init__(self, storage_directory: str = "~/dsRAG"):
        # Check if the directory exists, if not create it
        self.chat_thread_columns = ["thread_id", "supp_id", "kb_ids", "model", "temperature", "system_message", "auto_query_model", "auto_query_guidance", "target_output_length", "max_chat_history_tokens", "rse_params"]
        self.interactions_columns = ["thread_id", "message_id", "user_input", "user_input_timestamp", "model_response", "model_response_timestamp", "relevant_segments", "search_queries", "citations"]
        self.chat_thread_column_types = ["VARCHAR(256) PRIMARY KEY", "TEXT", "TEXT", "TEXT", "REAL", "TEXT", "TEXT", "TEXT", "TEXT", "INTEGER", "TEXT"]
        self.interactions_column_types = ["VARCHAR(256)", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT"]
        self.storage_directory = os.path.expanduser(storage_directory)
        if not os.path.exists(self.storage_directory):
            os.makedirs(self.storage_directory)
        self.db_path = f"{self.storage_directory}/chat_thread.db"
        # Create the chat threads table if it doesn't exist
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        result = c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='chat_threads'")
        if not result.fetchone():
            query_statement = f"CREATE TABLE chat_threads ({', '.join([f'{column} {column_type}' for column, column_type in zip(self.chat_thread_columns, self.chat_thread_column_types)])})"
            print (query_statement)
            c.execute(query_statement)
            # Create the interactions table. The thread_id column is a foreign key that references the thread_id column in the chat_threads table
            query_statement = f"CREATE TABLE interactions ({', '.join([f'{column} {column_type}' for column, column_type in zip(self.interactions_columns, self.interactions_column_types)])}, FOREIGN KEY(thread_id) REFERENCES chat_threads(thread_id))"
            c.execute(query_statement)
            conn.commit()
        conn.close()

    
    def create_chat_thread(self, chat_thread_params: dict) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Convert the kb_ids list to a string
        chat_thread_params["kb_ids"] = ",".join(chat_thread_params["kb_ids"])
        # Convert rse_params dict to JSON string if it exists
        if "rse_params" in chat_thread_params and chat_thread_params["rse_params"]:
            chat_thread_params["rse_params"] = json.dumps(chat_thread_params["rse_params"])
        # Create the chat thread, using the self.chat_thread_columns list to specify the order of the columns
        query_statement = f"INSERT INTO chat_threads ({', '.join(self.chat_thread_columns)}) VALUES ({'?, '.join(['']*len(self.chat_thread_columns))}?)"
        chat_thread_params_tuple = tuple([chat_thread_params.get(column, "") for column in self.chat_thread_columns])
        c.execute(query_statement, chat_thread_params_tuple)

        conn.commit()
        conn.close()
        return chat_thread_params

    def list_chat_threads(self, supp_id: str = None) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        query_statement = f"SELECT {', '.join(self.chat_thread_columns)} FROM chat_threads"
        if supp_id:
            query_statement += " WHERE supp_id = ?"
            c.execute(query_statement, (supp_id,))
        else:
            c.execute(query_statement)
        chat_threads = c.fetchall()
        conn.close()

        # Format the chat threads
        formatted_chat_threads = []
        for chat_thread in chat_threads:
            # Create the chat thread dictionary from the columns list and the chat_thread tuple
            formatted_chat_thread = dict(zip(self.chat_thread_columns, chat_thread))
            # Format the kb_ids string to a list
            formatted_chat_thread["id"] = formatted_chat_thread["thread_id"]
            # Remove the thread_id key
            del formatted_chat_thread["thread_id"]
            formatted_chat_thread["kb_ids"] = formatted_chat_thread["kb_ids"].split(",")
            # Parse rse_params from JSON if it exists and is not empty
            if formatted_chat_thread["rse_params"]:
                try:
                    formatted_chat_thread["rse_params"] = json.loads(formatted_chat_thread["rse_params"])
                except json.JSONDecodeError:
                    formatted_chat_thread["rse_params"] = {}
            else:
                formatted_chat_thread["rse_params"] = {}
            formatted_chat_threads.append(formatted_chat_thread)

        return formatted_chat_threads

    def get_chat_thread(self, thread_id: str) -> dict:
        """
        Returns a formatted chat thread dictionary with the following keys:
        - id: str
        - params: dict
        - interactions: list[dict]
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Get the chat thread by thread_id and the interactions for this thread. The interactions have a foreign key constraint on the thread_id column

        # Get the chat thread
        query_statement = f"SELECT {', '.join(self.chat_thread_columns)} FROM chat_threads WHERE thread_id = ?"
        c.execute(query_statement, (thread_id,))
        chat_thread = c.fetchone()
        if not chat_thread:
            return None
        # Format the chat thread
        chat_thread = dict(zip(self.chat_thread_columns, chat_thread))
        # Format the kb_ids string to a list
        chat_thread["kb_ids"] = chat_thread["kb_ids"].split(",")
        chat_thread["id"] = chat_thread["thread_id"]
        del chat_thread["thread_id"]
        # Parse rse_params from JSON if it exists and is not empty
        if chat_thread["rse_params"]:
            try:
                chat_thread["rse_params"] = json.loads(chat_thread["rse_params"])
            except json.JSONDecodeError:
                chat_thread["rse_params"] = {}
        else:
            chat_thread["rse_params"] = {}
        # Now get the interactions
        query_statement = f"SELECT {', '.join(self.interactions_columns)} FROM interactions WHERE thread_id = ?"
        c.execute(query_statement, (thread_id,))
        interactions = c.fetchall()
        # Format the interactions
        formatted_interactions = []
        for interaction in interactions:
            formatted_interaction = {
                "user_input": {
                    "content": interaction[1],
                    "timestamp": interaction[2]
                },
                "model_response": {
                    "content": interaction[3],
                    "timestamp": interaction[4],
                    "citations": json.loads(interaction[7]) if interaction[7] else []
                },
                "relevant_segments": json.loads(interaction[5]),
                "search_queries": json.loads(interaction[6])
            }
            formatted_interactions.append(formatted_interaction)
        
        formatted_chat_thread = {}
        formatted_chat_thread["interactions"] = formatted_interactions

        # Format the chat thread params
        params = {}
        keys = [key for key in self.chat_thread_columns if key != "thread_id"]
        for key in keys:
            params[key] = chat_thread[key]
        formatted_chat_thread["params"] = params

        # Add the id key
        formatted_chat_thread["id"] = chat_thread["id"]

        conn.close()
        return formatted_chat_thread

    def update_chat_thread(self, thread_id: str, chat_thread_params: dict) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Convert the kb_ids list to a string
        chat_thread_params["kb_ids"] = ",".join(chat_thread_params["kb_ids"])
        # Convert rse_params dict to JSON string if it exists
        if "rse_params" in chat_thread_params and chat_thread_params["rse_params"]:
            chat_thread_params["rse_params"] = json.dumps(chat_thread_params["rse_params"])
        else:
            chat_thread_params["rse_params"] = ""
            
        c.execute("UPDATE chat_threads SET supp_id = ?, kb_ids = ?, model = ?, temperature = ?, system_message = ?, auto_query_guidance = ?, target_output_length = ?, max_chat_history_tokens = ?, rse_params = ? WHERE thread_id = ?", 
                 (chat_thread_params["supp_id"], chat_thread_params["kb_ids"], chat_thread_params["model"], chat_thread_params["temperature"], 
                  chat_thread_params["system_message"], chat_thread_params["auto_query_guidance"], chat_thread_params["target_output_length"], 
                  chat_thread_params["max_chat_history_tokens"], chat_thread_params["rse_params"], thread_id))
        conn.commit()
        conn.close()

        # Return the updated chat thread
        return chat_thread_params

    def delete_chat_thread(self, thread_id: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()

    def add_interaction(self, thread_id: str, interaction: dict) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        message_id = str(uuid.uuid4())
        
        # add message_id to the interaction
        interaction["message_id"] = message_id
        
        formatted_interaction = {
            "thread_id": thread_id,
            "message_id": message_id,
            "user_input": interaction["user_input"]["content"],
            "user_input_timestamp": interaction["user_input"]["timestamp"],
            "model_response": interaction["model_response"]["content"],
            "model_response_timestamp": interaction["model_response"]["timestamp"],
            "relevant_segments": json.dumps(interaction["relevant_segments"]),
            "search_queries": json.dumps(interaction["search_queries"]),
            "citations": json.dumps(interaction["model_response"].get("citations", []))
        }

        # Create the interaction
        query_statement = f"INSERT INTO interactions ({', '.join(self.interactions_columns)}) VALUES ({'?, '*(len(self.interactions_columns)-1)}?)"
        interaction_tuple = tuple([formatted_interaction[column] for column in self.interactions_columns])
        c.execute(query_statement, interaction_tuple)
        conn.commit()
        conn.close()
        return interaction

    def _check_and_migrate_db(self):
        """
        For backwards compatibility, check if the citations and rse_params columns exist in the interactions and chat_threads tables, and add them if they don't.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Check if citations column exists in interactions table
        result = c.execute("PRAGMA table_info(interactions)")
        columns = [row[1] for row in result.fetchall()]
        
        if "citations" not in columns:
            c.execute("ALTER TABLE interactions ADD COLUMN citations TEXT")
            conn.commit()
        
        # Check if rse_params column exists in chat_threads table
        result = c.execute("PRAGMA table_info(chat_threads)")
        columns = [row[1] for row in result.fetchall()]
        
        if "rse_params" not in columns:
            c.execute("ALTER TABLE chat_threads ADD COLUMN rse_params TEXT")
            conn.commit()
        
        conn.close()