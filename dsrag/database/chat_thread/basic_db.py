import json
from .db import ChatThreadDB
import uuid

class BasicChatThreadDB(ChatThreadDB):
    def __init__(self):
        try:
            with open("chat_thread_db.json", "r") as f:
                self.chat_threads = json.load(f)
        except FileNotFoundError:
            self.chat_threads = {}

    def create_chat_thread(self, chat_thread_params: dict) -> dict:
        """
        chat_thread_params: dict with the following keys:
        - thread_id: str
        - kb_ids: list[str]
        - model: str
        - title: str
        - temperature: float
        - system_message: str
        - auto_query_guidance: str
        - target_output_length: str
        - max_chat_history_tokens: int

        Returns
        chat_thread: dict with the following keys:
        - params: dict
        - interactions: list[dict]
        """
        
        chat_thread = {
            "params": chat_thread_params,
            "interactions": []
        }
        
        self.chat_threads[chat_thread_params["thread_id"]] = chat_thread
        self.save()
        return chat_thread
    
    def list_chat_threads(self) -> list[dict]:
        return [self.chat_threads[thread_id] for thread_id in self.chat_threads]
    
    def get_chat_thread(self, thread_id: str) -> dict:
        return self.chat_threads.get(thread_id)
    
    def update_chat_thread(self, thread_id: str, chat_thread_params: dict) -> dict:
        self.chat_threads[thread_id]["params"] = chat_thread_params
        self.save()
        return chat_thread_params
    
    def delete_chat_thread(self, thread_id: str) -> dict:
        chat_thread = self.chat_threads.pop(thread_id, None)
        self.save()
        return chat_thread
        
    def add_interaction(self, thread_id: str, interaction: dict) -> dict:
        """
        interaction should be a dict with the following keys:
        - user_input: dict
            - content: str
            - timestamp: str
        - model_response: dict
            - content: str
            - citations: list[dict]
                - doc_id: str
                - page_numbers: list[int]
                - cited_text: str
            - timestamp: str
            - status: str (pending, streaming, highlighting_citations, finished, failed)
        - relevant_segments: list[dict]
            - text: str
            - doc_id: str
            - kb_id: str
        - search_queries: list[dict]
            - query: str
            - kb_id: str
        """
        message_id = str(uuid.uuid4())
        interaction["message_id"] = message_id
        
        # Set default status if not provided
        if "model_response" in interaction and "status" not in interaction["model_response"]:
            interaction["model_response"]["status"] = "pending"
            
        self.chat_threads[thread_id]["interactions"].append(interaction)
        self.save()
        return interaction
        
    def update_interaction(self, thread_id: str, message_id: str, interaction_update: dict) -> dict:
        """
        Updates an existing interaction in a chat thread.
        Only updates the fields provided in interaction_update.
        """
        # Find the interaction with the matching message_id
        for i, interaction in enumerate(self.chat_threads[thread_id]["interactions"]):
            if interaction.get("message_id") == message_id:
                # Update only the fields provided in interaction_update
                if "model_response" in interaction_update:
                    interaction["model_response"].update(interaction_update["model_response"])
                    
                    # Ensure status field exists (for backward compatibility)
                    if "status" not in interaction["model_response"]:
                        interaction["model_response"]["status"] = "finished"
                
                # Save the changes
                self.save()
                return {"message_id": message_id, "updated": True}
        
        return {"message_id": message_id, "updated": False, "error": "Interaction not found"}
    
    def save(self):
        """
        Save the ChatThreadDB to a JSON file.
        """
        with open("chat_thread_db.json", "w") as f:
            json.dump(self.chat_threads, f)

    def load(self):
        with open("chat_thread_db.json", "r") as f:
            self.chat_threads = json.load(f)