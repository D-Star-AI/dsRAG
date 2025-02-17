from abc import ABC, abstractmethod

class ChatThreadDB(ABC):
    @abstractmethod
    def create_chat_thread(self, chat_thread_params: dict) -> dict:
        """
        Creates a chat thread in the database.
        """
        pass

    @abstractmethod
    def list_chat_threads(self) -> list[dict]:
        """
        Lists all chat threads in the database.
        """
        pass

    @abstractmethod
    def get_chat_thread(self, thread_id: str) -> dict:
        """
        Gets a chat thread by ID.
        """
        pass

    @abstractmethod
    def update_chat_thread(self, thread_id: str, chat_thread_params: dict) -> dict:
        """
        Updates a chat thread by ID.
        """
        pass

    @abstractmethod
    def delete_chat_thread(self, thread_id: str) -> dict:
        """
        Deletes a chat thread by ID.
        """
        pass

    @abstractmethod
    def add_interaction(self, thread_id: str, interaction: dict) -> dict:
        """
        Adds an interaction to a chat thread.
        """
        pass