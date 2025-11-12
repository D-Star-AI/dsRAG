from abc import ABC, abstractmethod
from dsrag.chat.chat_types import ChatThreadParams

class ChatThreadDB(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            "subclass_name": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config) -> "ChatThreadDB":
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
            'BasicChatThreadDB': 'dsrag.database.chat_thread.basic_db',
            'SQLiteChatThreadDB': 'dsrag.database.chat_thread.sqlite_db'
        }
        
        if class_name in import_map:
            module_path = import_map[class_name]
            try:
                __import__(module_path)
            except ImportError as e:
                raise ImportError(f"Failed to import {module_path} for {class_name}: {e}")

    @abstractmethod
    def create_chat_thread(self, chat_thread_params: ChatThreadParams) -> dict:
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
        
    @abstractmethod
    def update_interaction(self, thread_id: str, message_id: str, interaction_update: dict) -> dict:
        """
        Updates an existing interaction in a chat thread.
        Only updates the fields provided in interaction_update.
        """
        pass