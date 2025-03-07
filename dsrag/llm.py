from abc import ABC, abstractmethod
from typing import Optional
import os
from dsrag.utils.imports import openai, anthropic, ollama


class LLM(ABC):
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
    def make_llm_call(self, chat_messages: list[dict]) -> str:
        """
        Takes in chat_messages (OpenAI format) and returns the response from the LLM as a string.
        """
        pass

class OpenAIChatAPI(LLM):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def make_llm_call(self, chat_messages: list[dict]) -> str:
        base_url = os.environ.get("DSRAG_OPENAI_BASE_URL", None)
        if base_url is not None:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        else:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        llm_output = response.choices[0].message.content.strip()
        return llm_output
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        })
        return base_dict

class AnthropicChatAPI(LLM):
    def __init__(self, model: str = "claude-3-haiku-20240307", temperature: float = 0.2, max_tokens: int = 1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def make_llm_call(self, chat_messages: list[dict]) -> str:
        base_url = os.environ.get("DSRAG_ANTHROPIC_BASE_URL", None)
        if base_url is not None:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=base_url)
        else:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        system_message = ""
        num_system_messages = 0
        normal_chat_messages = []
        for message in chat_messages:
            if message["role"] == "system":
                if num_system_messages > 0:
                    raise ValueError("ERROR: more than one system message detected")
                system_message = message["content"]
                num_system_messages += 1
            else:
                normal_chat_messages.append(message)

        message = client.messages.create(
            system=system_message,
            messages=normal_chat_messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return message.content[0].text
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        })
        return base_dict

class OllamaAPI(LLM):
    def __init__(
        self, model: str = "llama3", temperature: float = 0.2, max_tokens: int = 1000, client: "ollama.Client" = None
    ):
        self.client = client or ollama.Client()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client.pull(self.model)

    def make_llm_call(self, chat_messages: list[dict]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=chat_messages,
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return response["message"]["content"].strip()

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        return base_dict
