from abc import ABC, abstractmethod
import os

class LLM(ABC):
    @abstractmethod
    def make_llm_call(self, chat_messages: list[dict]) -> str:
        """
        Takes in chat_messages (OpenAI format) and returns the response from the LLM as a string.
        """
        pass

class OpenAIChatAPI(LLM):
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.2, max_tokens: int = 1000):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def make_llm_call(self, chat_messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        llm_output = response.choices[0].message.content.strip()
        return llm_output

class AnthropicChatAPI(LLM):
    def __init__(self, model_name: str = "claude-3-haiku-20240307", temperature: float = 0.2, max_tokens: int = 1000):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def make_llm_call(self, chat_messages: list[dict]) -> str:
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

        message = self.client.messages.create(
            system=system_message,
            messages=normal_chat_messages,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return message.content[0].text