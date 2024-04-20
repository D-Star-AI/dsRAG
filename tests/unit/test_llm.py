import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sprag.llm import OpenAIChatAPI, AnthropicChatAPI

def test_openai_chat_api():
    chat_api = OpenAIChatAPI()
    chat_messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    response = chat_api.make_llm_call(chat_messages)
    assert isinstance(response, str) and len(response) > 0, "Response should be a non-empty string"

def test_anthropic_chat_api():
    chat_api = AnthropicChatAPI()
    chat_messages = [
        {"role": "user", "content": "What's the weather like today?"},
    ]
    response = chat_api.make_llm_call(chat_messages)
    assert isinstance(response, str) and len(response) > 0, "Response should be a non-empty string"

if __name__ == "__main__":
    test_openai_chat_api()
    test_anthropic_chat_api()