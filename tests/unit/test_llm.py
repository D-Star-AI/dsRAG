import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sprag.llm import OpenAIChatAPI, AnthropicChatAPI, LLM

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

def test_save_and_load_from_dict():
    chat_api = OpenAIChatAPI(temperature=0.5, max_tokens=2000)
    config = chat_api.to_dict()
    chat_api_loaded = LLM.from_dict(config)
    assert chat_api_loaded.model == chat_api.model, "Loaded model should be the same as the original model"
    assert chat_api_loaded.temperature == chat_api.temperature, "Loaded temperature should be the same as the original temperature"
    assert chat_api_loaded.max_tokens == chat_api.max_tokens, "Loaded max_tokens should be the same as the original max_tokens"

if __name__ == "__main__":
    test_openai_chat_api()
    test_anthropic_chat_api()
    test_save_and_load_from_dict()