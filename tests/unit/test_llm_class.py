import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.llm import OpenAIChatAPI, AnthropicChatAPI, OllamaAPI, LLM, GeminiAPI


class TestLLM(unittest.TestCase):
    def test__openai_chat_api(self):
        chat_api = OpenAIChatAPI()
        chat_messages = [
            {"role": "user", "content": "Hello, how are you?"},
        ]
        response = chat_api.make_llm_call(chat_messages)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test__anthropic_chat_api(self):
        chat_api = AnthropicChatAPI()
        chat_messages = [
            {"role": "user", "content": "What's the weather like today?"},
        ]
        response = chat_api.make_llm_call(chat_messages)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test__ollama_api(self):
        try:
            chat_api = OllamaAPI()
        except Exception as e:
            print(e)
            if e.__class__.__name__ == "ConnectError":
                print ("Connection failed")
            return
        chat_messages = [
            {"role": "user", "content": "Who's your daddy?"},
        ]
        response = chat_api.make_llm_call(chat_messages)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test__gemini_api(self):
        try:
            chat_api = GeminiAPI()
        except ValueError as e:
            # Skip test if API key is not set
            if "GEMINI_API_KEY environment variable not set" in str(e):
                print(f"Skipping Gemini test: {e}")
                return
            else:
                raise # Re-raise other ValueErrors
        except Exception as e:
            # Handle other potential exceptions during init (e.g., genai configuration issues)
            print(f"Skipping Gemini test due to initialization error: {e}")
            return

        chat_messages = [
            {"role": "user", "content": "Explain the concept of RAG briefly."},
        ]
        try:
            response = chat_api.make_llm_call(chat_messages)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            # Catch potential API call errors (e.g., connection, authentication within genai)
            self.fail(f"GeminiAPI make_llm_call failed with exception: {e}")

    def test__save_and_load_from_dict(self):
        chat_api = OpenAIChatAPI(temperature=0.5, max_tokens=2000)
        config = chat_api.to_dict()
        chat_api_loaded = LLM.from_dict(config)
        self.assertEqual(chat_api_loaded.model, chat_api.model)
        self.assertEqual(chat_api_loaded.temperature, chat_api.temperature)
        self.assertEqual(chat_api_loaded.max_tokens, chat_api.max_tokens)

if __name__ == "__main__":
    unittest.main()