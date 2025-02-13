import unittest
import os
import base64
import requests
from pydantic import BaseModel

import os
import sys
sys.path.append(os.path.abspath("/Users/zach/Code/dsRAG"))

from dsrag.utils.llm import get_response


class TestLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Download and encode test image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/5/53/Sheba1.JPG"
        cls.image_data = base64.b64encode(requests.get(image_url).content).decode('utf-8')
        
        # Test configurations
        cls.test_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.0-flash"
        }
        
        # Check API keys
        cls.skip_tests = {
            "openai": "OPENAI_API_KEY" not in os.environ,
            "anthropic": "ANTHROPIC_API_KEY" not in os.environ,
            "gemini": "GEMINI_API_KEY" not in os.environ
        }

    def test_text_prompt(self):
        """Test basic text prompt with all providers"""
        for provider, model in self.test_models.items():
            with self.subTest(provider=provider):
                if self.skip_tests[provider]:
                    self.skipTest(f"Missing {provider.upper()} API key")
                
                result = get_response(
                    prompt="What is the capital of France?",
                    model_name=model
                )
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 5)
                self.assertIn("Paris", result)

    def test_multimodal_input(self):
        """Test image+text input with all providers"""
        messages = [{
            "role": "user",
            "content": [
                "What's shown in this image?",
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": self.image_data
                    }
                }
            ]
        }]

        for provider, model in self.test_models.items():
            with self.subTest(provider=provider):
                if self.skip_tests[provider]:
                    self.skipTest(f"Missing {provider.upper()} API key")
                
                result = get_response(
                    messages=messages,
                    model_name=model
                )
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 10)
                self.assertIn("cat", result.lower())

    def test_structured_output(self):
        """Test structured output with response model"""
        test_cases = [
            ("gpt-4o-mini", "Extract name and age from the following text: John is 30 years old"),
            ("claude-3-5-sonnet-20241022", "Extract name and age from the following text: Sarah, age 28"),
            ("gemini-2.0-flash", "Extract name and age from the following text: Mike, 42 years old")
        ]

        class UserModel(BaseModel):
            name: str
            age: int

        for model_name, prompt in test_cases:
            with self.subTest(model=model_name):
                provider = next(k for k,v in self.test_models.items() if v == model_name)
                if self.skip_tests[provider]:
                    self.skipTest(f"Missing {provider.upper()} API key")
                
                result = get_response(
                    prompt=prompt,
                    model_name=model_name,
                    response_model=UserModel
                )
                self.assertIsInstance(result, UserModel)
                self.assertGreater(result.age, 0)
                self.assertGreater(len(result.name), 1)

    def test_messages_vs_prompt(self):
        """Test both messages and prompt handling"""
        # Test messages priority
        result = get_response(
            messages=[{"role": "user", "content": "2+2"}],
            prompt="3+3",
            model_name=self.test_models["openai"]
        )
        self.assertIn("4", result)

        # Test prompt conversion
        result = get_response(
            prompt="2+2",
            model_name=self.test_models["openai"]
        )
        self.assertIn("4", result)

if __name__ == '__main__':
    unittest.main() 