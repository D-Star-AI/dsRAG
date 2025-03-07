import unittest
import os
import base64
from pydantic import BaseModel, Field
from typing import Optional
from dsrag.utils.llm import get_response


class TestLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load test image relative to test file location
        image_path = os.path.join(os.path.dirname(__file__), "..", "data", "page_7.png")
        with open(image_path, "rb") as f:
            cls.image_data = base64.b64encode(f.read()).decode('utf-8')
        
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
                        "media_type": "image/png",
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
                self.assertTrue(
                    any(term in result.lower() for term in ["machine learning", "llm", "definitions"]),
                    f"Response did not contain expected AI/ML terms: {result}"
                )

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

    def test_anthropic_system_message_instructor(self):
        """Test that system messages work with Anthropic when using Instructor"""
        if self.skip_tests["anthropic"]:
            self.skipTest("Missing ANTHROPIC API key")
        
        class Response(BaseModel):
            tone: str = Field(..., description="The tone of the response - must be either 'formal' or 'casual'")
            message: str
        
        system_message = "You are a very formal butler. Always respond in an extremely formal tone. You MUST begin each response with 'Good day, sir!'"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Say hi and tell me the weather"}
        ]
        
        result = get_response(
            messages=messages,
            model_name=self.test_models["anthropic"],
            response_model=Response
        )
        
        self.assertIsInstance(result, Response)
        self.assertEqual(result.tone, "formal")

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

    def test_streaming_text(self):
        """Test streaming text responses with all providers"""
        for provider, model in self.test_models.items():
            with self.subTest(provider=provider):
                if self.skip_tests[provider]:
                    self.skipTest(f"Missing {provider.upper()} API key")
                
                # Create a counter to track streaming chunks
                chunks = []
                accumulated_text = ""
                
                # Request a streaming response with a prompt that should generate longer output
                stream = get_response(
                    prompt="Explain what are the three primary colors.",
                    model_name=model,
                    stream=True
                )
                
                # Gather streaming chunks
                for chunk in stream:
                    chunks.append(chunk)
                    accumulated_text += chunk
                
                # Verify streaming returned something
                self.assertGreater(len(chunks), 0, f"No streaming chunks for {provider}")
                self.assertIsInstance(accumulated_text, str)
                self.assertGreater(len(accumulated_text), 5)
                
                # Log chunk count (helpful for debugging)
                print(f"{provider} streaming returned {len(chunks)} chunks")
                
                # Check for primary color terms in the response
                primary_colors = ["red", "blue", "yellow"]
                found_count = sum(1 for color in primary_colors if color.lower() in accumulated_text.lower())
                self.assertGreater(found_count, 0, f"No primary colors found in response: {accumulated_text}")

    def test_streaming_structured(self):
        """Test streaming structured outputs with all providers"""
        # No need to import citation types since we're using a custom model
        
        # Simple structured output test - doesn't require citations when no sources exist
        class CityInfo(BaseModel):
            city: str = Field(..., description="Name of the city")
            country: str = Field(..., description="Country where the city is located")
            population: Optional[int] = Field(None, description="Approximate population")
        
        test_cases = [
            ("gpt-4o-mini", "Tell me about Paris", CityInfo),
            ("claude-3-5-sonnet-20241022", "Tell me about Rome", CityInfo),
            ("gemini-2.0-flash", "Tell me about Tokyo", CityInfo)
        ]
        
        for model_name, prompt, model_class in test_cases:
            with self.subTest(model=model_name):
                provider = next(k for k,v in self.test_models.items() if v == model_name)
                if self.skip_tests[provider]:
                    self.skipTest(f"Missing {provider.upper()} API key")
                
                # Collect streaming response chunks
                chunks = []
                final_chunk = None
                
                # Request streaming structured response
                stream = get_response(
                    prompt=prompt,
                    model_name=model_name,
                    response_model=model_class,
                    stream=True
                )
                
                # Process streaming chunks
                for chunk in stream:
                    chunks.append(chunk)
                    final_chunk = chunk
                
                # Verify streaming returned something
                self.assertGreater(len(chunks), 0, f"No structured streaming chunks for {provider}")
                # Log chunk count (helpful for debugging)
                print(f"{provider} structured streaming returned {len(chunks)} chunks")
                
                # Check final response has expected structure
                if isinstance(final_chunk, dict):
                    self.assertIn('city', final_chunk, f"Final chunk missing 'city' field: {final_chunk}")
                    self.assertIn('country', final_chunk, f"Final chunk missing 'country' field: {final_chunk}")
                else:
                    # For pydantic models, check attributes
                    self.assertTrue(hasattr(final_chunk, 'city'), f"Final chunk missing 'city': {final_chunk}")
                    self.assertTrue(hasattr(final_chunk, 'country'), f"Final chunk missing 'country': {final_chunk}")
                    
                # For a more rigorous test, try to access the city and country fields
                try:
                    city = final_chunk.city if hasattr(final_chunk, 'city') else final_chunk.get('city')
                    country = final_chunk.country if hasattr(final_chunk, 'country') else final_chunk.get('country')
                    print(f"{provider} extracted: {city}, {country}")
                except Exception as e:
                    self.fail(f"Failed to access city/country fields: {str(e)}")

if __name__ == '__main__':
    unittest.main() 