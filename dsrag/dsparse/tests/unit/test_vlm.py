import os
import sys
import unittest
import json


# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from dsrag.dsparse.file_parsing.vlm import make_llm_call_gemini


class TestVLM(unittest.TestCase):
    """Test cases for VLM functionality with real API calls."""

    @unittest.skipIf('GEMINI_API_KEY' not in os.environ, "GEMINI_API_KEY not found in environment")
    def test_gemini_2_0_with_simple_schema(self):
        """Test Gemini 2.0 with a simple schema."""
        test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tests/data/page_7.jpg"))
        
        # Simple schema for testing
        test_schema = {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string"
                }
            }
        }
        
        try:
            # Make the call
            result = make_llm_call_gemini(
                image_path=test_image_path,
                system_message="Describe this image in a JSON object with a 'description' field.",
                model="gemini-2.0-flash",
                response_schema=test_schema,
                temperature=0.2
            )
                        
            # Verify we got a valid JSON response
            parsed_result = json.loads(result)
            self.assertIn('description', parsed_result)
            self.assertTrue(len(parsed_result['description']) > 0)
        except Exception as e:
            raise

    @unittest.skipIf('GEMINI_API_KEY' not in os.environ, "GEMINI_API_KEY not found in environment")
    def test_gemini_2_5_with_simple_schema(self):
        """Test Gemini 2.5 with a simple schema."""
        test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tests/data/page_7.jpg"))
        
        # Simple schema for testing
        test_schema = {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string"
                }
            }
        }
        
        try:
            # Make the call
            result = make_llm_call_gemini(
                image_path=test_image_path,
                system_message="Describe this image in a JSON object with a 'description' field.",
                model="gemini-2.5-flash-preview-04-17",
                response_schema=test_schema,
                temperature=0.2
            )
                        
            # Verify we got a valid JSON response
            parsed_result = json.loads(result)
            self.assertIn('description', parsed_result)
            self.assertTrue(len(parsed_result['description']) > 0)
        except Exception as e:
            raise

    @unittest.skipIf('GEMINI_API_KEY' not in os.environ, "GEMINI_API_KEY not found in environment")
    def test_gemini_2_0_with_complex_schema(self):
        """Test Gemini 2.0 with the VLM parsing schema."""
        test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tests/data/page_7.jpg"))
        
        # The actual complex schema used in VLM file parsing
        complex_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                    },
                    "content": {
                        "type": "string",
                    },
                },
                "required": ["type", "content"],
            },
        }
        
        # System message similar to what's used in VLM file parsing
        system_message = """
        You are a PDF -> MD file parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page. Each element must be represented using Markdown formatting.

        There are two categories of elements you need to identify: text elements and visual elements. Text elements are those that can be accurately represented using plain text. Visual elements are those that need to be represented as images to fully capture their content. For text elements, you must provide the exact text content. For visual elements, you must provide a detailed description of the content.

        Output format
        - Your output should be an ordered (from top to bottom) list of elements on the page, where each element is a dictionary with the following keys:
            - type: str - the type of the element (e.g., text, figure, title, etc.)
            - content: str - the content of the element
        """
        
        try:
            # Make the call
            result = make_llm_call_gemini(
                image_path=test_image_path,
                system_message=system_message,
                model="gemini-2.0-flash",
                response_schema=complex_schema,
                temperature=0.5
            )
                        
            # Verify we got a valid JSON response
            parsed_result = json.loads(result)
            self.assertTrue(isinstance(parsed_result, list))
            if len(parsed_result) > 0:
                self.assertIn('type', parsed_result[0])
                self.assertIn('content', parsed_result[0])
        except Exception as e:
            raise

    @unittest.skipIf('GEMINI_API_KEY' not in os.environ, "GEMINI_API_KEY not found in environment")
    def test_gemini_2_5_with_complex_schema(self):
        """Test Gemini 2.5 with the VLM parsing schema."""
        test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tests/data/page_7.jpg"))
        
        # The actual complex schema used in VLM file parsing
        complex_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                    },
                    "content": {
                        "type": "string",
                    },
                },
                "required": ["type", "content"],
            },
        }
        
        # System message similar to what's used in VLM file parsing
        system_message = """
        You are a PDF -> MD file parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page. Each element must be represented using Markdown formatting.

        There are two categories of elements you need to identify: text elements and visual elements. Text elements are those that can be accurately represented using plain text. Visual elements are those that need to be represented as images to fully capture their content. For text elements, you must provide the exact text content. For visual elements, you must provide a detailed description of the content.

        Output format
        - Your output should be an ordered (from top to bottom) list of elements on the page, where each element is a dictionary with the following keys:
            - type: str - the type of the element (e.g., text, figure, title, etc.)
            - content: str - the content of the element
        """
        
        try:
            # Make the call
            result = make_llm_call_gemini(
                image_path=test_image_path,
                system_message=system_message,
                model="gemini-2.5-flash-preview-04-17",
                response_schema=complex_schema,
                temperature=0.5
            )
                        
            # Verify we got a valid JSON response
            parsed_result = json.loads(result)
            self.assertTrue(isinstance(parsed_result, list))
            if len(parsed_result) > 0:
                self.assertIn('type', parsed_result[0])
                self.assertIn('content', parsed_result[0])
        except Exception as e:
            raise


if __name__ == "__main__":
    unittest.main()