import os
import sys
import unittest

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from dsrag.dsparse.file_parsing.vlm_clients import VLM, GeminiVLM, VertexAIVLM


class TestVLMClients(unittest.TestCase):
    def test_registry_contains_subclasses(self):
        self.assertIn("GeminiVLM", VLM._subclasses)
        self.assertIn("VertexAIVLM", VLM._subclasses)
        self.assertIs(VLM._subclasses["GeminiVLM"], GeminiVLM)
        self.assertIs(VLM._subclasses["VertexAIVLM"], VertexAIVLM)

    def test_gemini_to_from_dict_roundtrip(self):
        client = GeminiVLM()
        as_dict = client.to_dict()
        self.assertEqual(as_dict.get("subclass_name"), "GeminiVLM")
        self.assertIn("model", as_dict)
        rebuilt = VLM.from_dict(as_dict)
        self.assertIsInstance(rebuilt, GeminiVLM)
        self.assertEqual(rebuilt.model, client.model)

    def test_vertex_to_from_dict_roundtrip(self):
        client = VertexAIVLM(model="gemini-1.5-flash", project_id="proj", location="us-central1")
        as_dict = client.to_dict()
        self.assertEqual(as_dict.get("subclass_name"), "VertexAIVLM")
        self.assertEqual(as_dict.get("model"), "gemini-1.5-flash")
        self.assertEqual(as_dict.get("project_id"), "proj")
        self.assertEqual(as_dict.get("location"), "us-central1")
        rebuilt = VLM.from_dict(as_dict)
        self.assertIsInstance(rebuilt, VertexAIVLM)
        self.assertEqual(rebuilt.model, client.model)
        self.assertEqual(rebuilt.project_id, client.project_id)
        self.assertEqual(rebuilt.location, client.location)


if __name__ == "__main__":
    unittest.main()
